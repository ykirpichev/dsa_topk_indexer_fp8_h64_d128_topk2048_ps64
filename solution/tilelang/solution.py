"""DSA FP8 top-K indexer, TileLang implementation.

Same dispatch as solution/triton and solution/python. config.toml still
points at the CUDA submission; this file is a side-by-side port.

  max_num_pages <= 32   Fast path. The answer is the block-table transform of
                        [0, seq_len). Q, K and the weights are never read.
  otherwise             Stage 1 scores two pages per CTA with one FP8 GEMM.
                        Stage 2 is a 5+5+6 radix select.

The paged K address does not vectorize to 16 bytes inside this frontend: a
16-byte load comes back as a vector that cannot be indexed, and storing that
vector into the shared tile would skip the swizzle the GEMM reads. Each token
is therefore loaded as four 4-byte words, viewed as FP8, and stored one
element at a time.
"""

import collections

import tilelang
import tilelang.language as T
import torch

H = 64
D = 128
PAGE = 64
TOPK = 2048
PAGE_BYTES = 64 * 132          # 8192 FP8 bytes, then 64 FP32 scales
SCALE_FLOATS = PAGE_BYTES // 4
SCALE_OFFSET = (64 * 128) // 4
TILE = 128                     # 2 pages, one UMMA-M tile
PAGES_PER_TILE = 2
FAST_MAX_PAGES = 32
STAGE2_THREADS = 512
STAGE2_SCAN_STEPS = 9          # log2(STAGE2_THREADS)
STAGE2_MAX_BLOCK = 16384
MAX_GRAPHS = 32

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(target="cuda", pass_configs=_PASS)
def _fast_path(seq_lens, block_table, out):
    """out[b] = block-table transform of [0, seq_len), -1 past seq_len."""
    B = T.dynamic("B")
    max_pages = T.dynamic("max_pages")
    seq_lens: T.Tensor((B,), T.int32)
    block_table: T.Tensor((B, max_pages), T.int32)
    out: T.Tensor((B, TOPK), T.int32)

    with T.Kernel(B, threads=256) as b:
        seq = seq_lens[b]
        for n in T.Parallel(TOPK):
            # seq_len never reaches a page the table does not hold.
            page = block_table[b, T.min(n // PAGE, max_pages - 1)]
            phys = page * PAGE + n % PAGE
            out[b, n] = T.if_then_else(n < seq, phys, -1)


@tilelang.jit(target="cuda", pass_configs=_PASS)
def _stage1(q, k, kscale, w, seq_lens, block_table, logits):
    """FP16 logits. One program scores one M=128 tile (2 pages) of row b."""
    B = T.dynamic("B")
    k_len = T.dynamic("k_len")
    scale_len = T.dynamic("scale_len")
    max_pages = T.dynamic("max_pages")
    row_stride = T.dynamic("row_stride")

    q: T.Tensor((B, H, D), T.float8_e4m3fn)
    k: T.Tensor((k_len,), T.float8_e4m3fn)
    kscale: T.Tensor((scale_len,), T.float32)
    w: T.Tensor((B, H), T.float32)
    seq_lens: T.Tensor((B,), T.int32)
    block_table: T.Tensor((B, max_pages), T.int32)
    logits: T.Tensor((B, row_stride), T.float16)

    num_tiles = T.ceildiv(max_pages, PAGES_PER_TILE)
    with T.Kernel(num_tiles, B, threads=128) as (tile, b):
        q_s = T.alloc_shared((H, D), T.float8_e4m3fn)
        w_s = T.alloc_shared((H,), T.float32)
        k_s = T.alloc_shared((TILE, D), T.float8_e4m3fn)
        scale_s = T.alloc_shared((TILE,), T.float32)
        c_s = T.alloc_shared((TILE, H), T.float32)
        acc = T.alloc_shared((TILE,), T.float32)
        c_local = T.alloc_fragment((TILE, H), T.float32)
        # One uint32x4 holds a 16-byte gather. Viewing it as FP8 lets the
        # shared store stay element-wise, which is what applies the GEMM swizzle.
        words = T.alloc_local((4,), "uint32")
        gathered = T.view(words, (16,), "float8_e4m3fn")

        seq = seq_lens[b]
        # Rows that fit in top-K are emitted from the block table in stage 2.
        last = T.if_then_else(seq > TOPK, T.ceildiv(seq, TILE), 0)
        if tile < last:
            last_page = T.max((seq - 1) // PAGE, 0)
            tx = T.get_thread_binding()
            # 128 threads cover a row's 8 chunks of 16 bytes. ldg32 is the
            # widest load whose result is a scalar we can bitcast; a uint32x4
            # from ldg128 cannot be indexed in this frontend.
            for j in T.serial(4):
                vid = tx + j * 128
                h = vid // 8
                col = (vid % 8) * 16
                for part in T.serial(4):
                    words[part] = T.ldg32(q[b, h, col + part * 4])
                for e in T.serial(16):
                    q_s[h, col + e] = gathered[e]
            for j in T.serial(8):
                vid = tx + j * 128
                i = vid // 8
                col = (vid % 8) * 16
                tok = tile * TILE + i
                slot = T.min(tok // PAGE, last_page)
                page = T.cast(block_table[b, slot], "int64")
                row = tok % PAGE
                base = page * PAGE_BYTES + row * D + col
                for part in T.serial(4):
                    words[part] = T.ldg32(k[base + part * 4])
                for e in T.serial(16):
                    k_s[i, col + e] = gathered[e]
            for h in T.Parallel(H):
                w_s[h] = w[b, h]
            for i in T.Parallel(TILE):
                tok = tile * TILE + i
                slot = T.min(tok // PAGE, last_page)
                page = T.cast(block_table[b, slot], "int64")
                row = tok % PAGE
                scale_s[i] = kscale[page * SCALE_FLOATS + SCALE_OFFSET + row]
            T.sync_threads()

            T.clear(c_local)
            T.gemm(k_s, q_s, c_local, transpose_A=False, transpose_B=True)
            T.copy(c_local, c_s)
            for i, h in T.Parallel(TILE, H):
                c_s[i, h] = T.max(c_s[i, h], 0) * w_s[h]
            T.reduce_sum(c_s, acc, dim=1)
            T.sync_threads()
            for i in T.Parallel(TILE):
                tok = tile * TILE + i
                if tok < seq:
                    logits[b, tok] = T.cast(acc[i] * scale_s[i], "float16")


@tilelang.jit(target="cuda", pass_configs=_PASS)
def _stage2(bits, seq_lens, block_table, out, BLOCK):
    """Top-2048 of one row as physical slot ids. BLOCK is the padded row length."""
    B = T.dynamic("B")
    stride = T.dynamic("stride")
    max_pages = T.dynamic("max_pages")
    bits: T.Tensor((B, stride), T.int16)
    seq_lens: T.Tensor((B,), T.int32)
    block_table: T.Tensor((B, max_pages), T.int32)
    out: T.Tensor((B, TOPK), T.int32)

    elems = BLOCK // STAGE2_THREADS
    with T.Kernel(B, threads=STAGE2_THREADS) as b:
        keys = T.alloc_shared((BLOCK,), T.int32)
        hist = T.alloc_shared((64,), T.int32)
        partial_gt = T.alloc_shared((STAGE2_THREADS,), T.int32)
        partial_eq = T.alloc_shared((STAGE2_THREADS,), T.int32)
        count_gt = T.alloc_shared((STAGE2_THREADS,), T.int32)
        count_eq = T.alloc_shared((STAGE2_THREADS,), T.int32)
        tmp = T.alloc_shared((STAGE2_THREADS,), T.int32)
        pivot = T.alloc_shared((1,), T.int32)
        above = T.alloc_shared((1,), T.int32)
        found_digit = T.alloc_shared((1,), T.int32)
        found_above = T.alloc_shared((1,), T.int32)
        run = T.alloc_shared((1,), T.int32)

        tx = T.get_thread_binding()
        seq = seq_lens[b]

        # Short rows never look at logits: the whole row is the answer.
        for n in T.Parallel(TOPK):
            page = block_table[b, T.min(n // PAGE, max_pages - 1)]
            phys = page * PAGE + n % PAGE
            out[b, n] = T.if_then_else(n < seq, phys, -1)

        if seq > TOPK:
            base = tx * elems
            for j in T.serial(elems):
                idx = base + j
                src = T.min(idx, stride - 1)
                raw = T.cast(bits[b, src], "int32") & 0xFFFF
                # FP16 bits reordered so unsigned order matches float order.
                key = T.if_then_else(raw >= 0x8000, 0xFFFF - raw, raw + 0x8000)
                keys[idx] = T.if_then_else(idx < seq, key, -1)
            if tx == 0:
                pivot[0] = 0
                above[0] = 0
            T.sync_threads()

            # 5 + 5 + 6 bits from the top. Same split as the Triton kernel:
            # three narrow histograms rather than two 8-bit ones.
            rounds = ((11, 31, 11, 32), (6, 31, 6, 32), (0, 63, 0, 64))
            for shift, mask, place, nbins in rounds:
                T.clear(hist)
                T.sync_threads()
                for j in T.serial(elems):
                    k = keys[base + j]
                    if shift == 11:
                        pred = k >= 0
                        digit = k >> 11
                    elif shift == 6:
                        pred = (k >> 11) == (pivot[0] >> 11)
                        digit = (k >> 6) & mask
                    else:
                        pred = (k >> 6) == (pivot[0] >> 6)
                        digit = k & mask
                    if pred:
                        T.atomic_add(hist[digit], 1)
                T.sync_threads()
                if tx == 0:
                    run[0] = 0
                    found_digit[0] = 0
                    found_above[0] = 0
                    need = TOPK - above[0]
                    for bin_i in T.serial(nbins):
                        bin_id = nbins - 1 - bin_i
                        c = hist[bin_id]
                        if run[0] < need:
                            if run[0] + c >= need:
                                found_digit[0] = bin_id
                                found_above[0] = run[0]
                        run[0] = run[0] + c
                    pivot[0] = pivot[0] | (found_digit[0] << place)
                    above[0] = above[0] + found_above[0]
                T.sync_threads()

            # Count keys strictly above the pivot, and ties, per thread.
            pvt = pivot[0]
            partial_gt[tx] = 0
            partial_eq[tx] = 0
            for j in T.serial(elems):
                k = keys[base + j]
                if k > pvt:
                    partial_gt[tx] = partial_gt[tx] + 1
                elif k == pvt:
                    partial_eq[tx] = partial_eq[tx] + 1
            count_gt[tx] = partial_gt[tx]
            count_eq[tx] = partial_eq[tx]
            T.sync_threads()

            # Inclusive scan of the per-thread counts. `tmp` is the double buffer.
            for step in T.serial(STAGE2_SCAN_STEPS):
                offset = 1 << step
                for i in T.Parallel(STAGE2_THREADS):
                    extra = T.if_then_else(i >= offset, partial_gt[i - offset], 0)
                    tmp[i] = partial_gt[i] + extra
                T.sync_threads()
                for i in T.Parallel(STAGE2_THREADS):
                    partial_gt[i] = tmp[i]
                T.sync_threads()
            for step in T.serial(STAGE2_SCAN_STEPS):
                offset = 1 << step
                for i in T.Parallel(STAGE2_THREADS):
                    extra = T.if_then_else(i >= offset, partial_eq[i - offset], 0)
                    tmp[i] = partial_eq[i] + extra
                T.sync_threads()
                for i in T.Parallel(STAGE2_THREADS):
                    partial_eq[i] = tmp[i]
                T.sync_threads()

            # Exclusive start of this thread's chunk, then a serial emit.
            tmp[tx] = partial_gt[tx] - count_gt[tx]
            count_gt[tx] = above[0] + (partial_eq[tx] - count_eq[tx])
            for j in T.serial(elems):
                idx = base + j
                k = keys[idx]
                if k > pvt:
                    page = block_table[b, idx // PAGE]
                    out[b, tmp[tx]] = page * PAGE + idx % PAGE
                    tmp[tx] = tmp[tx] + 1
                elif k == pvt:
                    if count_gt[tx] < TOPK:
                        page = block_table[b, idx // PAGE]
                        out[b, count_gt[tx]] = page * PAGE + idx % PAGE
                        count_gt[tx] = count_gt[tx] + 1


# --- Host side ---------------------------------------------------------------

_scratch = {}
_streams = {}
_graphs = collections.OrderedDict()


def _ceil_div(a, b):
    return (a + b - 1) // b


def _row_stride(max_num_pages):
    """Logits per row: one M=128 tile for every two pages."""
    return _ceil_div(max_num_pages, PAGES_PER_TILE) * TILE


def _logits_scratch(device, count):
    buf = _scratch.get(device.index)
    if buf is None or buf.numel() < count:
        grown = 2 * buf.numel() if buf is not None else 0
        buf = torch.empty(max(count, grown), dtype=torch.float16, device=device)
        _scratch[device.index] = buf
    return buf


def _topk_torch(logits, seq_lens, block_table, out):
    """Stage 2 for rows longer than one TileLang block."""
    pos = torch.arange(logits.shape[1], device=logits.device)[None, :]
    score = torch.where(seq_lens[:, None] <= TOPK, -pos.float(), logits.float())
    score = score.masked_fill(pos >= seq_lens[:, None], float("-inf"))
    top = score.topk(TOPK, dim=1)
    page = PAGE
    phys = block_table.gather(1, top.indices // page) * page + top.indices % page
    out.copy_(torch.where(torch.isinf(top.values), -1, phys))


def _replay(key, device, launch):
    """Run `launch` through a cached CUDA graph, capturing on the first call."""
    if key not in _graphs:
        launch()
        try:
            stream = _streams.get(device.index)
            if stream is None:
                stream = _streams[device.index] = torch.cuda.Stream(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch()
        except Exception:
            graph = None
        _graphs[key] = graph
        if len(_graphs) > MAX_GRAPHS:
            _graphs.popitem(last=False)
        return
    graph = _graphs[key]
    _graphs.move_to_end(key)
    if graph is None:
        launch()
        return
    stream = _streams[device.index]
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    graph.replay()
    current.wait_stream(stream)


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    batch, max_num_pages = block_table.shape
    device = block_table.device
    if max_num_pages <= FAST_MAX_PAGES:
        def fast():
            _fast_path(seq_lens, block_table, topk_indices)
        key = (seq_lens.data_ptr(), block_table.data_ptr(), topk_indices.data_ptr(),
               batch, max_num_pages)
        _replay(key, device, fast)
        return

    row_stride = _row_stride(max_num_pages)
    scratch = _logits_scratch(device, batch * row_stride)
    logits = scratch[: batch * row_stride].view(batch, row_stride)
    k_fp8 = k_index_cache_fp8.view(torch.float8_e4m3fn).reshape(-1)
    k_scale = k_index_cache_fp8.view(torch.float32).reshape(-1)

    def stage1():
        _stage1(q_index_fp8, k_fp8, k_scale, weights, seq_lens, block_table, logits)

    block = 1 << (row_stride - 1).bit_length()
    if block > STAGE2_MAX_BLOCK:
        stage1()
        _topk_torch(logits, seq_lens, block_table, topk_indices)
        return

    bits = logits.view(torch.int16)

    def launch():
        stage1()
        _stage2(bits, seq_lens, block_table, topk_indices, block)

    key = (q_index_fp8.data_ptr(), k_index_cache_fp8.data_ptr(), weights.data_ptr(),
           seq_lens.data_ptr(), block_table.data_ptr(), topk_indices.data_ptr(),
           scratch.data_ptr(), batch, max_num_pages)
    _replay(key, device, launch)
