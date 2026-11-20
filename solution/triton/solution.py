"""DSA FP8 top-K indexer, Triton implementation.

Same algorithm and dispatch as the CUDA implementation in ../python, written so
the two can be benchmarked side by side (scripts/run_modal_triton_vs_cuda.py):

  max_num_pages <= 32  Fast path. The whole context fits in top-K, so the
                       answer is the block-table transform of [0, seq_len)
                       and Q, K and weights are never read.
  otherwise            Stage 1 scores every token on the FP8 tensor cores,
                       Stage 2 radix-selects the top 2048 and maps them to
                       physical slots. The two launches replay as one CUDA
                       graph so no host time sits between them.

What the CUDA version spells out by hand is left to the Triton compiler here:
`tl.dot` on FP8 operands lowers to tcgen05.mma (kind::f8f6f4) with a TMEM
accumulator, and `tl.range(num_stages=...)` turns the K loads into a cp.async
multi-buffer pipeline.

Per row b and logical token t < seq_len:
  page, slot = block_table[b, t // 64], t % 64
  logit[t]   = scale[page, slot] * sum_h ReLU(Q[b, h] . K[page, slot]) * w[b, h]
  out[b]     = physical ids (page * 64 + slot) of the 2048 largest logits,
               padded with -1 when seq_len < 2048
"""

import collections

import torch
import triton
import triton.language as tl

# Problem geometry. Triton kernels can only read module globals that are
# declared constexpr; host code unwraps them with `.value`.
H = tl.constexpr(64)                     # index heads
D = tl.constexpr(128)                    # head dim
PAGE = tl.constexpr(64)                  # tokens per KV page
TOPK = tl.constexpr(2048)
PAGE_BYTES = tl.constexpr(64 * 132)      # 8192 FP8 bytes, then 64 FP32 scales
SCALE_OFFSET = tl.constexpr(64 * 128)    # byte offset of the scales in a page
FAST_PATH_MAX_PAGES = tl.constexpr(32)   # TOPK // PAGE

# --- Tuned on B200 against the contest workloads ----------------------------

# Below this many pages every program scores one tile; at or above it programs
# loop over several tiles so Q, weights and TMEM are set up once per program.
PERSISTENT_MIN_PAGES = 64
SHORT_PAGES_PER_TILE = 2            # 2 pages = one M=128 MMA tile
PERSISTENT_STAGES = 3               # K pipeline depth in the tile loop
PERSISTENT_PROGRAMS = 132 * 4       # target programs per launch (~4 per SM)
PERSISTENT_MIN_TILES = 4            # per-program tile-count clamp
PERSISTENT_MAX_TILES = 64
STAGE1_WARPS = 4
STAGE2_ELEMS_PER_THREAD = 8         # Stage 2 warps = block / (32 * this), max 32
# Stage 2 holds a whole row in one block; longer rows take the torch fallback.
STAGE2_MAX_BLOCK = 16384
MAX_GRAPHS = 32


@triton.jit
def _fast_path(seq_lens_ptr, bt_ptr, out_ptr, max_num_pages):
    """out[b] = block-table transform of [0, seq_len), -1 past seq_len."""
    b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + b)
    p = tl.arange(0, FAST_PATH_MAX_PAGES)
    j = tl.arange(0, PAGE)
    # One load per page, not per token: [32] page ids broadcast to [32, 64].
    page = tl.load(bt_ptr + b * max_num_pages + p, mask=p < max_num_pages, other=0)
    tok = p[:, None] * PAGE + j[None, :]
    phys = page[:, None] * PAGE + j[None, :]
    tl.store(out_ptr + b * TOPK + tok, tl.where(tok < seq_len, phys, -1))


@triton.jit
def _stage1_logits(q_ptr, k_ptr, kscale_ptr, w_ptr, seq_lens_ptr, bt_ptr, logits_ptr,
                   max_num_pages, row_stride, tiles_per_prog,
                   PAGES: tl.constexpr, STAGES: tl.constexpr):
    """FP16 logits for tiles [x * tiles_per_prog, (x + 1) * tiles_per_prog) of row b.

    One tile is PAGES pages (PAGES * 64 tokens) scored by a single FP8 MMA:
    acc[token, head] = K[token] . Q[head], accumulated in FP32.
    """
    TILE: tl.constexpr = PAGES * PAGE
    b = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + b)
    # Stage 2 emits these rows straight from the block table and never reads
    # their logits, so there is nothing to score.
    if seq_len <= TOPK:
        return

    h = tl.arange(0, H)
    d = tl.arange(0, D)
    m = tl.arange(0, TILE)
    # Loaded once per program and reused by every tile.
    q = tl.load(q_ptr + b * (H * D) + h[:, None] * D + d[None, :])   # [H, D] fp8
    qt = tl.trans(q)                                                 # [D, H]
    w = tl.load(w_ptr + b * H + h)                                   # [H] f32

    first = tl.program_id(0) * tiles_per_prog
    # Tiles entirely past seq_len are skipped: Stage 2 only reads logits below
    # seq_len, so they never need a value.
    last = tl.minimum(first + tiles_per_prog, (seq_len + TILE - 1) // TILE)
    last_page = (seq_len - 1) // PAGE
    for t in tl.range(first, last, num_stages=STAGES):
        tok = t * TILE + m                                           # [TILE]
        # Tokens past seq_len in the last tile re-read the row's final live
        # page. That keeps every address valid without a masked load, and
        # their logits are dropped by the store mask below.
        slot = tl.minimum(tok // PAGE, last_page)
        page = tl.load(bt_ptr + b * max_num_pages + slot).to(tl.int64)
        row = tok % PAGE
        k = tl.load(k_ptr + (page * PAGE_BYTES + row * D)[:, None] + d[None, :])  # [TILE, D] fp8
        scale = tl.load(kscale_ptr + page * (PAGE_BYTES // 4) + SCALE_OFFSET // 4 + row)

        acc = tl.dot(k, qt, out_dtype=tl.float32)                    # [TILE, H]
        logit = tl.sum(tl.maximum(acc, 0.0) * w[None, :], axis=1) * scale
        tl.store(logits_ptr + b * row_stride + tok, logit.to(tl.float16),
                 mask=tok < seq_len)


@triton.jit
def _select_digit(digit, mask, need, BINS: tl.constexpr):
    """One radix-select round over the keys under `mask`.

    Returns the digit of the need-th largest key and the number of keys whose
    digit is larger.
    """
    hist = tl.histogram(digit, BINS, mask=mask)
    at_or_above = tl.cumsum(hist, 0, reverse=True)       # keys with digit >= bin
    # Bins [0, d] are exactly the ones that still have `need` keys at or above
    # them. One sum gives both answers: how many such bins (high bits) and how
    # many keys sit in the bins above d (low bits, always below `need`).
    packed = tl.sum(tl.where(at_or_above >= need, 1 << 20, hist), axis=0)
    return (packed >> 20) - 1, packed & 0xFFFFF


@triton.jit
def _stage2_topk(logit_bits_ptr, seq_lens_ptr, bt_ptr, out_ptr,
                 row_stride, max_num_pages, BLOCK: tl.constexpr):
    """Top-2048 of row b's logits as physical slot ids. One program per row.

    Radix select on the FP16 bit patterns, then a prefix-sum compaction; BLOCK
    is a power of two covering the whole row.
    """
    b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + b)
    row_bt = bt_ptr + b * max_num_pages
    row_out = out_ptr + b * TOPK

    # Whole row is inside top-K: same transform as the fast path.
    if seq_len <= TOPK:
        n = tl.arange(0, TOPK)
        keep = n < seq_len
        page = tl.load(row_bt + n // PAGE, mask=keep, other=0)
        tl.store(row_out + n, tl.where(keep, page * PAGE + n % PAGE, -1))
        return

    i = tl.arange(0, BLOCK)
    valid = i < seq_len
    bits = tl.load(logit_bits_ptr + b * row_stride + i, mask=valid, other=0)
    bits = bits.to(tl.int32) & 0xFFFF
    # FP16 bits reordered so integer order matches float order: positives move
    # above 0x8000, negatives are mirrored below it. Slots past seq_len get -1,
    # below every real key.
    key = tl.where(valid, tl.where(bits >= 0x8000, 0xFFFF - bits, bits + 0x8000), -1)

    # K-th largest key by radix select, 5 + 5 + 6 bits from the top. Triton
    # 3.7 lowers tl.histogram to one warp ballot per bin-index bit per key, so
    # three narrow rounds cost less than the two 8-bit rounds of the CUDA
    # version. `above` counts the keys strictly beyond the pivot.
    d, above = _select_digit(key >> 11, valid, TOPK, 32)
    pivot = d << 11
    d, more = _select_digit((key >> 6) & 31, (key >> 11) == d, TOPK - above, 32)
    above += more
    pivot = pivot | (d << 6)
    d, more = _select_digit(key & 63, (key >> 6) == (pivot >> 6), TOPK - above, 64)
    above += more
    pivot = pivot | d

    # Every key above the pivot is taken; ties at the pivot fill the rest in
    # token order. One prefix sum ranks both groups: the low 16 bits count
    # keys above the pivot, the high 16 bits count ties.
    gt = key > pivot
    eq = key == pivot
    rank = tl.cumsum(gt.to(tl.int32) + (eq.to(tl.int32) << 16), 0)
    pos = tl.where(gt, (rank & 0xFFFF) - 1, above + (rank >> 16) - 1)
    take = gt | (eq & (pos < TOPK))

    # Logical token id -> physical slot, fused into the emit.
    page = tl.load(row_bt + i // PAGE, mask=take, other=0)
    tl.store(row_out + pos, page * PAGE + i % PAGE, mask=take)


# --- Host side ---------------------------------------------------------------

_scratch = {}                            # device index -> FP16 logits buffer
_streams = {}                            # device index -> graph capture stream
_graphs = collections.OrderedDict()      # launch key -> CUDAGraph, or None


def _ceil_div(a, b):
    return (a + b - 1) // b


def _stage1_plan(batch, max_num_pages):
    """(pages_per_tile, tiles_per_prog, grid_x, stages, logits_per_row)."""
    if max_num_pages < PERSISTENT_MIN_PAGES:
        pages, stages = SHORT_PAGES_PER_TILE, 1
        tiles = _ceil_div(max_num_pages, pages)
        tiles_per_prog = 1
    else:
        pages, stages = 2, PERSISTENT_STAGES
        tiles = _ceil_div(max_num_pages, pages)
        programs = min(tiles, _ceil_div(PERSISTENT_PROGRAMS, batch))
        tiles_per_prog = min(max(_ceil_div(tiles, programs), PERSISTENT_MIN_TILES),
                             PERSISTENT_MAX_TILES)
    logits_per_row = tiles * pages * PAGE.value
    return pages, tiles_per_prog, _ceil_div(tiles, tiles_per_prog), stages, logits_per_row


def _logits_scratch(device, count):
    """Per-device FP16 buffer, grown geometrically so its address is stable."""
    buf = _scratch.get(device.index)
    if buf is None or buf.numel() < count:
        grown = 2 * buf.numel() if buf is not None else 0
        buf = torch.empty(max(count, grown), dtype=torch.float16, device=device)
        _scratch[device.index] = buf
    return buf


def _topk_torch(logits, seq_lens, block_table, out):
    """Stage 2 for rows too long for one Triton block. Not used by the contest."""
    topk, page = TOPK.value, PAGE.value
    pos = torch.arange(logits.shape[1], device=logits.device)[None, :]
    # Rows that fit in top-K were not scored: rank their tokens by position.
    score = torch.where(seq_lens[:, None] <= topk, -pos.float(), logits.float())
    score = score.masked_fill(pos >= seq_lens[:, None], float("-inf"))
    top = score.topk(topk, dim=1)
    phys = block_table.gather(1, top.indices // page) * page + top.indices % page
    out.copy_(torch.where(torch.isinf(top.values), -1, phys))


def _replay(key, device, launch):
    """Run `launch` through a cached CUDA graph.

    The graph bakes in tensor addresses, so the key is the full argument
    identity. A miss launches directly and captures for next time; a capture
    failure pins the key to direct launches.
    """
    if key not in _graphs:
        launch()
        try:
            stream = _streams.get(device.index)
            if stream is None:
                # Capture needs a real stream; the caller may be on the legacy
                # default stream, which cannot be captured.
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
    # torch replays on the capture stream: order it after the caller's pending
    # work, and make the caller's stream wait for the result.
    stream = _streams[device.index]
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    graph.replay()
    current.wait_stream(stream)


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    batch, max_num_pages = block_table.shape

    if max_num_pages <= FAST_PATH_MAX_PAGES.value:
        _fast_path[(batch,)](seq_lens, block_table, topk_indices, max_num_pages, num_warps=4)
        return

    device = block_table.device
    pages, tiles_per_prog, grid_x, stages, row_stride = _stage1_plan(batch, max_num_pages)
    scratch = _logits_scratch(device, batch * row_stride)
    # The KV cache is int8 storage: view it once as FP8 values and once as the
    # FP32 scales that follow them in every page.
    k_fp8 = k_index_cache_fp8.view(torch.float8_e4m3fn)
    k_scale = k_index_cache_fp8.view(torch.float32)

    def stage1():
        _stage1_logits[(grid_x, batch)](
            q_index_fp8, k_fp8, k_scale, weights, seq_lens, block_table, scratch,
            max_num_pages, row_stride, tiles_per_prog,
            PAGES=pages, STAGES=stages, num_warps=STAGE1_WARPS)

    block = triton.next_power_of_2(row_stride)
    if block > STAGE2_MAX_BLOCK:
        stage1()
        _topk_torch(scratch[: batch * row_stride].view(batch, row_stride),
                    seq_lens, block_table, topk_indices)
        return

    logit_bits = scratch.view(torch.int16)
    stage2_warps = min(32, block // (32 * STAGE2_ELEMS_PER_THREAD))

    def launch():
        stage1()
        _stage2_topk[(batch,)](
            logit_bits, seq_lens, block_table, topk_indices,
            row_stride, max_num_pages, BLOCK=block, num_warps=stage2_warps)

    key = (q_index_fp8.data_ptr(), k_index_cache_fp8.data_ptr(), weights.data_ptr(),
           seq_lens.data_ptr(), block_table.data_ptr(), topk_indices.data_ptr(),
           scratch.data_ptr(), batch, max_num_pages)
    _replay(key, device, launch)
