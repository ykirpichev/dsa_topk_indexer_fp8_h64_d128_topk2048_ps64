"""
DSA TopK indexer — B200: CuTe DSL building blocks + PyTorch orchestration.

**Default path (Modal / flashinfer-bench):** reference-style GPU dequant of the full
page cache, vectorized paged gather into `K_batched`, `torch.bmm`, relu×weights,
`topk`, and a small PyTorch page-table transform. This matches the dataset reference
and passes smoke workloads.

**Opt-in CuTe JIT** (`DSA_CUTE_JIT=1`): inline-PTX gather + page-transform (TVM-FFI).
Modal smoke with JIT on still shows **some** `INCORRECT_NUMERICAL` workloads (race vs
reference ordering); use for perf experiments only until fixed. Scale loads use
`gmem_addr_f32` for 4-byte alignment; `torch.cuda.synchronize()` after CuTe launches.

**Opt-in FP8 TMMA** (`DSA_FP8_TMMA_MM=1`): uses **`blackwell_fp8_batched_mm.py`**, patterned
after NVIDIA CUTLASS Blackwell tutorials (`examples/python/CuTeDSL/blackwell/tutorial_gemm/`):
`from_dlpack` + `cute.compile(..., --generate-line-info --enable-tvm-ffi)` + `tcgen05`
`make_trivial_tiled_mma`. Requires `S % 128`.

**CuTe experimental TMA** (`DSA_FP8_TMMA_MM_EXT=1`, also set `DSA_FP8_TMMA_MM=1`): same FP8
GEMM but host/device use `cutlass.cute.experimental` (`tma_load`, `get_cta_v_map_ab`) with
TMA + SMEM pipeline + TMEM accumulator (see CUTLASS `dense_gemm_cute_pipeline.py`). Requires
CUDA toolkit 13.1+ where `import cutlass.cute.experimental` succeeds.

Dependencies: `nvidia-cutlass-dsl` (see `config.toml`). Set `CUTE_DSL_ARCH=sm_100a` when
compiling tcgen05 kernels.
"""

from __future__ import annotations

import functools
import os
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32, Uint8
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

try:
    from .blackwell_fp8_batched_mm import fp8_batched_mm_hs as _fp8_batched_mm_hs
    from .blackwell_fp8_batched_mm import _N_TILE as _N_TILE_FP8
except ImportError:
    from blackwell_fp8_batched_mm import fp8_batched_mm_hs as _fp8_batched_mm_hs
    from blackwell_fp8_batched_mm import _N_TILE as _N_TILE_FP8

try:
    import cutlass.cute.experimental as _cute_experimental  # noqa: F401
except NotImplementedError:
    _cute_experimental = None

# Opt-in: CuTe JIT gather + page-transform (default off; use PyTorch indexing which matches reference).
_USE_CUTE_JIT = os.environ.get("DSA_CUTE_JIT", "").lower() in ("1", "true", "yes")
# Blackwell FP8 TMMA matmul: set DSA_FP8_TMMA_MM=1 when the tcgen05 kernel compiles.
_USE_FP8_TMMA_MM = os.environ.get("DSA_FP8_TMMA_MM", "").lower() in ("1", "true", "yes")

# --- PTX: global loads / FP8 decode -------------------------------------------------


@dsl_user_op
def ld_global_u8(addr: cutlass.Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_f32(addr: cutlass.Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_i32(addr: cutlass.Int64, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fp8_e4m3_u8_to_f32(fp8_u8: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_u8).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f32.e4m3 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def gmem_addr_u8(m: cute.Tensor, linear: Int32, *, loc=None, ip=None) -> cutlass.Int64:
    elem = m.iterator + linear
    return cutlass.Int64(llvm.ptrtoint(T.i64(), elem.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def gmem_addr_f32(m: cute.Tensor, linear: Int32, *, loc=None, ip=None) -> cutlass.Int64:
    elem = m.iterator + linear
    return cutlass.Int64(llvm.ptrtoint(T.i64(), elem.llvm_ptr, loc=loc, ip=ip))


class _GatherDequantKernel:
    """Maps one CUDA block per token, D threads — same as version_v1 CUDA kernel."""

    def __init__(self, d_model: int):
        self.d_model = d_model

    @cute.jit
    def __call__(
        self,
        m_cache: cute.Tensor,
        m_bt: cute.Tensor,
        m_k: cute.Tensor,
        B: Int32,
        S: Int32,
        P: Int32,
        PS: Int32,
        HDS: Int32,
        actual_pages: Int32,
        stream,
    ):
        num_tokens = B * S
        self.kernel(m_cache, m_bt, m_k, B, S, P, PS, HDS, actual_pages).launch(
            grid=[num_tokens, 1, 1],
            block=[self.d_model, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m_cache: cute.Tensor,
        m_bt: cute.Tensor,
        m_k: cute.Tensor,
        B: Int32,
        S: Int32,
        P: Int32,
        PS: Int32,
        HDS: Int32,
        actual_pages: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        token_idx = bidx
        d = tidx
        Dm = Int32(self.d_model)

        if d >= Dm:
            return

        b = token_idx // S
        s = token_idx - b * S
        t = s % PS
        page_slot = s // PS

        # Match Triton/CUDA: only upper clamp (host bt is already clamped to [0, P-1])
        page_id = Int32(m_bt[(b, page_slot)])
        page_id = cutlass.min(page_id, P - Int32(1))

        # SoA per page (same as CUDA/Triton): [PS*D fp8][PS*4 scales]
        page_byte = page_id * (PS * HDS)
        fp8_off = page_byte + t * Dm + d
        scale_off = page_byte + PS * Dm + t * Int32(4)

        fp8_u8 = ld_global_u8(gmem_addr_u8(m_cache, fp8_off))
        scale = ld_global_f32(gmem_addr_f32(m_cache, scale_off))
        val = fp8_e4m3_u8_to_f32(fp8_u8 & Uint32(0xFF)) * scale

        out_lin = token_idx * Dm + d
        st_global_f32_out(m_k, out_lin, val)


@dsl_user_op
def st_global_f32_out(m: cute.Tensor, linear: Int32, val: Float32, *, loc=None, ip=None) -> None:
    addr = gmem_addr_f32(m, linear, loc=loc, ip=ip)
    st_global_f32_addr(addr, val, loc=loc, ip=ip)


@dsl_user_op
def st_global_f32_addr(addr: cutlass.Int64, val: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _PageTransformKernel:
    def __init__(self, block_threads: int):
        self.block_threads = block_threads

    @cute.jit
    def __call__(
        self,
        m_local: cute.Tensor,
        m_bt_row: cute.Tensor,
        m_out: cute.Tensor,
        actual_topk: Int32,
        PS: Int32,
        stream,
    ):
        grid_x = cute.ceil_div(actual_topk, Int32(self.block_threads))
        self.kernel(m_local, m_bt_row, m_out, actual_topk, PS).launch(
            grid=[grid_x, 1, 1],
            block=[self.block_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        m_local: cute.Tensor,
        m_bt_row: cute.Tensor,
        m_out: cute.Tensor,
        actual_topk: Int32,
        PS: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        i = bidx * bdim + tidx
        if i >= actual_topk:
            return
        local = Int32(m_local[i])
        page = local // PS
        off = local % PS
        gid = Int32(m_bt_row[page]) * PS + off
        st_global_i32(gmem_addr_i32(m_out, i), gid)


@dsl_user_op
def gmem_addr_i32(m: cute.Tensor, linear: Int32, *, loc=None, ip=None) -> cutlass.Int64:
    elem = m.iterator + linear
    return cutlass.Int64(llvm.ptrtoint(T.i64(), elem.llvm_ptr, loc=loc, ip=ip))


@functools.cache
def _compiled_gather(d_model: int) -> Callable:
    sym_B = cute.sym_int()
    sym_S = cute.sym_int()
    sym_P = cute.sym_int()
    sym_PS = cute.sym_int()
    sym_HDS = cute.sym_int()
    sym_ap = cute.sym_int()
    sym_nbytes = cute.sym_int()

    kobj = _GatherDequantKernel(d_model)
    m_cache = cute.runtime.make_fake_compact_tensor(
        Uint8, (sym_nbytes,), stride_order=(0,), assumed_align=1
    )
    m_bt = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_B, sym_ap), stride_order=(1, 0), assumed_align=4
    )
    m_k = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_B, sym_S, d_model), stride_order=(2, 1, 0), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kobj,
        m_cache,
        m_bt,
        m_k,
        sym_B,
        sym_S,
        sym_P,
        sym_PS,
        sym_HDS,
        sym_ap,
        stream_fake,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compiled_page_transform(block_threads: int) -> Callable:
    sym_topk = cute.sym_int()
    sym_ps = cute.sym_int()
    sym_max_pages = cute.sym_int()

    kobj = _PageTransformKernel(block_threads)
    m_local = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_topk,), stride_order=(0,), assumed_align=4
    )
    m_bt_row = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_max_pages,), stride_order=(0,), assumed_align=4
    )
    m_out = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_topk,), stride_order=(0,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kobj,
        m_local,
        m_bt_row,
        m_out,
        sym_topk,
        sym_ps,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def _pytorch_dequant_full_cache(cache_u8: torch.Tensor, p: int, ps: int, d: int, hds: int) -> torch.Tensor:
    """Per-page FP8 + scale → f32 [P, PS, d] (same layout as competition reference)."""
    cache = cache_u8.view(torch.uint8)
    kv_flat = cache.view(p, ps * hds)
    fp8_bytes = kv_flat[:, : ps * d].contiguous()
    fp8_tensor = fp8_bytes.view(p, ps, d).view(torch.float8_e4m3fn)
    fp8_f32 = fp8_tensor.to(torch.float32)
    scale_bytes = kv_flat[:, ps * d :].contiguous()
    scale = scale_bytes.view(p, ps, 4).view(torch.float32)
    return fp8_f32 * scale


def _pytorch_gather_dequant_k_batched(
    cache_u8: torch.Tensor,
    bt_i32: torch.Tensor,
    b: int,
    s_pad: int,
    d: int,
    p: int,
    ps: int,
    hds: int,
) -> torch.Tensor:
    device = cache_u8.device
    k_all = _pytorch_dequant_full_cache(cache_u8, p, ps, d, hds)
    bb = torch.arange(b, device=device, dtype=torch.int64).view(b, 1).expand(b, s_pad)
    ss = torch.arange(s_pad, device=device, dtype=torch.int64).view(1, s_pad).expand(b, s_pad)
    page_col = ss // ps
    page_id = bt_i32[bb, page_col].to(torch.int64).clamp(0, p - 1)
    t = ss % ps
    return k_all[page_id, t]


def _run_gather_dequant(
    cache_u8: torch.Tensor,
    bt_i32: torch.Tensor,
    k_batched: torch.Tensor,
    b: int,
    s: int,
    p: int,
    ps: int,
    hds: int,
    actual_pages: int,
) -> None:
    if not _USE_CUTE_JIT:
        k_batched.copy_(
            _pytorch_gather_dequant_k_batched(
                cache_u8, bt_i32, b, s, k_batched.shape[2], p, ps, hds
            )
        )
        return
    try:
        fn = _compiled_gather(k_batched.shape[2])
        fn(
            cache_u8,
            bt_i32,
            k_batched,
            Int32(b),
            Int32(s),
            Int32(p),
            Int32(ps),
            Int32(hds),
            Int32(actual_pages),
        )
        torch.cuda.synchronize()
    except Exception:
        k_batched.copy_(
            _pytorch_gather_dequant_k_batched(
                cache_u8, bt_i32, b, s, k_batched.shape[2], p, ps, hds
            )
        )


def _pytorch_page_transform(
    local_idx: torch.Tensor,
    bt_row: torch.Tensor,
    out_row: torch.Tensor,
    actual_topk: int,
    ps: int,
) -> None:
    li = local_idx[:actual_topk]
    page = li // ps
    off = li % ps
    gpage = bt_row[page]
    out_row[:actual_topk].copy_(gpage * ps + off)


def _run_page_transform(
    local_idx: torch.Tensor,
    bt_row: torch.Tensor,
    out_row: torch.Tensor,
    actual_topk: int,
    ps: int,
) -> None:
    if not _USE_CUTE_JIT:
        _pytorch_page_transform(local_idx, bt_row, out_row, actual_topk, ps)
        return
    try:
        fn = _compiled_page_transform(256)
        fn(
            local_idx,
            bt_row,
            out_row,
            Int32(actual_topk),
            Int32(ps),
        )
        torch.cuda.synchronize()
    except Exception:
        _pytorch_page_transform(local_idx, bt_row, out_row, actual_topk, ps)


@torch.no_grad()
def kernel(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    b = int(q_index_fp8.size(0))
    _h = int(q_index_fp8.size(1))
    d = int(q_index_fp8.size(2))
    p = int(k_index_cache_fp8.size(0))
    ps = int(k_index_cache_fp8.size(1))
    hds = int(k_index_cache_fp8.size(3))
    k_topk = int(topk_indices.size(1))

    topk_indices.fill_(-1)

    cache_u8 = (
        k_index_cache_fp8
        if k_index_cache_fp8.dtype == torch.uint8
        else k_index_cache_fp8.view(torch.uint8)
    )

    sl_cpu = seq_lens.cpu()
    sl_vec = [int(sl_cpu[i]) for i in range(b)]
    max_seq_len = max(sl_vec) if sl_vec else 0
    if max_seq_len == 0:
        return

    max_pages_needed = (max_seq_len + ps - 1) // ps
    actual_pages = min(max_pages_needed, int(block_table.size(1)))
    s = actual_pages * ps

    bt_i32 = (
        block_table[:, :actual_pages].to(torch.int32).clamp(0, p - 1).contiguous()
    )

    device = q_index_fp8.device
    k_batched = torch.empty((b, s, d), device=device, dtype=torch.float32)
    _run_gather_dequant(cache_u8, bt_i32, k_batched, b, s, p, ps, hds, actual_pages)

    q_f32 = q_index_fp8.to(torch.float32)
    if _USE_FP8_TMMA_MM and s % _N_TILE_FP8 == 0:
        logits = torch.empty((b, _h, s), device=device, dtype=torch.float32)
        _fp8_batched_mm_hs(q_f32, k_batched, logits)
    else:
        logits = torch.bmm(q_f32, k_batched.transpose(1, 2).contiguous())
    weighted = logits.relu() * weights.contiguous().unsqueeze(2)

    # Batched score reduce + mask (avoid Python loop over heads)
    col = torch.arange(s, device=device, dtype=torch.int64).view(1, s)
    sl_t = seq_lens.to(device).to(torch.int64).view(b, 1)
    valid = col < sl_t
    scores = (weighted * valid.unsqueeze(1).to(dtype=weighted.dtype)).sum(dim=1)
    scores = scores.masked_fill(~valid, float("-inf"))

    bt_ptr = bt_i32
    out_ptr = topk_indices

    for bi in range(b):
        sl = sl_vec[bi]
        if sl == 0:
            continue
        actual_topk = min(k_topk, sl)
        _, topk_local = scores[bi, :].topk(actual_topk, dim=-1, largest=True, sorted=True)
        topk_local_i32 = topk_local.to(torch.int32).contiguous()
        _run_page_transform(
            topk_local_i32,
            bt_ptr[bi],
            out_ptr[bi, :actual_topk],
            actual_topk,
            ps,
        )


__all__ = ["kernel"]
