#!/usr/bin/env python3
"""
Micro-benchmark: Python CuTe DSL TopK indexer vs vectorized PyTorch reference.

Requires CUDA. Uses CUDA events for timing. Shapes match the DSA definition name
(H=64, D=128, PS=64, K_topk=2048) with configurable B and sequence length.

Run:
  python scripts/bench_topk_cuda.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "solution" / "python"))

# kernel.py discovers pip nvcc (nvidia-cuda-nvcc); optional override:
#   CUDA_HOME=/path/to/cuda-13.2
from kernel import kernel as cute_kernel  # noqa: E402


def torch_gather_k_batched(
    cache_u8: torch.Tensor,
    bt_i32: torch.Tensor,
    b: int,
    s: int,
    p: int,
    ps: int,
    hds: int,
    d_model: int,
) -> torch.Tensor:
    """Same layout as CuTe gather: FP8 K bytes + per-(page,tok) float32 scale."""
    device = cache_u8.device
    flat = cache_u8.reshape(-1)
    b_ = torch.arange(b, device=device, dtype=torch.long)[:, None, None].expand(b, s, d_model)
    s_ = torch.arange(s, device=device, dtype=torch.long)[None, :, None].expand(b, s, d_model)
    d_ = torch.arange(d_model, device=device, dtype=torch.long)[None, None, :].expand(b, s, d_model)
    page_slot = s_ // ps
    t_ = s_ % ps
    pid = bt_i32[b_.long(), page_slot.long()].clamp(0, p - 1)
    page_byte = pid * (ps * hds)
    fp8_off = page_byte + t_ * d_model + d_
    sb = (page_byte + ps * d_model + t_ * 4)[:, :, 0].reshape(-1)
    lin = fp8_off.reshape(-1).long()
    buf = torch.empty(lin.numel(), dtype=torch.float8_e4m3fn, device=device)
    buf.view(torch.uint8).copy_(flat[lin])
    k_fp32 = buf.view(b, s, d_model).float()
    idx4 = sb[:, None] + torch.arange(4, device=device, dtype=torch.long)[None, :]
    u4 = flat[idx4]
    scales = u4.view(torch.float32).squeeze(-1).view(b, s)
    return k_fp32 * scales[:, :, None]


@torch.no_grad()
def ref_kernel(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    """Pure PyTorch reference (same numerics path as version_v1 host code)."""
    b = int(q_index_fp8.size(0))
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
    sl_vec = [int(seq_lens[i].item()) for i in range(b)]
    max_seq_len = max(sl_vec) if sl_vec else 0
    if max_seq_len == 0:
        return

    max_pages_needed = (max_seq_len + ps - 1) // ps
    actual_pages = min(max_pages_needed, int(block_table.size(1)))
    s = actual_pages * ps
    bt_i32 = block_table[:, :actual_pages].to(torch.int32).clamp(0, p - 1).contiguous()

    k_batched = torch_gather_k_batched(cache_u8, bt_i32, b, s, p, ps, hds, d)
    q_f32 = q_index_fp8.to(torch.float32)
    logits = torch.bmm(q_f32, k_batched.transpose(1, 2))
    weighted = logits.relu() * weights.contiguous().unsqueeze(2)

    for bi in range(b):
        sl = sl_vec[bi]
        if sl == 0:
            continue
        actual_topk = min(k_topk, sl)
        scores = weighted[bi, :, :sl].sum(0)
        _, topk_local = scores.topk(actual_topk, dim=-1, largest=True, sorted=True)
        topk_local_i32 = topk_local.to(torch.int32)
        bt_row = bt_i32[bi]
        out_row = topk_indices[bi, :actual_topk]
        for j in range(actual_topk):
            local = int(topk_local_i32[j].item())
            page = local // ps
            off = local % ps
            gid = int(bt_row[page].item()) * ps + off
            out_row[j] = gid


def make_inputs(
    device: torch.device,
    b: int,
    h: int,
    d_model: int,
    p: int,
    ps: int,
    seq_len: int,
    k_topk: int,
    seed: int = 0,
) -> tuple:
    torch.manual_seed(seed)
    hds = ps * d_model + ps * 4
    q = torch.randn(b, h, d_model, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    cache = torch.randint(0, 255, (p, ps, 1, hds), device=device, dtype=torch.uint8)
    w = torch.rand(b, h, device=device, dtype=torch.float32)
    sl = torch.full((b,), seq_len, device=device, dtype=torch.int32)
    max_pages = (seq_len + ps - 1) // ps
    bt = torch.randint(0, p, (b, max_pages), device=device, dtype=torch.int32)
    topk = torch.full((b, k_topk), -1, device=device, dtype=torch.int32)
    return q, cache, w, sl, bt, topk


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, args, kwargs, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args, **kwargs)
    sync_cuda()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    sync_cuda()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


def main():
    if not torch.cuda.is_available():
        print("CUDA required for this benchmark.")
        sys.exit(1)

    device = torch.device("cuda", int(os.environ.get("CUDA_DEVICE", "0")))
    torch.cuda.set_device(device)

    b = int(os.environ.get("BENCH_B", "4"))
    h = 64
    d_model = 128
    p = 256
    ps = 64
    seq_len = int(os.environ.get("BENCH_SEQ", "2048"))
    k_topk = 2048

    q, cache, w, sl, bt, topk = make_inputs(device, b, h, d_model, p, ps, seq_len, k_topk)
    topk_ref = topk.clone()
    topk_cute = topk.clone()

    ref_kernel(q, cache, w, sl, bt, topk_ref)
    sync_cuda()
    cute_kernel(q, cache, w, sl, bt, topk_cute)
    sync_cuda()

    if not torch.equal(topk_ref, topk_cute):
        bad = (topk_ref != topk_cute).sum().item()
        print(f"Correctness: FAIL ({bad} mismatched indices)")
        # show first diff
        for i in range(b):
            if not torch.equal(topk_ref[i], topk_cute[i]):
                print(f" batch {i} ref {topk_ref[i, :8]} cute {topk_cute[i, :8]}")
                break
        sys.exit(1)
    print("Correctness: OK (topk indices match reference)")

    warmup = int(os.environ.get("BENCH_WARMUP", "5"))
    iters = int(os.environ.get("BENCH_ITERS", "20"))

    def run_ref():
        topk_ref.fill_(-1)
        ref_kernel(q, cache, w, sl, bt, topk_ref)

    def run_cute():
        topk_cute.fill_(-1)
        cute_kernel(q, cache, w, sl, bt, topk_cute)

    # Prime CuTe JIT compile outside timed region
    run_cute()
    sync_cuda()

    ms_ref = bench(run_ref, (), {}, warmup, iters)
    ms_cute = bench(run_cute, (), {}, warmup, iters)
    speedup = ms_ref / ms_cute if ms_cute > 0 else float("inf")

    print(
        f"Shapes: B={b} H={h} D={d_model} seq_len={seq_len} PS={ps} K_topk={k_topk} "
        f"(warmup={warmup}, iters={iters})"
    )
    print(f"Reference (torch gather + bmm + topk): {ms_ref:.4f} ms/it")
    print(f"CuTe DSL solution:                     {ms_cute:.4f} ms/it")
    print(f"Speedup (ref / cute):                  {speedup:.2f}x")


if __name__ == "__main__":
    main()
