# B200 Modal benchmark summary

## Why FP8 TMMA looked “slowest” before

1. **Wrong math vs reference:** The old path **re-quantized** Q and K with **per-row** scales after building **full f32 `K_batched`**. That is **not** the same as `torch.bmm(q_f32, k_f32^T)` on **cache-dequantized** K, so logits (and **top-k**) diverged → **INCORRECT_NUMERICAL** on some workloads, and any timing comparison was invalid.

2. **Extra work:** The FP8 path still did **full f32 gather** then **quantize again** → more bandwidth and kernels than `bmm` alone.

3. **`torch.bmm` is already strong on B200** for this small batch-GEMM shape; without **TMA + fusion**, FP8 UMMA does not automatically win.

## Fix (current Python path)

- **Triton** `triton_gather_fp8.py`: gather **raw FP8 bytes** + **per-token K scale** `[B,S]` (no f32 K tensor).
- **CuTe** `fp8_batched_mm_hs`: **FP8 × FP8 → FP32** UMMA (same as before, **no** per-row re-quant).
- **After multiply:** `logits.mul_(k_scale.unsqueeze(1))` in **FP32** so logits match **decode(fp8)×scale** semantics.

## Modal smoke (8 workloads), CUDA 13.2, B200

| Mode | Geomean vs ref | PASSED |
|------|----------------|--------|
| Python, `torch.bmm` + PyTorch f32 gather | **1.18×** | 8/8 |
| Python, FP8 gather + UMMA + K-scale | **1.19×** | 8/8 |

Latencies ~**0.90–0.96 ms**; FP8 path is **~flat to slightly faster** than bmm here, not slower.

## TMA note

A **TMA + pipelined SMEM + TMEM** kernel (CUTLASS `fp16_gemm_0` style) was attempted with **`make_tiled_tma_atom_A/B`**; it hit **COMPILE_ERROR** on **`nvidia-cutlass-dsl==4.4.2`** for this tile on Modal. The working path remains **bulk GMEM loads → registers → UMMA** until the DSL supports this FP8 configuration or a minimal repro is fixed upstream.

## Commands

```bash
# Python + bmm
modal run scripts/run_modal.py   # with config language = python

# Python + FP8 UMMA + K scales
FIB_MODAL_SMOKE=1 FIB_MODAL_DSA_FP8_TMMA_MM=1 modal run scripts/run_modal.py
```
