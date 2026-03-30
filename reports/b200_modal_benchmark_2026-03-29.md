# B200 Modal benchmark summary (smoke, 8 workloads)

**Definition:** `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`  
**Environment:** Modal `gpu="B200:1"`, image `flashinfer/flashinfer-ci-cu132:latest`, CUDA **13.2**, `flashinfer-bench` + `nvidia-cutlass-dsl==4.4.2`.  
**Command:** `FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py` (optional `FIB_MODAL_DSA_FP8_TMMA_MM=1` for Python FP8 matmul).

## Results

| Configuration | Geomean speedup vs reference | Correctness (smoke) | Typical latency (order of) |
|---------------|------------------------------|---------------------|----------------------------|
| **Triton** (`solution/triton/kernel.py`) | **3.27×** | 8/8 **PASSED** | ~0.28–0.38 ms |
| **Python** (`torch.bmm`, full-cache dequant + gather) | **1.22×** | 8/8 **PASSED** | ~0.91–0.99 ms |
| **Python + `DSA_FP8_TMMA_MM=1`** (CuTe FP8 UMMA + per-row quant) | **1.14×** | **7/8 PASSED** | ~0.91–0.97 ms (similar; one workload **INCORRECT_NUMERICAL**) |

Per-workload tables from the Modal run are in the job logs (columns: `T`, `S_pad`, `Lat(ms)`, `Spdup`, `Match`, roofline helpers `RF_Lµs`, `RF_Iµs`, `%Pk_I`).

### Notes

1. **Roofline columns** (`RF_L`, `RF_I`, `%Pk_I`) stay very small vs measured latency: the simple HBM model does not capture most of the real cost (tensor-core GEMM, `topk`, framework overhead, etc.). Treat them as orientation only.
2. **FP8 TMMA path:** On this smoke set it is **not faster** than `torch.bmm` and **breaks strict top-k agreement** on at least one workload (logits differ from FP32 `bmm`, so sorted `topk` indices can change). Treat as experimental unless quantization is aligned with the reference numerics.

## Recommendation

- **Ship / tune against Triton** for this track unless Python is required: ~**2.7×** higher geomean than Python on the same smoke workloads.
- Keep Python as a readable reference and for CuTe experiments; do not enable `DSA_FP8_TMMA_MM` for correctness-gated runs until numerics match.

## Further speedup ideas (prioritized)

1. **Triton path**
   - Fuse **gather + dequant** with the **first GEMM slice** (or stages) so `K_batched` f32 is not fully materialized at peak size.
   - Overlap **page-table / indexing** with compute where the framework allows (streams, CUDA graphs if the bench supports fixed shapes).
   - Try **TF32** for `bmm` only if the benchmark tolerance allows (often changes top-k; verify).

2. **Python path**
   - Replace full-cache `K_all` dequant with **on-the-fly gather** (single pass over needed pages only) to cut HBM traffic when `S_pad` is large vs logical tokens.
   - **Vectorize the batch loop** (`topk` / page transform) with batched ops or a small Triton/CuTe kernel for the tail.

3. **FP8 / Tensor Core**
   - If pursuing FP8 again: use a **reference-matched** scaling policy (e.g. same scales as reference kernel if exposed), or accept **approximate** top-k and a different eval mode.
   - Consider **cuBLASLt / `torch._scaled_mm`** only after proving bit-for-bit or index-for-index agreement on the trace workloads.

4. **Profiling**
   - Run **NCU** on Modal (or a dedicated job) on the largest smoke workload to see whether time is in GEMM, `topk`, or PyTorch overhead; that single data point should reorder the list above.
