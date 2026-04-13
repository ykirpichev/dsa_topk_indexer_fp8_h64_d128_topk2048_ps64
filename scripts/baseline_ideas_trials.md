# Baseline-inspired trials (FlashInfer trace)

Source: [flashinfer_deepgemm_wrapper](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/blob/main/solutions/baseline/dsa/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/flashinfer_deepgemm_wrapper_2ba145.json) — `deep_gemm.fp8_paged_mqa_logits` + `flashinfer.top_k_page_table_transform(logits.float16, ...)`.

## Implemented in-repo (CUDA path)

| Trial | How | Modal smoke (17 wl, strict rtol/atol) | Combine? |
|-------|-----|----------------------------------------|----------|
| **A. FP16 top-k scores** | `FIB_TOPK_FP16=1` → `-DFIB_TOPK_FP16` in `binding.py`; `scores.to(kHalf)` before `topk_out` | **8.51×**, 17/17, match 1.0 (vs **8.24×** without flag, same session order) | Optional A/B; default remains FP32 |
| **B. PTX `-O3` + ftz** | already default in `binding.py` | prior full 128 **12.53×** | **Yes** — keep as default |
| **C. Gather-mask** | already in `kernel.cu` | prior | **Yes** — keep |
| **D. Vectorized FP8 gather** (paged-FP8 style) | `gather_dequant_vec4_kernel`: 32 thr/token, **uint32** load ×4 FP8, **scale** in shared mem; **D=128** only | **8.20×**, 17/17, match 1.0 | **Yes** — inspired by paged FP8 kernels; JIT `topk_cuda_cublas_gather_vec4` |

## Not in CUDA repo (needs Python deps / different entry)

| Idea | Package | Note |
|------|---------|------|
| FP8 paged MQA logits | `deep_gemm` | Replace gather+bmm+relu*weight+sum |
| `top_k_page_table_transform` | `flashinfer` | May replace `topk`+`page_transform` |
| `get_paged_mqa_logits_metadata` | `deep_gemm` | SM scheduling |

To try **deep_gemm + flashinfer** end-to-end you’d add a **second** solution (`main.py`) and flip `config.toml` — not done here.

## Suggested combine (current branch)

Default submission path: **gather-mask + ftz + PTX `-O3`** (no `FIB_TOPK_FP16`).

Re-run smoke with `FIB_TOPK_FP16=1` on full 128 before shipping fp16 path; if geomean improves consistently, tag separately.
