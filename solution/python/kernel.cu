/*
 * DSA FP8 Top-K Indexer — NVIDIA B200 (sm_100a).
 *
 * MLSys 2026 FlashInfer Kernel Generation Contest, DSA track. Built as a
 * PyTorch extension via torch.utils.cpp_extension.load() (see solution.py).
 *
 * For batch row `b` and context position `t` in [0, seq_lens[b]):
 *
 *   S[h]       = sum_d Q[b,h,d] * K[page,slot,d]          (FP8 x FP8 -> FP32)
 *   logit[b,t] = scale[page,slot] * sum_h ReLU(S[h]) * weights[b,h]
 *   topk[b,:]  = indices of the k = 2048 largest logits in row b
 *   topk[b,j]  = block_table[b, topk[b,j]/64] * 64 + (topk[b,j] % 64)
 *
 * One translation unit; each header is included exactly once, in this order:
 *
 *   dsa_config.cuh              geometry + tuned constants
 *   tcgen05_ptx.h               inline PTX (cp.async, mbarrier, tcgen05)
 *   umma_desc.h                 UMMA descriptor bit layouts
 *   stage1_common.cuh           SMEM swizzle + descriptor helpers
 *   fast_path.cuh               Stage 1+2 bypass (context fits in top-K)
 *   stage1_short.cuh            Stage 1, mid-size contexts
 *   stage1_persistent_ws.cuh    Stage 1, large contexts (warp-specialised)
 *   stage2_topk.cuh             radix top-K + block-table transform
 *   graph_cache.cuh             shape-keyed CUDA graph capture/replay
 *   dispatch.cuh                host entry, plan selection, launches
 */

#include "dispatch.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &dsa_topk_indexer_cuda, "DSA TopK indexer (SM100a UMMA)");
}
