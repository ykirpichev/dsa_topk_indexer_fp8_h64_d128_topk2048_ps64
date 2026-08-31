// Problem geometry and tuned constants. Every threshold here was measured on
// B200 against the contest workload set; re-run scripts/run_modal_compare.py
// after touching one.

#pragma once

#include <cstdint>

namespace {

constexpr int kNumHeads        = 64;
constexpr int kHeadDim         = 128;
constexpr int kPageSize        = 64;
constexpr int kHeadDimWithSf   = 132;
constexpr int kPageBytes       = kPageSize * kHeadDimWithSf;  // 8448
constexpr int kScaleOffsetBytes= kPageSize * kHeadDim;        // 8192
constexpr int kTopK            = 2048;
constexpr int kBlockKv         = 64;
constexpr int kStage2Threads   = 1024;
constexpr int kRadix           = 256;
constexpr int kRadixRounds     = 2;
constexpr int kOrderedBits     = 16;

// Two Stage-1 kernels are compiled and the host picks one per call:
// `short` takes 1 page per UMMA on a flat (max_num_pages, B) grid and has the
// lowest setup cost; `persistent_ws` takes 2 pages per UMMA in warp-
// specialised CTAs that set up Q/TMEM/mbarriers once and loop over tiles.
constexpr int kPagesPerUMMA = 2;

// K/kscale cp.async pipeline depth: UMMA consumes buffer i % kKVStages while
// prefetches for i+1 and i+2 are in flight. 4 stages measured 0.9% slower on
// the 40-63 page bucket -- the math path is the bottleneck, not the producer.
constexpr int kKVStages = 3;

// --- Host dispatch thresholds ----------------------------------------------

// The whole context fits in top-K; 69 of the 128 contest workloads.
constexpr int kFastPathMaxPages = kTopK / kPageSize;   // 32

// Above this the persistent kernel wins; below it the short kernel's flat grid
// saturates the SMs sooner. 64 beat 40 by 8.9% on the 40-63 page bucket.
constexpr int kPersistentPageThreshold = 64;

// Target ~4 CTAs per SM over 132 SMs, then clamp per-CTA tile-pairs.
constexpr int kPersistentSmTarget = 132 * 4;
constexpr int kPersistentMinTiles = 4;
constexpr int kPersistentMaxTiles = 64;

// Stage-1 plan for a call; also part of the graph cache key.
enum class Plan : uint8_t {
    FastPath     = 0,  // context fits in top-K: skip Stage 1 and Stage 2
    Short        = 1,  // kFastPathMaxPages < pages < kPersistentPageThreshold
    PersistentWs = 2,  // pages >= kPersistentPageThreshold
};

}  // namespace
