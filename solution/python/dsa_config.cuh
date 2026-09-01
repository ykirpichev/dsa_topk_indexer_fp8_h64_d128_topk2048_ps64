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

// Stage-1 plan for a call; derivable from max_num_pages alone.
enum class Plan : uint8_t {
    FastPath     = 0,  // context fits in top-K: skip Stage 1 and Stage 2
    Short        = 1,  // kFastPathMaxPages < pages < kPersistentPageThreshold
    PersistentWs = 2,  // pages >= kPersistentPageThreshold
};

inline Plan select_plan(int max_num_pages) {
    if (max_num_pages <= kFastPathMaxPages) return Plan::FastPath;
    if (max_num_pages >= kPersistentPageThreshold) return Plan::PersistentWs;
    return Plan::Short;
}

inline int count_kv_tile_pairs(int max_num_pages) {
    return (max_num_pages + kPagesPerUMMA - 1) / kPagesPerUMMA;
}

inline int stage2_max_len(Plan plan, int max_num_pages) {
    if (plan == Plan::FastPath) return 0;
    if (plan == Plan::PersistentWs) {
        return count_kv_tile_pairs(max_num_pages) * (kPagesPerUMMA * kBlockKv);
    }
    return max_num_pages * kPageSize;
}

struct PersistentGrid {
    int max_kv_tile_pairs = 0;
    int tiles_per_cta     = 0;
    int num_splits        = 0;
};

inline PersistentGrid compute_persistent_grid(int B, int max_num_pages) {
    PersistentGrid g{};
    g.max_kv_tile_pairs = count_kv_tile_pairs(max_num_pages);
    int num_splits = (kPersistentSmTarget + B - 1) / B;  // >= 1 for B >= 1
    if (num_splits > g.max_kv_tile_pairs) num_splits = g.max_kv_tile_pairs;
    g.tiles_per_cta = (g.max_kv_tile_pairs + num_splits - 1) / num_splits;
    if (g.tiles_per_cta < kPersistentMinTiles) g.tiles_per_cta = kPersistentMinTiles;
    if (g.tiles_per_cta > kPersistentMaxTiles) g.tiles_per_cta = kPersistentMaxTiles;
    g.num_splits = (g.max_kv_tile_pairs + g.tiles_per_cta - 1) / g.tiles_per_cta;
    return g;
}

}  // namespace
