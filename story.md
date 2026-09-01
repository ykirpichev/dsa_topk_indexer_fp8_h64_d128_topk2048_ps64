# The Story Behind the Kernel

It started, as many things do, with a $20/month Cursor subscription.

I am not a kernel engineer by trade. But when the MLSys 2026 FlashInfer AI Kernel Generation Contest opened a DSA track on NVIDIA B200, I decided to try. The task: optimize a paged FP8 Top-K Indexer and a Sparse Attention kernel for DeepSeek sparse attention decoding. Production-grade CUDA. Warp specialization. Inline PTX. Tensor memory. The kind of work that normally takes a small team of specialists months to get right.

My plan was to use agents for the heavy lifting.

## Phase 1 — Six weeks against an immature evaluator

The first six weeks were a lesson in invisible walls.

The evaluator compared raw output index vectors element-wise. Any custom top-K implementation — radix select, CUB sort, atomic emit — produced a valid result that the evaluator rejected as `INCORRECT_NUMERICAL` because the index ordering didn't match the reference's tie-breaking. The only path that reliably passed was `torch::topk`. So that was the ceiling: a fused FP8 page-gather kernel feeding ATen's top-K. Ceiling: roughly 7×.

During this phase I leaned on Cursor cloud agents running asynchronously — multiple branches, multiple hypotheses in parallel, reviews happening whenever results came in. The agents mostly pushed back on anything too aggressive, which in retrospect was correct given the evaluator constraint. With strong human supervision it was possible to extract some speedup from the Python baseline, but the custom CUDA pipeline was blocked.

I bumped my Cursor subscription to $60, then $200. The billing chart was modest but growing.

## Phase 2 — Two weeks, everything unlocked

Then PR #354 landed in flashinfer-bench. A new `DsaTopkIndexerEvaluator` that compared sorted value vectors, checked block-table reachability, and provided a correct FP8 baseline. Overnight, the constraint that had blocked six weeks of work disappeared.

The billing chart tells the rest of the story. When Opus 4.7 dropped and the contest leaderboard results became visible, the spend spiked hard. The agents were cooking.

In two weeks, the indexer went from constrained ATen ops to a full SM100a UMMA pipeline: `tcgen05.mma` logit scoring, three-path host dispatch, warp-specialized persistent kernel with producer/math warpgroup handoff over mbarrier, Stage-2 radix top-K with a 32 KB SMEM ordered-key cache, and a CUDA graph cache that collapsed fast-path latency from 6.5 µs to 2.3 µs. The attention kernel got a fused online-softmax K-loop, a CTA-per-token WMMA path for H=16 shapes, and split-K with FP32 partial buffers. Final result: 38.4× indexer, 14.18× attention, 28.96× geometric mean across the track.

This phase was synchronous and tight. Short iteration loops, human in the loop on every measurement, no hypothesis surviving longer than it took to benchmark it.

## The playbook

Looking back, six things made the difference — and one that should have.

**Fast feedback loop.** Smoke tests for correctness and debugging, performance benchmarks only when the implementation was ready. Running the full test suite during exploration is a way to feel productive while being slow.

**Incremental hypothesis.** One change at a time, measured on real B200 hardware. If numbers don't improve, revert immediately. Agent confidence is not evidence; measurements are.

**Parallel agents on branches.** In Phase 1, Cursor cloud agents explored multiple directions concurrently while I reviewed results asynchronously. In Phase 2, time pressure forced a single synchronous loop — but the branch-per-hypothesis discipline carried over.

**Every major refactoring gets its own branch.** Don't mix a structural rewrite with a micro-optimization. Run them separately, compare clean baselines. This rule saved several cycles where combined changes masked whether anything was actually helping.

**Auto-track what worked and what didn't.** Every experiment was logged. Negative results were treated as first-class artifacts. About 50 micro-optimizations were tried and reverted; knowing what was already ruled out prevented re-exploring dead ends.

**Human as judge.** Accept or reject based on benchmark numbers only — never on the agent's stated confidence.
**Revert aggressively.** If it doesn't measure better, it goes. No "promising but needs more work." A clean baseline makes the next hypothesis easier to evaluate.

## What could have been better

The cupti measurement infrastructure was only properly fixed on the last day before submission. An unknown number of hypotheses across both phases were rejected on imprecise numbers — some of those might have been worth keeping. The CUDA graph cache speedup is a concrete example: the fast-path improvement from 6.5 µs to 2.3 µs was measured alongside the cupti fix, so the two effects are confounded. CUDA graphs might be a candidate for reverting. Fixing the measurement environment early is at least as important as fixing the kernel.

Cursor cloud agents were underused overall. There were stretches, especially in Phase 2, where a single-threaded human-agent loop became the bottleneck. More parallel exploration — even just running two branches overnight — would have bought back time. New agentic harnesses like OpenClaw are worth watching here, though OpenClaw itself didn't work well for low-level CUDA work when tried. Not every tool is the right tool — and learning to use Claude more efficiently is still an open item.

## TL;DR

1. **Baseline matters.** An immature evaluator capped six weeks of work at 7×. A mature one unlocked 38.4× in two weeks.
2. **Environment matters.** Fast feedback loop, smoke tests, real hardware access, correct measurement tooling — without these, agents iterate blind.
3. **Compute matters.** About $500 in API + Modal B200 compute spend, and a $200/month Cursor subscription. The cost of getting to 3rd place internationally.
4. **Agents amplify you by more than 10×.** I am not a kernel engineer by trade. Warp specialization, TMEM, inline PTX, radix sort — all agent-generated diffs, all accepted only when the numbers improved. Without modern tools, this result would not have been possible.
