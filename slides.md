---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
    color: #1e293b;
    padding: 44px 56px;
    font-size: 18px;
  }
  h1, h2, h3 { margin: 0; }

  /* ── 1. Title ── */
  section.title {
    display: flex; flex-direction: column; justify-content: center;
    background: linear-gradient(135deg, #f8fafc 55%, #eff6ff);
  }
  section.title h1 {
    font-size: 2.3em; line-height: 1.15; color: #0f172a; margin-bottom: 18px;
  }
  section.title h1 em { font-style: normal; color: #2563eb; }
  section.title .sub { font-size: 0.9em; color: #64748b; margin-bottom: 28px; }
  section.title .badge {
    display: inline-block;
    border: 1.5px solid #2563eb66; border-radius: 8px;
    padding: 10px 22px; color: #2563eb; font-size: 1.05em; font-weight: 700;
    background: #2563eb11;
  }

  /* ── 2. Story ── */
  section.story { padding: 28px 44px; font-size: 16px; }
  section.story h2 { font-size: 1.3em; color: #0f172a; border-bottom: 2px solid #2563eb33; padding-bottom: 6px; margin-bottom: 12px; }
  section.story .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  section.story .phase {
    background: #f1f5f9; border-radius: 8px; padding: 10px 14px;
    border-left: 3px solid #cbd5e1;
  }
  section.story .phase.hot { border-left-color: #2563eb; }
  section.story .phase h3 { font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 4px; }
  section.story .phase .tag { font-size: 0.92em; font-weight: 700; color: #0f172a; margin-bottom: 4px; line-height: 1.25; }
  section.story .phase ul { list-style: none; padding: 0; margin: 0; }
  section.story .phase ul li { font-size: 0.72em; color: #475569; padding: 1px 0 1px 12px; position: relative; line-height: 1.3; }
  section.story .phase ul li::before { content: "▸"; position: absolute; left: 0; color: #94a3b8; }
  section.story .phase.hot ul li::before { color: #2563eb88; }
  section.story .phase ul li ul { margin: 2px 0 0 0; padding: 0; }
  section.story .phase ul li ul li { font-size: 0.95em; color: #64748b; padding-left: 12px; }
  section.story .phase ul li ul li::before { content: "·"; color: #cbd5e1; }
  section.story .phase .result { margin-top: 6px; font-size: 0.82em; font-weight: 700; }
  section.story .phase .result.bad { color: #dc2626; }
  section.story .phase .result.good { color: #2563eb; }
  section.story .chart-kicker {
    margin-top: 10px; text-align: center; font-size: 0.76em; color: #64748b;
  }
  section.story .chart-kicker strong { color: #d97706; }
  section.story .charts {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 6px;
  }
  section.story .charts figure { margin: 0; }
  section.story .charts img {
    width: 100%; max-height: 168px; object-fit: contain;
    border-radius: 6px; border: 1px solid #e2e8f0; display: block;
    background: #fff;
  }
  section.story .charts figcaption {
    font-size: 0.65em; color: #64748b; text-align: center; margin-top: 3px;
  }

  /* ── 3. Playbook ── */
  section.recipe h2 { font-size: 1.45em; color: #0f172a; border-bottom: 2px solid #2563eb33; padding-bottom: 8px; margin-bottom: 22px; }
  section.recipe .recipe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  section.recipe .card {
    background: #f8fafc; border-radius: 10px; padding: 14px 16px;
    border-top: 2px solid #2563eb44;
  }
  section.recipe .card .num { font-size: 1.4em; font-weight: 800; color: #2563eb33; float: right; }
  section.recipe .card h3 { font-size: 0.88em; font-weight: 700; color: #0f172a; margin-bottom: 5px; }
  section.recipe .card p { font-size: 0.76em; color: #64748b; margin: 0; line-height: 1.45; }

  /* ── 4. Tech ── */
  section.tech { padding: 36px 48px; }
  section.tech h2 { font-size: 1.35em; color: #0f172a; border-bottom: 2px solid #2563eb33; padding-bottom: 8px; margin-bottom: 16px; }
  section.tech .cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
  section.tech .col h3 { font-size: 0.82em; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  section.tech ul { list-style: none; padding: 0; margin: 0; }
  section.tech ul li { font-size: 0.75em; color: #475569; padding: 2.5px 0 2.5px 13px; position: relative; line-height: 1.4; }
  section.tech ul li::before { content: "▸"; position: absolute; left: 0; color: #2563eb55; }
  section.tech ul li strong { color: #0f172a; }
  section.tech .bar {
    margin-top: 16px; background: #eff6ff; border: 1.5px solid #2563eb33;
    border-radius: 8px; padding: 10px 20px;
    display: flex; justify-content: space-around; align-items: center;
  }
  section.tech .bar .m { text-align: center; }
  section.tech .bar .m .v { font-size: 1.55em; font-weight: 800; color: #2563eb; }
  section.tech .bar .m .l { font-size: 0.68em; color: #64748b; }
  section.tech .bar .sep { width: 1px; height: 34px; background: #e2e8f0; }

  /* ── 5. TLDR ── */
  section.tldr {
    display: flex; flex-direction: column; justify-content: center;
    background: linear-gradient(135deg, #f8fafc 55%, #f0fdf4);
  }
  section.tldr h2 { font-size: 1.5em; color: #0f172a; margin-bottom: 28px; }
  section.tldr .points { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 28px; }
  section.tldr .point {
    background: #f1f5f9; border-radius: 10px; padding: 16px 20px;
    border-left: 3px solid #05966966;
  }
  section.tldr .point .num { font-size: 0.78em; color: #059669; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }
  section.tldr .point .text { font-size: 1.0em; color: #0f172a; font-weight: 600; }
  section.tldr .point .sub { font-size: 0.78em; color: #64748b; margin-top: 3px; }
  section.tldr .kicker { font-size: 1.0em; color: #059669; font-weight: 700; text-align: center; margin-bottom: 14px; }
  section.tldr .next-time { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  section.tldr .next-time .card {
    background: #fffbeb; border: 1px solid #fcd34d66; border-radius: 8px;
    padding: 10px 14px; text-align: center;
  }
  section.tldr .next-time .card .label { font-size: 0.65em; color: #d97706; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }
  section.tldr .next-time .card .text { font-size: 0.75em; color: #78350f; }
  section.tldr .footer { font-size: 0.72em; color: #94a3b8; text-align: center; }

  /* ── 6. Thank you ── */
  section.thanks {
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    text-align: center; background: linear-gradient(135deg, #f8fafc 55%, #eff6ff);
  }
  section.thanks h2 { font-size: 2.2em; color: #0f172a; margin-bottom: 8px; }
  section.thanks .contact { font-size: 0.85em; color: #64748b; margin-bottom: 32px; }
  section.thanks .orgs { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 18px; width: 100%; margin-bottom: 28px; }
  section.thanks .org { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; align-items: center; gap: 10px; }
  section.thanks .org img { height: 36px; object-fit: contain; }
  section.thanks .org .label { font-size: 0.72em; color: #64748b; line-height: 1.4; }
  section.thanks .org .role { font-size: 0.78em; font-weight: 600; color: #0f172a; }
  section.thanks .links { font-size: 0.72em; color: #94a3b8; }
  section.thanks .links a { color: #2563eb; text-decoration: none; }
---

<!-- _class: title -->

# Agent-Assisted Kernel Engineering:<br>*What Actually Worked*

<div class="sub">
Yury Kirpichev, on behalf of Team Wombat &nbsp;·&nbsp; MLSys 2026 · FlashInfer Contest · DSA Track
</div>

<div class="badge">3rd place &nbsp;·&nbsp; 28.96× geomean speedup &nbsp;·&nbsp; NVIDIA B200 (sm_100a)</div>

---

<!-- _class: story -->

## Two phases

<div class="grid">

<div class="phase">
<h3>Phase 1 · Cursor cloud agents</h3>
<div class="tag">Mar – early Apr · old evaluator</div>
<ul>
  <li>Focused only on DSA Indexer kernel</li>
  <li>Cursor $20/mo → $60 (Apr 4)</li>
  <li>Claude $20/mo</li>
  <li>Cursor cloud agents, async supervision mode, many branches</li>
  <li>Operations must be numerically stable to pass evaluation</li>
</ul>
<div class="result bad">Ceiling: ~7× speedup</div>
</div>

<div class="phase hot">
<h3>Phase 2 · tight sync loop · 2 weeks</h3>
<div class="tag">Apr 21 – May · tight sync loop</div>
<ul>
  <li>Cursor bumped to $200/mo</li>
  <li>Claude $20/mo</li>
  <li>Started using Opus 4.7 (was available at 50% price)</li>
  <li>Interim results email: top 3 at ~11× DSA</li>
  <li>Updated evaluator → full custom CUDA pipeline:
    <ul>
      <li>UMMA kernels, warp specialization, radix select, inline PTX…</li>
    </ul>
  </li>
</ul>
<div class="result good">38.4× indexer &nbsp;·&nbsp; 14.18× attention &nbsp;·&nbsp; 28.96× track</div>
</div>

</div>

<div class="chart-kicker">
</div>

<div class="charts">
<figure>
<img src="images/diagrams/diag2.png" alt="Indexer mean speedup by submission milestone" />
<figcaption>Indexer speedup — ~7× plateau, then cliff after PR #354</figcaption>
</figure>
<figure>
<img src="images/cursor_usage.png" alt="Cursor cumulative spend by model" />
<figcaption>Switched to Opus 4.7 + interim results email (top 3, ~11× DSA) → spending spike</figcaption>
</figure>
</div>

---

<!-- _class: tech -->

## What the agents produced

<div class="cols">
<div class="col">
<h3>Top-K Indexer (FP8)</h3>
<ul>
  <li><strong>FP8 UMMA</strong> logit scoring — <code>tcgen05.mma</code> (M=128,N=64,K=32)</li>
  <li><strong>3-path host dispatch:</strong> fast / short / warp-spec persistent</li>
  <li>Fast path: block-table only, int4 stores</li>
  <li>Warp specialization: producer 40 regs ↔ math 232 regs, mbarrier handoff</li>
  <li>Stage-2 <strong>radix top-K,</strong> 32 KB SMEM ordered-key cache</li>
  <li>Critical path = TMEM readout, not MMA (confirmed by measurement)</li>
</ul>
</div>
<div class="col">
<h3>Sparse Attention (BF16)</h3>
<ul>
  <li><strong>Fused online-softmax K-loop</strong> — one HBM pass per KV tile</li>
  <li><strong>CTA-per-token WMMA</strong> for H=16 (tensor-core shaped)</li>
  <li>Split-K with FP32 partial buffers + reducer for underfill grids</li>
  <li>GPU density pre-scan → route high/low-density shapes</li>
  <li>WMMA beats UMMA here: skinny head GEMM, overhead > benefit</li>
</ul>
</div>
<div class="col">
<h3>Shared</h3>
<ul>
  <li>Shape-aware host dispatch everywhere</li>
  <li>~50 micro-opts measured and reverted; only wins kept</li>
</ul>
</div>
</div>

<div class="bar">
  <div class="m"><div class="v">38.4×</div><div class="l">Indexer vs FlashInfer</div></div>
  <div class="sep"></div>
  <div class="m"><div class="v">14.18×</div><div class="l">Attention vs DeepGEMM</div></div>
  <div class="sep"></div>
  <div class="m"><div class="v">28.96×</div><div class="l">Geomean · DSA Track</div></div>
  <div class="sep"></div>
  <div class="m"><div class="v">151/151</div><div class="l">Workloads PASS</div></div>
</div>

---

<!-- _class: recipe -->

## The playbook

<div class="recipe-grid">

<div class="card">
<span class="num">1</span>
<h3>Fast feedback loop</h3>
<p>Smoke tests for correctness, perf tests only when implementation is ready. Never run the full suite during exploration.</p>
</div>

<div class="card">
<span class="num">2</span>
<h3>Incremental hypothesis</h3>
<p>One change at a time, measured on real B200 hardware. Reject immediately if numbers don't improve.</p>
</div>

<div class="card">
<span class="num">3</span>
<h3>Explore in parallel</h3>
<p>Cursor cloud agents exploring different hypotheses concurrently. Phase 1: async review. Phase 2: serial tight loop.</p>
</div>

<div class="card">
<span class="num">4</span>
<h3>Auto-track +/−</h3>
<p>Every experiment logged. Negative results treated as first-class — ~50 micro-opts tried and reverted, all documented.</p>
</div>

<div class="card">
<span class="num">5</span>
<h3>Numbers decide</h3>
<p>Accept/reject based on measured numbers only — agent confidence is not evidence.</p>
</div>

<div class="card">
<span class="num">6</span>
<h3>Revert aggressively</h3>
<p>If it doesn't measure better, it goes. No "promising but needs more work." Keeps the baseline clean for the next hypothesis.</p>
</div>

</div>

---

<!-- _class: tldr -->

## TL;DR

<div class="points">

<div class="point">
<div class="num">1</div>
<div class="text">Baseline</div>
<div class="sub">Start from the strongest available baseline. Every optimization is only as meaningful as what you measure it against.</div>
</div>

<div class="point">
<div class="num">2</div>
<div class="text">Environment</div>
<div class="sub">Smoke tests + B200 access + fast iterations = agents outpace manual development.</div>
</div>

<div class="point">
<div class="num">3</div>
<div class="text">Compute</div>
<div class="sub">Modal B200 compute, $200/mo Cursor. Unlocked warp specialization, TMEM, inline PTX, radix sort — all agent-generated.</div>
</div>

<div class="point">
<div class="num">4</div>
<div class="text">Agents</div>
<div class="sub">Amplify you &gt;10× — and that's conservative. With the right setup, agents redefine what's possible.</div>
</div>

</div>

<div class="kicker">Agents are ready. Is your workflow?</div>

<div class="next-time">
<div class="card">
<div class="label">Next time</div>
<div class="text">Fix measurement tooling on day one — bad benchmarks reject good ideas</div>
</div>
<div class="card">
<div class="label">Next time</div>
<div class="text">Use Cursor cloud agents and Claude more aggressively — still learning the right workflow</div>
</div>
</div>

<div class="footer">
github.com/ykirpichev/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 &nbsp;·&nbsp;
github.com/ykirpichev/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
</div>

---

<!-- _class: thanks -->

## Thank You

<div class="contact">mlsys26-contest-contact@nvidia.com &nbsp;·&nbsp; discord.gg/XEH5wGKXZA (GPU Mode #flashinfer)</div>

<div class="orgs">

<div class="org">
<img src="images/mlsys-logo.svg" alt="MLSys" />
<div class="role">Conference Host</div>
<div class="label">MLSys 2026<br>Bellevue, WA · May 18–22</div>
</div>

<div class="org">
<img src="images/nvidia-logo.svg" alt="NVIDIA" />
<div class="role">Contest Organizer</div>
<div class="label">FlashInfer AI Kernel Generation Contest</div>
</div>

<div class="org">
<img src="images/flashinfer-logo.png" alt="FlashInfer" />
<div class="role">Benchmark & Baseline</div>
<div class="label">FlashInfer team &amp; flashinfer-bench</div>
</div>

<div class="org">
<img src="images/modal-logo.png" alt="Modal" />
<div class="role">Compute Sponsor</div>
<div class="label">B200 GPU access for<br>development &amp; benchmarking</div>
</div>

</div>

<div class="links">
mlsys.org &nbsp;·&nbsp; mlsys26.flashinfer.ai
</div>
