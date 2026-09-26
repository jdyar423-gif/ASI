# China as a Natural Experiment in Low-Compute AI: Export Controls, Compute-Efficiency Innovation, Brain-Inspired/Neuromorphic and World-Model Programs (as of Sept 2026)

Method and labels. epoch.ai, arxiv.org, huggingface.co, cset.georgetown.edu, chinatalk, cfr.org, fortune/techcrunch/yahoo, zju.edu.cn and gov.cn were blocked for direct fetch in this session. Most facts therefore come from one of three routes:
- primary GitHub READMEs and PDFs, fetched directly and marked **[S]**. These cover DeepSeek V3.2/Engram, Kimi K2/K2.5/K3, Qwen3.5–3.8, GLM-5.x, MiniMax M1/M2/M2.5, MiniCPM, SpikingBrain 1.0/2.0, Emu3.5 and RoboBrain;
- anthropic.com, fetched directly and marked [S];
- search-engine extracts, marked **(snippet)**.

Other labels:
- **[S2]** a secondary source quoting a primary one.
- **[E]** my own arithmetic.
- **[A]** my own assumption.
- **Vendor** means a result the lab reported about its own work. **Independent** means a third party measured it.

Arithmetic conventions: training FLOP ≈ 6 × active params × tokens; inference ≈ 2 × active params per token. The earlier notes already cover the DeepSeek-V3/R1 basics, Qwen3-Next, Kimi Linear, Moonlight/Muon, Engram at a high level, the densing law and the Epoch compute ledger. They are referenced here, not repeated.

---

## Q1. Export-control timeline and the compute actually available to Chinese labs vs US frontier labs (2022–2026)

### Takeaway
Chinese labs have operated with roughly **5–15× less installed AI compute** than US labs. Epoch's supercomputer data give about 15% vs 75% of global GPU-cluster performance. Hyperscaler capex differs by roughly 10×: about $30B for ByteDance vs about $660–690B for the top five US firms in 2026. Chinese domestic accelerators are about **one generation behind per chip**, being about 3× weaker in BF16 than GB200 and about 2.3× less power-efficient per FLOP at system level. They are also **HBM-bottlenecked**. Policy swung repeatedly in 2025–26: the H20 was banned, then re-allowed with a 15% revenue share, then discouraged by Beijing, and the H200 was approved in December 2025 / January 2026 but stalled. Smuggling (Epoch median about 660k H100e cumulative through 2025) plausibly supplied about a third of China's stock. Estimates of the absolute gap disagree by more than an order of magnitude.

### Cited Findings
**Timeline of controls and reversals**
- **Oct 7, 2022**: BIS imposed PRC-wide licensing on advanced computing chips (A100/H100 class).
- **Oct 17, 2023** (effective **Nov 16, 2023**): the rule was tightened and the A800/H800 "loophole" chips were closed off. — [CSIS](https://www.csis.org/analysis/insight-us-semiconductor-export-controls-update); [CSET explainer](https://cset.georgetown.edu/article/bis-2023-update-explainer/); [Federal Register 2023-23055](https://www.federalregister.gov/documents/2023/10/25/2023-23055/implementation-of-additional-export-controls-certain-advanced-computing-items-supercomputer-and) (snippet)
  - DeepSeek-V3's 2.8M H800-hours ran on chips that "were allowed to be sold to China for a few months in 2023 before a later rule halted sales". — [Yahoo/OpenAI memo coverage](https://finance.yahoo.com/news/openai-accuses-deepseek-distilling-us-221629899.html) (snippet)
- **Apr 2025**: the US restricted even the export-compliant H20, and Nvidia took a multibillion-dollar writedown. — [Built In](https://builtin.com/articles/trump-lifts-ai-chip-ban-china-nvidia) (snippet)
- **Jul–Aug 2025**: H20 licenses were promised. Nvidia and AMD agreed to pay **15% of China chip revenue** to the US government. — [NPR, Aug 11, 2025](https://www.npr.org/2025/08/11/nx-s1-5498689/trump-nvidia-h20-chip-sales-china); [CNBC/FT, Aug 10, 2025](https://www.cnbc.com/2025/08/10/nvidia-amd-15percent-of-china-chip-sales-revenues-to-us-ft-reports.html) (snippet)
- **Aug 2025**: Beijing urged firms to avoid the H20. The CAC summoned Nvidia over alleged "backdoor" risks, and Tencent paused H20 purchases. — [Bloomberg, Aug 12, 2025](https://www.bloomberg.com/news/articles/2025-08-12/china-urges-firms-not-to-use-nvidia-h20-chips-in-new-guidance); [Trivium China](https://triviumchina.com/2025/08/18/tencent-pauses-h20-buys-amid-cac-scrutiny/) (snippet)
- **Dec 8, 2025**: Trump announced that H200s may ship to approved Chinese customers, with a 25% revenue share.
- **Jan 15, 2026**: a BIS rule moved H200/MI325X licensing from "presumption of denial" to "case-by-case", described as a **13× increase in permitted computing power**.
- Reports then said about **400k approved H200s sat unshipped** because of a Commerce/State dispute, and that Chinese customs told agents H200 imports were not permitted. — [Introl](https://introl.com/blog/bis-h200-china-export-policy-ai-overwatch-act-2026); [BIS press release](https://www.bis.gov/press-release/department-commerce-revises-license-review-policy-semiconductors-exported-china); [TechRadar](https://www.techradar.com/pro/trump-set-to-allow-nvidia-h200-chips-to-be-exported-to-china) (snippet)

**Installed compute and capex**
- Epoch: the US holds about **75%** of global GPU-cluster performance and China about **15%** (May 2025). The observed dataset covers only about 10–20% of global aggregate compute. — [Epoch data insight](https://epoch.ai/data-insights/ai-supercomputers-performance-share-by-country); [Epoch trends in AI supercomputers](https://epoch.ai/blog/trends-in-ai-supercomputers) (snippet)
- Conflicting absolute estimates (all snippet):
  - "US ~850k H100e vs China ~110k (2025)", about 9×;
  - "US 39.7M vs China 400k H100e (May 2025)", about 100×. The methodology is unclear and probably not like-for-like;
  - MIIT's "effective compute index" says China held about **1.58M H100-equivalents** (dense FP16) as of June 30, 2026. — [search aggregate: nextbigfuture / axis-intelligence / CFR](https://www.cfr.org/articles/new-ai-chip-export-policy-china-strategically-incoherent-and-unenforceable) (snippet)
  - Another summary states China trails "~10× in installed capacity and ~35× in advanced chip manufacturing". — [Medium (A. Masood)](https://medium.com/@adnanmasood/the-measured-distance-quantifying-the-us-china-frontier-ai-gap-176e3a0ad2ed) (snippet)
  - Only the Epoch 15% vs 75% share figure comes from a primary methodology I can identify.
- Capex, 2026:
  - ByteDance about RMB 160B (about $23B), raised to over RMB 200B (about $29B), with about $13B for AI processors.
  - Alibaba RMB 380B (about $53B) over 3 years.
  - Tencent over RMB 36B.
  - Versus Amazon about $200B, Alphabet $175–185B, Meta $115–135B and Microsoft about $120B+; the five US firms together are about **$660–690B**. — [Futurum](https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/); [TrendForce, May 6, 2026](https://www.trendforce.com/presscenter/news/20260506-13033.html); [BigGo Finance](https://finance.biggo.com/news/lfE0EZ4BYH_ypPqOu9ei) (snippet)
- US frontier cluster: **xAI Colossus 2**, about 550–555k GB200/GB300 GPUs at about 1 GW (operational Jan 2026). — [Introl](https://introl.com/blog/xai-colossus-2-gigawatt-expansion-555k-gpus-january-2026) (snippet)
- Epoch's scaling-rate finding: the top-10 Chinese LLMs by training compute have grown about **3×/yr since late 2021**, versus about 5×/yr elsewhere. Grok-2 used about 2× the compute of China's largest known model, Doubao-pro. — [Epoch: China compute trends](https://epoch.ai/data-insights/china-compute-trends) (snippet; dated before 2026)

**Domestic chips (Huawei Ascend)**
- 910C output target of about **600k units in 2026**, about 2× 2025, and up to **1.6M Ascend dies** in total. The 950PR is due Q1 2026 (750k-unit target) and the 950DT late 2026 (about 100k).
- **HBM is the bottleneck**: CXMT is projected to make only about 2M HBM stacks in 2026, enough for about **250–300k 910C-equivalent packages**. SMIC's improved 7nm-class process makes the dies. — [Invezz/Bloomberg](https://invezz.com/news/2025/09/29/huawei-to-double-ai-chip-output-in-2026-targeting-1-6-million-dies/); [SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp); [tech-insider 950PR](https://tech-insider.org/huawei-ascend-950pr-ai-chip-nvidia-china-2026/) (snippet)
- Per chip, the 910C does about **780 TFLOPs BF16 vs 2,500** for GB200, with 128 vs 192 GB of memory and 3.2 vs 8 TB/s of bandwidth.
- At system level, **CloudMatrix 384** (384× 910C) gives about **300 PFLOPs BF16 vs 180** for GB200 NVL72, but draws **about 560 kW vs 145 kW**. That makes it **2.3× less power-efficient per FLOP**; SemiAnalysis calls it "a generation behind per chip, ahead in scale-up". — [SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72); [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power) (snippet)

**Smuggling**
- **Epoch** ("Diversion and resale"): **290k–1.6M H100e** smuggled through 2025, **median 660k**, "roughly a third of China's total compute".
- A CNAS-style Monte Carlo for 2025 alone gives a **median of 312k H100e** (90% CI 176k–565k). — [Epoch chip-smuggling](https://epoch.ai/publications/chip-smuggling); [CNAS](https://www.cnas.org/publications/reports/countering-ai-chip-smuggling-has-become-a-national-security-priority) (snippet)
- DOJ case: at least $160M of H100/H200s allegedly smuggled between Oct 2024 and May 2025. — [CNBC, Dec 31, 2025](https://www.cnbc.com/2025/12/31/160-million-export-controlled-nvidia-gpus-allegedly-smuggled-to-china.html) (snippet)

**Lab-level hardware evidence**
- DeepSeek R2 was delayed after failed training runs on Ascend (FT, Aug 2025). DeepSeek reportedly reverted to Nvidia for training and used Ascend for inference. — [TrendForce](https://www.trendforce.com/news/2025/08/14/news-deepseek-r2-model-launch-reportedly-delayed-amid-huawei-ascend-chip-hurdles/); [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-reportedly-urged-by-chinese-authorities-to-train-new-model-on-huawei-hardware-after-multiple-failures-r2-training-to-switch-back-to-nvidia-hardware-while-ascend-gpus-handle-inference) (snippet)
- The Information (Dec 2025) reported that DeepSeek was using **2,000–2,300 smuggled B200/B100s**. A US official later said V4 was built on Blackwells in an Inner Mongolia data center. **Nvidia called this "farfetched"; DeepSeek denies it and says V4 used H800s plus Ascend 910C.** — [CNBC](https://www.cnbc.com/2025/12/10/nvidia-report-china-deepseek-ai-blackwell-chips.html); [Let's Data Science](https://letsdatascience.com/blog/deepseek-v4-smuggled-nvidia-blackwell-chips-inner-mongolia) (snippet; contested)
- A Huawei-led report (July 22, 2026) describes full-parameter **post-training** of DeepSeek-V4-Pro on about **1,000 Ascend 910C** at **34.22% MFU**. It does **not** establish what pre-trained V4; Tsinghua's Liu Zhiyuan told MIT Tech Review it was likely mainly Nvidia. — [arXiv 2607.20145 "SLAI T-Rex"](https://arxiv.org/pdf/2607.20145); [implicator.ai](https://www.implicator.ai/deepseeks-huawei-chip-training-claim-finally-gets-its-benchmarks-and-its-doubters/) (snippet)
- GLM-5 (744B, Feb 11, 2026) is said by secondary sources to have been **trained entirely on about 100,000 Huawei Ascend chips**. Zhipu's own README does not state this. — [Let's Data Science](https://letsdatascience.com/blog/china-trained-frontier-ai-model-glm-5-without-nvidia) (S2, unverified)
- Kimi K3's README says several agentic evaluations were run "on H20 GPUs (instead of H100 in the official setting)" and "recalibrated for H20". This is direct evidence of which Nvidia part a top Chinese lab has in quantity. — [Kimi-K3 README](https://github.com/MoonshotAI/Kimi-K3) [S]
- SpikingBrain (CAS) trained on "hundreds of MetaX C550 GPUs" at 23.4% MFU, versus 25.8% on an A800 cluster with the same configuration. — [SpikingBrain report](https://github.com/BICLab/SpikingBrain-7B) [S]

### Inferences
- Across credible sources the installed-compute ratio is roughly **5× (Epoch share) to 10× (capex, H100e)**, and larger for frontier-scale single clusters: Colossus 2 at about 550k Blackwells vs reported Chinese single-run clusters of 10³–10⁵ chips. The binding constraints look **operational rather than absolute**: HBM, per-chip efficiency, software maturity (R2's Ascend failures) and policy whiplash. Chinese labs appear to pretrain mainly on legacy or grey-market Nvidia (H800/H20, allegedly Blackwell) and to use Ascend increasingly for inference and post-training.
- For the "natural experiment" framing, the treatment is **about 1 OOM less compute, not 3–6 OOM**. Any conclusion about "extremely low compute" must extrapolate far beyond the observed range.

### Gaps
- There is no reconcilable, like-for-like US vs China H100e stock for 2026. Estimates range from about 9× to about 100×, and the MIIT's 1.58M figure uses an undisclosed "effective index".
- The Dec 2024 HBM/entity-list controls and the status of the 2026 H200 shipments were not re-verified from primary sources.
- The hardware used for DeepSeek-V4, GLM-5 and Kimi K3 pretraining remains undisclosed or contested.

---

## Q2. Efficiency innovations from Chinese labs (2024–Sept 2026): measured multipliers, novelty vs adoption, and performance per training FLOP vs the US frontier

### Takeaway
Chinese labs produced a dense stream of **sparsity-and-memory innovations**:
- *compute sparsity*: fine-grained MoE with 16/896 experts;
- *attention sparsity*: NSA, DSA, CSA/HCA, IndexShare, MoBA, and the linear/hybrid KDA, Gated DeltaNet and Lightning designs;
- *residual-stream*: mHC, AttnRes;
- *memory*: Engram;
- *precision*: FP8 at scale, INT4/MXFP4/FP4 QAT;
- *RL-algorithm*: GRPO, CISPO, async RL.

Individually each is a vendor-reported multiplier of about **1.1–2.5× for training** and **3–20× for long-context inference**. Several are genuinely original and have been adopted across labs (MLA, GRPO, DSA, mHC, Engram, KDA, AttnRes). Others adapt Western ideas (Muon, Gated DeltaNet, MTP, linear attention). Chinese flagship pretraining compute is about **5e24–2e25 FLOP**, versus US frontier estimates of about **5e25–5e26+**. That is roughly **1–1.5 OOM less** for models about 4–8 months behind.

### Cited Findings
**DeepSeek**
- **V3.2-Exp (Sept 29, 2025) → V3.2 (Dec 2025): DeepSeek Sparse Attention (DSA).**
  - A "lightning indexer" (FP8) selects the top-k tokens (one secondary description says about 64–128; the primary value was not verified), making attention O(L·k). One secondary source claims about 0.125 GB of KV vs 67 GB for dense attention at 128K (unverified). — [Medium explainer](https://kchandan.medium.com/llm-cost-engineering-how-deepseek-v3-2-could-cut-llm-inference-costs-9b147124f109) (snippet)
  - Training was "deliberately aligned" with V3.1-Terminus, and benchmarks came out "on par" (vendor). — [DeepSeek-V3.2-Exp README](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) [S]; [DeepSeek API news](https://api-docs.deepseek.com/news/news250929/)
  - API prices were cut about 50% (secondary). — [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/09/deepseek-v3-2-exp/) (snippet)
  - V3.2's **post-training RL budget "exceeds 10% of the pre-training cost"**. The RL ran over **1,800+ environments and 85,000+ agent tasks**. — [DeepSeek-V3.2 paper, arXiv 2512.02556](https://arxiv.org/html/2512.02556v1); [AI News](https://www.artificialintelligence-news.com/news/deepseek-v3-2-matches-gpt-5-lower-training-costs/) (snippet)
- **mHC (Manifold-Constrained Hyper-Connections, arXiv 2512.24880, Dec 31, 2025).**
  - Residual mixing matrices are projected onto the Birkhoff polytope (doubly stochastic, via Sinkhorn–Knopp). In a 27B model, unconstrained hyper-connections gave >3,000× signal gain and diverged; mHC fixes this.
  - It reports **+2.1% on BBH for 6.7% extra training overhead** across 3B/9B/27B models (vendor).
  - It was **adopted by DeepSeek-V4 and by Zhipu's GLM-5.3-Flash**. — [arXiv 2512.24880](https://arxiv.org/pdf/2512.24880) (snippet); [GLM-5 README](https://github.com/zai-org/GLM-5) [S]
  - The base idea, hyper-connections, came from ByteDance Seed. — [arXiv 2409.19606](https://arxiv.org/abs/2409.19606)
- **Engram (Jan 2026) [S, from the paper PDF in the repo].**
  - The iso-parameter, iso-FLOP comparison is Engram-27B vs MoE-27B, both with 3.8B activated params and 262B tokens.
  - Gains: **MMLU +3.4, CMMLU +4.0, MMLU-Pro +1.8, BBH +5.0, ARC-C +3.7, DROP +3.3, HumanEval +3.0, MATH +2.4, GSM8K +2.2**. Multi-Query NIAH went **84.2→97.0**.
  - "With only **82% of the pre-training FLOPs** (41k vs 50k [steps]), Engram-27B matches the baseline" in the long-context setting, which is about a **1.2× compute-equivalent**.
  - A **100B-parameter table offloaded to host memory costs under 3% overhead**. — [Engram repo and paper](https://github.com/deepseek-ai/Engram)
- **DeepSeek-V4 (tech report arXiv 2606.19348, June 2026).**
  - V4-Pro: **1.6T total / 49B active**. V4-Flash: 284B / 13B. Pretrained on **more than 32T tokens**, 1M context. Uses hybrid **CSA + HCA** attention, mHC, and **the Muon optimizer "for the majority of modules"**. FP4 QAT was applied to MoE experts and the indexer during post-training.
  - At 1M context, V4-Pro uses **27% of V3.2's single-token FLOPs and 10% of its KV cache**; V4-Flash uses **10% of the FLOPs and 7% of the KV**. KV cache is about **2% of a BF16 GQA8 baseline** (vendor). — [arXiv 2606.19348](https://arxiv.org/abs/2606.19348); [HF blog](https://huggingface.co/blog/deepseekv4) (snippet)
  - Release timing conflicts: an April 2026 release was expected ([TechXplore](https://techxplore.com/news/2026-04-deepseek-china-ai-ambitions.html)); an aggregator reports V4-Flash launched July 31 and V4-Pro Aug 13, 2026 ([TrendForce/aggregator](https://insights.trendforce.com/p/kimi-k3-us-china-ai-gap), snippet).
  - **Training compute is undisclosed.** A follow-up, **DeepSeek-V4.1-Flash** (arXiv 2609.19969, Sept 2026), continues KV compression. — (snippet)

**Moonshot (Kimi)**
- **Kimi K2 (July 2025):** 1T/32B, 15.5T tokens, MuonClip, "zero training instability". **K2.5** added about 15T more multimodal tokens of continued pretraining plus "Agent Swarm" parallel sub-agents. — [Kimi-K2 README](https://github.com/MoonshotAI/Kimi-K2); [Kimi-K2.5 README](https://github.com/MoonshotAI/Kimi-K2.5) [S]
- **K2 Thinking (Nov 2025):** INT4 QAT gave about 2× output speed and a 594 GB file versus about 1 TB. A **"$4.6M" training cost** comes from an unnamed source; **CEO Yang Zhilin said it "is not an official number"**. — [DeepLearning.AI](https://www.deeplearning.ai/the-batch/kimi-k2-thinking-outperforms-proprietary-models-with-new-techniques-for-agentic-tool-use); [Yicai](https://www.yicaiglobal.com/news/kimi-k2-thinkings-reported-usd46-million-training-cost-isnt-official-moonshot-ceo-says) (snippet)
- **Attention Residuals (AttnRes, arXiv 2603.15031, Mar 2026):** softmax attention over earlier layers' outputs replaces the fixed residual sum. Reported gain is **~25% training efficiency for <2% extra compute** (secondary). — (snippet)
- **Kimi K3 (announced July 16, 2026; full weights July 27) [S, README, vendor].**
  - Size: **2.8T total / 104B active**, 93 layers (**69 KDA + 24 gated MLA**), 896 experts with 16 selected plus 2 shared, "Stable LatentMoE". Native vision (401M MoonViT-V2), 1M context, **MXFP4 weights / MXFP8 activations QAT from SFT onward**.
  - The latent MoE is claimed to give "**~2.5× improvement in overall scaling efficiency over Kimi K2**".
  - Self-reported, at max effort, against Claude Fable 5 / GPT-5.6 Sol / Claude Opus 4.8:
    - GPQA-D 93.5 vs 92.6 / 94.1 / 91.0;
    - HLE-Full 43.5 (no tools) vs 53.3 / 44.5 / 49.8;
    - SWE-Marathon 42.0 vs 35.0 / 39.0 / 40.0;
    - FrontierSWE 81.2 vs 86.6 / 71.3 / 66.7;
    - BrowseComp 91.2 vs 88.0 / 90.4 / 84.3;
    - OSWorld-Verified 84.8 vs 85.0 / 83.0 / 83.4;
    - GDPval-AA v2 Elo 1686 vs 1747 / 1736 / 1593.
  - Harness differences apply (Kimi Code vs Codex/Claude Code). Claude Fable 5 "hit fallbacks on 35% of the tasks" on SWE-Marathon. Training compute and tokens are undisclosed. — [Kimi-K3 README](https://github.com/MoonshotAI/Kimi-K3); [Simon Willison, Jul 16, 2026](https://simonwillison.net/2026/Jul/16/kimi-k3/)

**Alibaba (Qwen)**
- **Qwen3.5 (Feb 16 – Mar 2, 2026):** Gated DeltaNet + sparse MoE hybrid, early-fusion multimodal, "RL scaled across million-agent environments". **Qwen3.5-397B-A17B decodes 8.6× / 19.0× faster than Qwen3-Max at 32K / 256K context, "with comparable performance"** (vendor). — [Qwen3.5/3.6/3.8 README](https://github.com/QwenLM/Qwen3.6) [S]; [Qwen on X](https://x.com/Alibaba_Qwen/status/2023331062433153103) (snippet)
- **Qwen3.6** (Apr 2026: 35B-A3B, 27B dense). **Qwen3.8** (Max announced Aug 3; **2.4T-A95B open weights Aug 12, 2026**; 27B on Aug 14): 512 routed experts with 10 active, **69 of 92 layers linear attention**, 262K context. — [README](https://github.com/QwenLM/Qwen3.6) [S]; [MindStudio](https://www.mindstudio.ai/blog/qwen3-8-2-4t-a95b-model-overview); [NVIDIA blog](https://developer.nvidia.com/blog/serve-qwen3-8-2-4t-a95b-a-2-4t-parameter-model-with-configurable-reasoning-on-nvidia-gb300-nvl72/) (snippet)

**MiniMax**
- **M1 (June 2025):** lightning (linear) attention hybrid, 1M context, and the **CISPO** RL algorithm, which clips importance-sampling weights. It uses **25% of DeepSeek-R1's FLOPs at 100K generation length**. **Full RL ran on 512 H800s in 3 weeks for $534,700** (vendor). — [MiniMax-M1 README](https://github.com/MiniMax-AI/MiniMax-M1) [S]; [arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
  - [E]: 512 × 504 h = 258k H800-h (≈ $2.07/h), which is at most 9.2e23 peak BF16 FLOP.
- **M2 (Oct 2025):** 230B total / **10B active**, ranked #1 among open models on the Artificial Analysis composite at release (vendor citing AA).
- **M2.5 (2026):** SWE-Bench Verified **80.2%**, BrowseComp 76.3%, RL in "hundreds of thousands" of environments (over 200k for coding), and "**$1 per hour at 100 tok/s**" (vendor). — [MiniMax-M2 README](https://github.com/MiniMax-AI/MiniMax-M2); [MiniMax-M2.5 README](https://github.com/MiniMax-AI/MiniMax-M2.5) [S]

**Zhipu (Z.ai)**
- **GLM-5 (Feb 11, 2026):** grew from 355B/32B to **744B/40B**, pretraining data from 23T to **28.5T tokens**. It **integrates DeepSeek's DSA** and adds the **slime** async RL infrastructure.
- **GLM-5.2:** "**IndexShare**" reuses one indexer across 4 sparse-attention layers, giving **2.9× fewer per-token FLOPs at 1M context**; MTP acceptance length improves by up to 20%.
- **GLM-5.3** uses the same base, and "every gain comes from post-training": +50% on the in-house Code Bench and emergent cyber capability.
- **GLM-5.3-Flash:** new base (320B-A18B), **sparse + linear hybrid attention + mHC**, and a 30T-token multimodal corpus (vendor). — [GLM-5 README](https://github.com/zai-org/GLM-5) [S]

**ModelBest / Tsinghua (MiniCPM)**
- MiniCPM4/4.1 (June/Sept 2025): trainable sparse attention (InfLLM-V2, which "can train a sparse attention model with only 5B long-text tokens"). About **7× decoding vs Qwen3-8B on Jetson AGX Orin**, and 3× in reasoning.
- **MiniCPM-SALA** (Feb 2026; sparse + linear attention, 9B): up to **3.5× Qwen3-8B speed at 256K**, and 1M context on one RTX 5090.
- **MiniCPM5-2B** (Sept 7, 2026): average score **53.9 vs ≤51.1** for the larger comparison models (Qwen3.5-4B, Gemma-4-E4B and others). Post-training is 400B SFT tokens plus domain RL teachers plus **on-policy distillation** (vendor). — [MiniCPM README](https://github.com/OpenBMB/MiniCPM) [S]
- The densing law (3.3-month capability-density doubling) is in the earlier notes.

**Originality vs adoption (my classification; origins cited)**
- **Chinese-originated, then diffused:**
  - MLA ([DeepSeek-V2, arXiv 2405.04434](https://arxiv.org/abs/2405.04434));
  - GRPO ([DeepSeekMath, arXiv 2402.03300](https://arxiv.org/abs/2402.03300));
  - NSA ([arXiv 2502.11089](https://arxiv.org/abs/2502.11089)) → DSA, adopted by Zhipu → IndexShare;
  - hyper-connections (ByteDance, [arXiv 2409.19606](https://arxiv.org/abs/2409.19606)) → mHC, adopted by Zhipu;
  - Engram;
  - MoBA ([arXiv 2502.13189](https://arxiv.org/abs/2502.13189)), reused by CAS SpikingBrain2.0 [S];
  - KDA; AttnRes; CISPO; Lightning Attention ([MiniMax-01, arXiv 2501.08313](https://arxiv.org/abs/2501.08313));
  - the first frontier-scale FP8 pretraining (V3).
- **Western-originated, adapted at scale:**
  - Muon (K. Jordan, Dec 2024; [blog](https://kellerjordan.github.io/posts/muon/)) → MuonClip (Moonshot), then **adopted by DeepSeek-V4**;
  - Gated DeltaNet ([NVIDIA/MIT, arXiv 2412.06464](https://arxiv.org/abs/2412.06464)) → Qwen3-Next/3.5/3.8 and KDA;
  - multi-token prediction ([Meta, arXiv 2404.19737](https://arxiv.org/abs/2404.19737)) → DeepSeek-V3 and GLM-5.2;
  - MoE and linear/state-space attention lineages generally.
- **Cross-lab diffusion inside China is fast**, within about 1–5 months. DSA reached Zhipu GLM-5; mHC reached GLM-5.3-Flash; Muon reached DeepSeek-V4; MoBA reached SpikingBrain2.0.

**Performance per training FLOP vs the US frontier**
- Epoch-derived values (via GitHub mirrors, earlier notes): DeepSeek-V3 **3.3e24**, Qwen3-235B-A22B 4.75e24, **Kimi K2.5 5.8e24**, **GLM-5 6.84e24**, versus GPT-4 2.1e25, GPT-4o about 3.8e25 (low confidence), Grok 3 3.5e26, GPT-4.5 3.8e26, Grok 4 **5.0e26**. GPT-5 is about 5e25 including RL. — [compute_ledger.md, Epoch mirror](https://github.com/soeunpark98/ai-carbon-visualization/blob/main/public/data/part3/3a.csv); [Epoch on GPT-5](https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont) (snippet)
- [E] 6ND for 2026 Chinese flagships:
  - DeepSeek-V4-Pro: 6 × 49e9 × 32–33e12 = **9.4–9.7e24**;
  - V4-Flash: **2.5e24**;
  - Kimi K3: 104B active × 20–40T [A] = **1.2–2.5e25**;
  - Qwen3.8: 95B active × 36T [A] = **~2e25**.
- An independent controlled study (nanoGPT scale) finds that **compute-independent innovations** (RoPE, FlashAttention, LayerNorm and similar) give a combined compute-equivalent gain of **up to 3.5×**. The largest historical gains (Transformer, MoE) are **compute-dependent**. The authors conclude that hardware restrictions are "insufficient to prevent all capability gains" from algorithms. — [Sanderson et al., arXiv 2505.04075](https://arxiv.org/abs/2505.04075v2) (snippet)

### Inferences
- **Stacked vendor multipliers from Chinese labs (2025–26):**
  - training: optimizer about 1.1–2× (independent evidence at the low end), mHC about 1.05×, AttnRes about 1.25×, Engram about 1.2×, LatentMoE "2.5× vs K2" (partly overlapping with MoE sparsity). Together roughly **3–6×** over a 2024 DeepSeek-V3-style baseline;
  - long-context inference: DSA/CSA/HCA/IndexShare/linear hybrids give **10–50×** reductions in FLOP and KV at 128K–1M.
  - This matches the earlier notes' view that inference multipliers far exceed training multipliers.
- Chinese flagships in mid-2026 use about **1e25 FLOP**, about **1–1.5 OOM below** US frontier runs (≥5e25–5e26). Chinese labs are *also scaling up* (1T params → 1.6–2.8T; 15T → 30T+ tokens) as compute arrives. Efficiency research is used to **stretch** scarce compute, not to replace scale.
- Genuinely original contributions cluster in **sparsity (attention, experts, memory) and training stability at scale** (mHC, MuonClip, AttnRes). These are plausibly shaped by memory-bandwidth and HBM scarcity (H800/H20 have reduced interconnect or FLOPs) and by serving cost pressure. None is a new learning paradigm.

### Gaps
- There are no official FLOP or GPU-hour disclosures for DeepSeek-V4, Kimi K3, Qwen3.5/3.8, GLM-5.x or MiniMax M2.x. Epoch estimates for K3/V4/Qwen3.8 were not accessible.
- There is no independent (non-vendor) ablation of DSA, mHC, AttnRes or LatentMoE at more than 100B scale.
- The exact DSA top-k and the V4 CSA/HCA mechanics were not read from the primary PDF (arXiv blocked).

---

## Q3. Did scarcity produce a different paradigm, or better engineering of the same one? RL, sparsity, distillation, and the gap over time

### Takeaway
The evidence points to **better engineering of the same paradigm**: decoder-only next-token pretraining on 15–36T tokens, MoE, then large-scale RL in agentic environments, plus heavy **distillation**, including (per Anthropic and OpenAI) unauthorized distillation from US frontier models at 10⁸-exchange scale. Scarcity shifted the *mix*. RL is at least 10% of pretraining cost at DeepSeek, sparsity is maximal, and there is aggressive quantization and open-weight release. It did not change the *kind* of method. The capability gap has stayed around **4–8 months** (Epoch: 7-month average since 2023, min 4, max 14). No Chinese model had held the ECI frontier as of early 2026.

### Cited Findings
**Gap over time**
- **Epoch (Jan 2026, L. Emberson):** since 2023, every ECI-frontier model has been American. Chinese models trailed by an **average of 7 months (min 4, max 14)**. The first Chinese model above GPT-4 came in May 2024, 14 months later. "**No Chinese model has yet surpassed the ECI of OpenAI's o3**" (Apr 2025). The gap "closely resembles" the gap between open-weight and closed models. — [Epoch data insight](https://epoch.ai/data-insights/us-vs-china-eci); [PolitiFact, Jan 29, 2026](https://www.politifact.com/factchecks/2026/jan/29/donald-trump/trump-says-the-us-leads-china-by-a-lot-in-ai-exper/) (snippet)
- Epoch: open-weight models trailed closed ones by about **4 months** (about 8 ECI points) over Jan–May 2026 (earlier notes). — [Epoch](https://epoch.ai/data-insights/open-closed-eci-gap)
- Mid-2026 (snippet aggregate):
  - Kimi K3's score "falls between US scores dated Feb 5 and Mar 5", i.e. **about 4.4–5.3 months**;
  - a fitted trend line gives a 6.08-month gap;
  - Nathan Lambert estimates **3–5 months**;
  - Chinese models trail by 1–16% on leaderboards, or 7–8 months. — [TrendForce "Kimi K3 and the 7% Gap"](https://insights.trendforce.com/p/kimi-k3-us-china-ai-gap); [Renovate QR](https://renovateqr.com/blog/chinese-ai-models-guide) (snippet)

**Distillation from US models (allegations; the labs dispute or have not confirmed)**
- **Anthropic (Feb 23, 2026) [S]:** about **24,000 fraudulent accounts** and over 16M exchanges:
  - **DeepSeek over 150k**, targeting reasoning, reward-model functionality and "censorship-safe alternatives";
  - **Moonshot over 3.4M**, targeting agentic reasoning, tool use, computer-use agents and vision;
  - **MiniMax over 13M**, targeting agentic coding and tool orchestration.
  - Attribution was by IP correlation, request metadata and infrastructure indicators. — [Anthropic](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)
- **Anthropic (Sept 10, 2026):** an **Alibaba-attributed campaign of 151M exchanges (May–July 2026)**, peaking at about 3M per day across about 3,500 accounts. A fixed prompt forced Claude Opus 4.6/4.7 to write out its full chain of thought. In total, **seven China-based labs, about 190M exchanges**. — [TechCrunch](https://techcrunch.com/2026/09/10/anthropic-details-distillation-campaigns-from-alibaba-moonshot-ai-and-deepseek/); [Better Stack summary](https://betterstack.com/community/guides/ai/anthropic-threat-report-2026/); [Quartz](https://qz.com/anthropic-chinese-ai-labs-distillation-alibaba-deepseek-moonshot-091126) (snippet)
- **OpenAI memo to the House China Select Committee (Feb 12, 2026):** DeepSeek engaged in "ongoing efforts to free-ride", including "**new, obfuscated methods**". China rejected US distillation claims naming six firms. — [Bloomberg](https://www.bloomberg.com/news/articles/2026-02-12/openai-accuses-deepseek-of-distilling-us-models-to-gain-an-edge); [Rest of World](https://restofworld.org/2026/openai-deepseek-distillation-dispute-us-china/); [tech-insider](https://tech-insider.org/china-rejects-us-ai-distillation-claims-2026/) (snippet)

**RL / post-training intensity**
- DeepSeek-V3.2's RL budget was over 10% of pretraining (secondary sources say typical open models spend about 1–2%). MiniMax M2.5 used RL in more than 200k real-world environments. Qwen3.5 used RL in "million-agent environments". Zhipu says GLM-5.3's gains come entirely from post-training. — sources as in Q2 [S/snippet]

**Market outcome**
- Chinese models passed US models in OpenRouter weekly tokens during **Feb 9–15, 2026** (4.12T vs 2.94T). Chinese workflows handle about 75% of engineering tasks "reasonably well" at about **1/5 the cost**. — [Fortune, Sept 13, 2026, via search](https://fortune.com/2026/09/13/china-us-ai-models-efficient/) (snippet)

### Inferences
- **Paradigm:** the same one. What scarcity changed:
  1. **Sparsity maximalism.** Active/total ratios reached about 3.7% (Kimi K3 104B/2.8T) and about 4% (Qwen3.8 95B/2.4T), and 3:1 or 75% linear-attention layers became standard.
  2. **Inference-cost engineering** (KV down to about 2% of baseline).
  3. **RL-heavy post-training.**
  4. **Importing capability via distillation.**

  Items 3 and 4 match the earlier ledger's finding that "cheap" capability usually imports upstream compute.
- **Size of the distillation channel [E/A].** 190M exchanges × 10³–10⁴ tokens [A] ≈ **2e11–2e12 tokens** of frontier-model outputs, including chain of thought. For comparison, R1's distillation set was about 4.3e9 tokens and MiniCPM5's SFT set 4e11 tokens. So borrowed frontier outputs are on the same scale as a full modern post-training corpus.
  - Teacher inference is roughly 2 × N_active(teacher) × tokens. That is about 1e23–1e25 FLOP if the teacher has 1e11–1e12 active params [A]; frontier teacher sizes are undisclosed.
  - Under the ledger's "full attribution" rule, part of US frontier pretraining (≥5e25–5e26) is implicitly borrowed.
  - The capability share this explains is **unknown**; attribution is contested and the labs deny wrongdoing.
- **Gap stability despite about 10× less compute** is itself the key datum; see Q7.

### Gaps
- No public, controlled measurement exists of how much Chinese model capability depends on distilled US-model data, as opposed to own RL or data.
- The Epoch ECI gap was not re-read after K3/V4/Qwen3.8 (July–Aug 2026); the mid-2026 gap estimates are aggregator snippets.
- The accused labs' official responses were not read in primary form.

---

## Q4. Chinese brain-inspired and neuromorphic programs: measured results

### Takeaway
China runs the world's largest neuromorphic hardware (Darwin Monkey: about 2B spiking neurons, 960 chips, about 2 kW) and the most visible "spiking LLMs" (CASIA SpikingBrain 1.0/2.0). Measured results are modest and mostly **inference-efficiency** results. SpikingBrain models are **converted from Qwen transformers**; 99%+ of their training compute is borrowed. Their spiking runs as "pseudo-spiking" on GPUs, and their energy gains are **theoretical MAC-energy estimates**. They recover about 90% of base-model quality at 7B. Darwin Monkey has demos (brain simulations, running a DeepSeek LLM) but no published capability or energy-per-task benchmarks that I could find. None of this shows a lower-compute path to *new* capability.

### Cited Findings
**SpikingBrain 1.0 ("瞬悉1.0", CASIA, Li Guoqi and Xu Bo; arXiv 2509.05276, Sept 2025; accepted by TMLR 2026) [S, report PDF]**
- The 7B (pure linear) and 76B-A12B (hybrid-linear MoE, 16 experts top-1 plus 1 shared, about 15% active) models are **both converted from Qwen2.5-7B-base**, with **about 150B tokens** of continual pretraining ("<2% of the data" of a from-scratch run of more than 10T).
- They were trained on "**hundreds of MetaX C550 GPUs**". MFU was 23.4% (1,558 tokens/s per GPU), versus 25.8% on A800.
- **SpikingBrain-7B MMLU 65.84 / CMMLU 71.58 vs Qwen2.5-7B-base 74.21 / 81.73.** It recovers "nearly 90%" of base performance, comparable to Mistral-7B and Llama3-8B. The 76B "nearly closes the gap" and is comparable to Llama2-70B, Mixtral-8×7B and Gemma2-27B.
- **TTFT speedup is over 100× at 4M tokens** (26.5× at 1M, measured on H100).
- Spiking sparsity is **69.15%**, with an average of 1.13 spikes per channel; 18.4% of channels never fire.
- Estimated MAC energy is **0.034 pJ vs FP16 and INT8 MACs, i.e. −97.7% (43.5×) and −85.2% (6.8×)**. This is based on "published hardware energy consumption data at 45nm", not measured on a chip.
- The released W8ASpike uses "**pseudo-spiking**… rather than true asynchronous event-driven spiking on neuromorphic hardware". — [SpikingBrain-7B repo and report](https://github.com/BICLab/SpikingBrain-7B); [CASIA press release](https://ia.cas.cn/kxyj/kydt_1/202509/t20250908_7963201.html) (snippet: "trained and served on a domestic thousand-card GPU platform")

**SpikingBrain 2.0 (Apr 2026) [S, README]**
- Two 5B models (SpB2.0-5B and VL-5B) are **converted from Qwen3-4B / Qwen3-VL-4B**, with only **14B tokens** of continued training.
- Dual-Space Sparse Attention combines **MoBA** (Moonshot) sparse softmax with **SSE** sparse linear attention on Gated DeltaNet.
- **TTFT is 10.13× faster at 4M context**, and quality "remains close to Qwen3-4B" (vendor). — [SpikingBrain2.0 repo](https://github.com/BICLab/SpikingBrain2.0)

**Darwin Monkey / "Wukong" (Zhejiang Univ. State Key Lab of Brain-Machine Intelligence + Zhejiang Lab, Pan Gang; unveiled Aug 2025)**
- Specs: **960 Darwin-3 chips**, more than **2B spiking neurons** and more than **100B synapses** ("close to a macaque"), about **2,000 W**, 15 blade servers, more than 2.35M neurons per chip, with on-chip online learning.
- Demos: simulated *C. elegans*, zebrafish, mouse and macaque brains, and **ran a DeepSeek LLM** for content, math and logic tasks.
- No quantitative task performance, energy-per-token or learning benchmarks appear in the coverage. — [ZJU](https://www.zju.edu.cn/english/2025/0910/c19573a3079424/page.htm); [Live Science](https://www.livescience.com/technology/computing/chinas-darwin-monkey-is-the-worlds-largest-brain-inspired-supercomputer); [Global Times](https://www.globaltimes.cn/page/202508/1339961.shtml) (snippet)

**Tsinghua Tianjic line**
- The hybrid ANN/SNN Tianjic chip was a Nature cover in **2019**; TianjicX appeared in Science Robotics. I found **no 2025–26 large-scale capability result**. — [Tsinghua](https://www.tsinghua.edu.cn/en/info/1244/3005.htm); [Shi Luping page](https://faculty.dpi.tsinghua.edu.cn/shiluping/en/article/12951/content/2928.htm) (snippet)

**China Brain Project**
- Budget about **RMB 5B (about $746M) in the 14th Five-Year Plan**; another snippet gives "over RMB 3B". It covers cognition, brain disease and brain-inspired computing.
- A primate brain-mapping collaboration launched in Nov 2025. — [Science](https://www.science.org/content/article/china-bets-big-brain-research-massive-cash-infusion-and-openness-monkey-studies); [CAS/Science Nov 2025](https://english.cas.cn/newsroom/cas_media/202511/t20251106_1096131.shtml) (snippet; the budget figures conflict)

**BAAI**
- Emu3's next-token-prediction-only multimodal model was published in **Nature (Jan 28, 2026)**. It matches task-specific models on perception and generation, and includes video generation and vision-language-action robotics. — [Nature](https://www.nature.com/articles/s41586-025-10041-x) (snippet)
- This is mainstream scaling, not brain-inspired computation. BAAI's world-model work is covered in Q5.

### Inferences
- **Compute accounting [E].**
  - SpikingBrain-7B's own compute: 6 × 7.6e9 × 1.5e11 ≈ **7e21 FLOP**, against Qwen2.5-7B's 18T-token pretraining of about **8.2e23**. Own share is **about 0.8%**; borrowed is **about 99%**.
  - SpikingBrain2.0-5B: own ≈ 3.4e20 against Qwen3-4B's 8.6e23. Own share is **about 0.04%**.
  - The "brain-inspired" label describes the *inference substrate* (linear/sparse attention plus spike-coded activations), not a brain-like *learning* process.
- **Energy.** Darwin Monkey runs about 2B neurons at about 2 kW, i.e. **about 1 kW per billion neurons**. It demonstrates scale, not capability. No evidence was found that it learns or performs tasks competitively with GPU-based models per joule or per FLOP.
- Net: Chinese neuromorphic and spiking work is **real and well-funded**, but so far it is **an inference-efficiency and hardware-sovereignty play** (MetaX, domestic chips). It rides on transformer pretraining and does not show a low-compute route to general capability.

### Gaps
- There are no measured (wall-plug) energy numbers for SpikingBrain on real neuromorphic hardware, and no Darwin-3 or Darwin Monkey task benchmarks.
- There is no independent evaluation of SpikingBrain 1.0/2.0 quality beyond vendor tables.
- The China Brain Project's 2026+ (15th Five-Year Plan) budget was not found.

---

## Q5. Chinese world-model and embodied-AI efforts and their compute

### Takeaway
Chinese world-model work in 2025–26 is large and fast-moving. The main efforts are BAAI Emu3.5 and Orca, AgiBot Genie Envisioner / GE-Sim 2.0 / GO-2, GigaAI GigaWorld-0, Tencent HunyuanWorld, and RoboBrain 2.0/2.5. It mostly **follows the Western scaling recipe**: unified next-token or next-state prediction on 10T+ multimodal tokens or 10⁵ hours of video, and world models used as **data engines and simulators** for robot policies. Compute is largely undisclosed. Nothing here is low-compute or brain-inspired in a way that departs from the mainstream.

### Cited Findings
- **Emu3.5 (BAAI, Oct 2025):**
  - trained with "unified next-token prediction" on **over 10T interleaved vision-language tokens** from video frames and transcripts, plus large-scale multimodal RL;
  - **Discrete Diffusion Adaptation gives about 20× faster inference "without performance loss"**;
  - "matches Gemini 2.5 Flash Image" on generation and editing (vendor). — [Emu3.5 README](https://github.com/baaivision/Emu3.5) [S]; [arXiv 2510.26583](https://arxiv.org/abs/2510.26583)
- **Orca (BAAI, arXiv 2606.30534, June 29, 2026):**
  - a "general world foundation model" using Next-State-Prediction in a unified latent space;
  - trained on **125K hours of video + 160M event annotations**, with "unconscious" dense video transitions plus "conscious" language-described events;
  - reads out to text, image prediction and embodied action;
  - **Orca-4B released July 14, 2026**. It "outperforms similar-sized specialized baselines" (vendor). — [arXiv 2606.30534](https://arxiv.org/abs/2606.30534); [project](https://orca-wm.github.io/) (snippet)
- **RoboBrain 2.0 (BAAI; 7B/32B)** is an "embodied brain" VLM for perception, reasoning and planning. **RoboBrain 2.5** adds 3D spatial reasoning and dense temporal value estimation ("Robo-Dopamine"). — [RoboBrain README](https://github.com/FlagOpen/RoboBrain2.0) [S]
- **AgiBot:**
  - Genie Envisioner (2025), described as an "action-driven world model";
  - **GE-Sim 2.0 (Apr 10, 2026)**, a "physical evolution engine" for closed-loop robot training in simulation;
  - **GO-2** foundation model (Apr 9, 2026) and the **AGIBOT WORLD 2026** dataset (Apr 7, 2026). — [AgiBot](https://www.agibot.com/article/231/detail/57.html); [The Robot Report](https://www.therobotreport.com/agibot-unveils-genie-envisioner-2-0-advance-world-models-scalable-simulators-embodied-ai/) (snippet)
- **GigaWorld-0** ("World Models as Data Engine to Empower Embodied AI", arXiv 2511.19861). **Pelican-Sim 1.0** (arXiv 2609.12036, a general world-model simulator for embodied intelligence). **WorldArena 2.0** (arXiv 2605.17912, an embodied world-model benchmark). **Tencent HunyuanWorld 1.0** (explorable 3D worlds from text or images). — (snippet titles and abstracts only)
- **Policy:** the embodied AI industry was written into the 2025 Government Work Report. It is expected to reach **RMB 1 trillion** in 2026 (industry forecast), and MIIT (Aug 26, 2026) listed embodied AI and BCI as 15th Five-Year Plan growth points. — [21jingji](https://www.21jingji.com/article/20260601/herald/bbb06bf429f7e42c2cbce14e652a3171.html); [QQ News](https://news.qq.com/rain/a/20260826A05OFT00) (snippet)

### Inferences
- Chinese world models look like **scaled multimodal LMs** (Emu3/3.5: 10T+ tokens; at a typical 30B-class size that would be about 1e24 FLOP [A]) plus **sim-to-real data engines**. They do not follow a DreamerV3-style, small-compute, online model-based RL route.
- The one lower-data signal is Orca's **125K hours** of video, about 0.1× V-JEPA 2's 1M+ hours. But it is a 4B model with only vendor evaluation.
- In the earlier ledger's terms, these efforts sit in the **~1e22–1e24 FLOP** tier (estimated; undisclosed) and inherit LLM backbones: RoboBrain on Qwen-VL-class models, Orca with text readouts. They supply **no evidence** that a world-model continual-learning agent is cheaper to build than an LLM agent.

### Gaps
- Parameter counts, token counts and training FLOP are undisclosed for Emu3.5, Orca, GO-2 and GE-Sim 2.0. The Emu3.5 parameter count could not be verified this session.
- There are no independent robot-success-rate benchmarks comparing Chinese and US embodied foundation models (e.g., vs π0/Gemini Robotics).

---

## Q6. Chinese government and strategy statements on "AGI via brain-inspired / low-compute / small-data" paths: claims vs evidence

### Takeaway
Official strategy is **pluralistic**. The 15th Five-Year Plan (Mar 2026) calls for "more efficient training and inference methods" and for **exploring AGI development paths** through multimodal models, agents, **embodied AI** and **swarm intelligence**, alongside general large models. A vocal scientific camp promotes alternatives to LLM scaling: BIGAI's Zhu Songchun ("small data, big task", value- and causality-driven), CAS brain-inspired work and Zhejiang neuromorphics. In practice, however, **capability leadership and compute flow to LLM labs**. The alternative programs publish **self-evaluated demos without standard benchmarks or compute disclosure**.

### Cited Findings
- **15th Five-Year Plan outline (Mar 2026):** "加快研究更加高效的模型训练和推理方法。鼓励多模态、智能体、具身智能、群体智能等技术创新，探索通用人工智能发展路径" ("accelerate research on more efficient model training and inference methods; encourage innovation in multimodal, agent, embodied and swarm intelligence; explore development paths for AGI"). It also calls for "模芯云用" (model-chip-cloud-application) co-innovation, the parallel development of general and industry models, and a model-capability evaluation system. BCI and embodied AI are named as future industries. — [CCDI full text](https://www.ccdi.gov.cn/toutiaon/202603/t20260313_479354.html); [NDRC PDF](https://www.ndrc.gov.cn/fggz/fzzlgh/gjfzgh/202603/U020260317369114704096.pdf); [Yicai: "AI mentioned 8 times"](https://www.yicai.com/news/102884423.html) (snippet; wording from search extract)
- **CSET (Hannas, Chang et al.):**
  - "Chinese Critiques of LLMs" (Jan 2025) documents leading Chinese scientists arguing that LLMs alone won't reach general AI, and China pursuing "brain-inspired" and hybrid BCI paths in parallel;
  - "Wuhan's AI Development" (May 2025);
  - "**China's Embodied AI: A Path to AGI**" (Dec 2025). — [CSET Jan 2025 PDF](https://cset.georgetown.edu/wp-content/uploads/CSET-Chinese-Critiques-of-Large-Language-Models-Finding-the-Path-to-General-Artificial-Intelligence.pdf); [CSET Dec 2025](https://cset.georgetown.edu/publication/chinas-embodied-ai-a-path-to-agi/) (snippet)
- **BIGAI / Zhu Songchun:**
  - "small data, big tasks" set against LLMs' "big data, small tasks";
  - TongTong 1.0 (2024) was described as a 3–4-year-old virtual child; **TongTong 2.0 was claimed at a "5–6-year-old child" level**;
  - **TongTong 3.0 (Mar 29, 2026, Zhongguancun Forum)** claims upgrades in spatial, cognitive and social intelligence (it enters an "AI Town" multi-agent world) and is "causality- and value-driven" rather than data-driven;
  - BIGAI evaluates its own agents with its own "**Tong Test**". — [Wikipedia: BIGAI](https://en.wikipedia.org/wiki/Beijing_Institute_for_General_Artificial_Intelligence); [NCSTI, Mar 30, 2026](https://www.ncsti.gov.cn/kjdt/kjrd/202603/t20260330_242361.html); [Sina](https://finance.sina.com.cn/jjxw/2026-03-29/doc-inhssmvu7039201.shtml) (snippet)
- CAS SpikingBrain's framing: "brain-inspired mechanisms to drive the next generation of efficient and scalable large model design". Its measured results are in Q4. — [SpikingBrain report](https://github.com/BICLab/SpikingBrain-7B) [S]

### Inferences
- **Claims vs evidence:**
  - TongTong's "child-level" claims rest on an in-house test, with no ARC-AGI, standard benchmark, compute or data disclosure found. There is no public evidence that "small data, big task" systems match LLM agents on any external benchmark.
  - Brain-inspired LLMs work as converted transformers (Q4).
  - Darwin Monkey has no capability metrics.
- **Revealed preference:**
  - The 15th Five-Year Plan's concrete levers are **efficiency plus embodiment plus agents**.
  - Chinese labs' scaled-up 2026 releases (1.6–2.8T MoE) show that when compute becomes available, it goes to **scaling the mainstream paradigm**.
  - The alternative-path rhetoric functions as **hedging and sovereignty policy** (domestic chips, BCI, embodied industry) more than as a demonstrated low-compute AGI route.

### Gaps
- There is no primary text of the 15th Five-Year Plan's AI chapter (gov.cn blocked); the wording is from search extracts.
- There are no independent evaluations of TongTong 2.0/3.0 and no compute, parameter or data figures for them.
- No quantitative national funding split between LLM, brain-inspired and embodied programs was found.

---

## Q7. Bottom line: elasticity of capability to compute, and implications for ranking (world-model continual-learning agent vs compact cognitive core vs self-improvement)

### Takeaway
The natural experiment shows that **catch-up capability is highly inelastic to compute**. About 1–1.5 OOM less training compute and about 5–10× less installed compute has cost Chinese labs only **about 4–8 months** of frontier time. **Frontier leadership remains compute-elastic**: zero Chinese ECI-frontier models since 2023, and Chinese labs scale up the moment they can. Scarcity drove **sparsity, memory lookup, long-context efficiency, RL-heavy post-training and distillation**, i.e. a better "compact/sparse core + borrowed teacher" recipe. It did not produce a new paradigm. Brain-inspired and world-model programs have **not** shown a cheaper route to general capability. For the ranking, this **supports "compact cognitive core" as the most compute-efficient demonstrated direction**, but only as a *follower* strategy that depends on upstream (often foreign) frontier compute. It **neither supports nor refutes** the world-model continual-learning agent. It shows **self-improvement via RL on a strong base as a practical multiplier** that is still bounded by teacher and base-model quality.

### Cited Findings
- 7-month average lag (4–14) since 2023, with no Chinese model surpassing o3 on ECI as of early 2026. — [Epoch](https://epoch.ai/data-insights/us-vs-china-eci) (snippet)
- Chinese flagship training compute is 3.3e24–6.8e24 (Epoch values for V3, K2.5, GLM-5), versus Grok 4 5e26, GPT-4.5 3.8e26 and GPT-5 about 5e25. — [Epoch mirror via compute_ledger.md](https://github.com/soeunpark98/ai-carbon-visualization/blob/main/public/data/part3/3a.csv); [Epoch GPT-5](https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont) (snippet)
- China's share of cluster performance is about 15% vs 75% for the US. Top Chinese training compute grew about 3×/yr vs about 5×/yr elsewhere. — [Epoch](https://epoch.ai/data-insights/ai-supercomputers-performance-share-by-country); [Epoch](https://epoch.ai/data-insights/china-compute-trends) (snippet)
- Compute-independent algorithmic gains are up to about 3.5× (nanoGPT); the big gains are compute-dependent. — [arXiv 2505.04075](https://arxiv.org/abs/2505.04075v2) (snippet)
- Distillation at 10⁷–10⁸ exchanges targeted reasoning traces, agentic coding and tool use. — [Anthropic](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks) [S]; [TechCrunch](https://techcrunch.com/2026/09/10/anthropic-details-distillation-campaigns-from-alibaba-moonshot-ai-and-deepseek/) (snippet)
- DeepSeek RL is over 10% of pretraining. M1's RL cost $0.53M, and R1's RL $0.29M (earlier notes). — sources in Q2/Q3

### Inferences
- **Elasticity estimate [E, rough].** Pairs of (Chinese model, closest US model):
  - DeepSeek-V3 (3.3e24, Dec 2024) vs GPT-4o (about 3.8e25, May 2024): **about 11× less compute, about 7 months later**;
  - Kimi K2.5 (5.8e24, Jan 2026) vs GPT-5 (about 5e25, Aug 2025): **about 9× less, about 5 months later**.
  
  So **~10× compute ≈ ~5–7 months** of frontier time in catch-up mode. Frontier training compute grows about 4–5×/yr, so 6 months is about 2–2.2×. Followers therefore extract about **4–5× more capability per FLOP** than leaders at the same date.
  
  This is consistent with the "catch-up algorithmic progress" of 16–60×/yr in the earlier notes, and it is inflated by (a) knowing what already works, (b) open literature, and (c) distillation from the leaders. Measured as ECI per log-FLOP, the curve is **shallow for followers and steep for leaders**.
- **What scarcity did not produce:** evidence that any Chinese group reached frontier capability with ≤1e23 FLOP, or that non-transformer, brain-inspired or small-data systems (SpikingBrain, Darwin Monkey, TongTong) match LLM agents. The cheapest "brain-inspired" LLMs are about 99% borrowed compute.
- **Direction ranking implications:**
  1. **Compact cognitive core** (sparse MoE + sparse/linear attention + external memory such as Engram + low precision + RL post-training + distillation). This is the best-supported low-compute route in the Chinese data: 10B-active MiniMax-M2 and 17B-active Qwen3.5 are near-frontier for agents, and inference multipliers are 10–50× at long context. **Caveat:** it depends on borrowed pretraining and teacher compute, so it lowers *marginal* rather than *first-discovery* cost.
  2. **Self-improvement.** RL and on-policy distillation are the largest post-2024 levers (DeepSeek RL >10%; GLM-5.3 "every gain from post-training"). But the heavy reliance on distilling *other labs'* models shows that self-generated improvement signals have not yet replaced a stronger external teacher.
  3. **World-model continual-learning agent.** Chinese efforts (Emu3.5, Orca, GE-Sim) are early, compute-hungry and benchmark-thin. The natural experiment offers **no evidence** for or against its compute efficiency.
- **Caveat on the experiment itself.** The "treatment" is only about 1 OOM and is leaky: smuggling is about 1/3 of stock, there are legal H800/H20s, and capability is imported via distillation. So it cannot test the "extremely low compute (≥3 OOM less)" regime. It *does* show that clever sparsity and RL buy about 1 OOM of catch-up efficiency, which is roughly the same as the frontier's total algorithmic progress over about 2 years.

### Gaps
- There are no official training FLOPs for 2026 Chinese flagships and no FLOPs for 2026 US frontier models, so the elasticity estimate rests on two 2024–26 pairs and assumptions.
- No study cleanly separates the capability contributions of distillation, own RL and architecture in Chinese models.
- No Chinese (or other) evidence was found of a sub-1e23-FLOP full-pipeline system with general, LLM-class ability.
