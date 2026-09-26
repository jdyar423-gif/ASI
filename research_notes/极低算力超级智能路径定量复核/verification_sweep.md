# Verification sweep: the most load-bearing unverified claims in the round-2 report (极低算力超级智能路径深化)

Scope: 20 claims that the round-2 conclusions lean on hardest, checked as of 26 Sep 2026. Access conditions: WebFetch is blocked for arxiv.org, arcprize.org, nature.com, huggingface.co, *.github.io, wikipedia, techmeme and thenextweb. Direct curl is blocked for the same hosts. The routes that worked were:
- (R1) paper full-text or LaTeX mirrors found with GitHub code search, then downloaded from raw.githubusercontent.com;
- (R2) official code repositories and their data files, including loading a weights file;
- (R3) the ARC Prize community-leaderboard metadata repo on GitHub;
- (R4) search-engine snippets, counted only when two or more independent results agree.

Every row names its route. "Search snippet" means text from a search-engine summary, not a page I read in full.

Status legend:
- **Confirmed**: primary text, or at least two independent sources, match the report.
- **Corrected**: the source gives a different number or different wording.
- **Refuted**: the source contradicts the report's claim.
- **Unverifiable**: no adequate source reached.

## Q0. Master verification table: which load-bearing claims survive?

### Takeaway
31 claims were checked:
- **20 confirmed as stated**, most now backed by primary or near-primary text;
- **1 split**: VibeThinker-3B's scores are confirmed, but its compute cannot be verified;
- **6 corrected**;
- **3 refuted**;
- **1 still unverifiable**: Astra's recurrent depth.

Five findings matter enough to change the report:
1. **Gundlach's "91%" is in the paper.** The report downgraded it, but it is real. It is a log share measured at the paper's own "2025 frontier" of about 2×10²³ FLOP.
2. **The ARC-AGI-3 "20–50× gap" mixes evaluation sets.** It sets public-set results against private-set results. Run on the 25 public games, the Kaggle winner's own base model, Qwen3.6-27B, scores 19.8% RHAE. On Kaggle's hidden games the same model scores 1.21%.
3. **Tycho is more expensive than OPINE-World, not cheaper.** Tycho's own artifacts show about $5.8K per Opus 4.8 run, against $1,040 for OPINE-World.
4. **Puro-2B is not free of borrowed compute.** Its pretraining mix contains datasets with LLM-generated or LLM-rewritten components.
5. **Cursor's +2.28% is an aggregate A/B gain, not a per-round gain.**

LeWorldModel's paper lists no AMI Labs affiliation.

### Cited Findings

| # | Claim | Round-2 report says | What the best source says | Status | Impact on conclusions | Route |
|---|---|---|---|---|---|---|
| 1 | Gundlach et al.: "91% from two scale-dependent innovations" | Downgraded to 「未核实」. Treated as a body-only snippet with a contradictory compute label; suggests the "91%" may be conflated with a "~91×" figure | The v1 full text states it twice. Contributions item 2: "LSTMs to Transformers, and Kaplan to Chinchilla re-balancing… Together, these account for 91% of total efficiency gains when extrapolating to the 2025 compute frontier". §4 gives the arithmetic: "Of the total measured progress of 21,400× (relative to an LSTM), … 846× … LSTMs to Kaplan Transformers, and nearly 10× … Chinchilla rebalancing. Together, these comprise 91%". The "2025 compute frontier (2 × 10²³ FLOPs)" label is the paper's own definition: a trend fit to Epoch *notable* models, not the largest run. A separate "~91×" (LSTM→Transformer in 2017) also exists. ([paper mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md); [arXiv 2511.21622](https://arxiv.org/abs/2511.21622)) | **Confirmed** (reverses the report's downgrade). "91%" is a log share: ln(8,460)/ln(21,400) = 0.907. LSTM→Transformer alone is 68% | The first-round wording can be restored with a caveat. The "frontier" here is about 2×10²³ FLOP, i.e. inside the report's tier 1–2 range (see Q1 inferences) | R1 |
| 2 | Gundlach abstract figures (<10× + <10× → <100×; 6,930× of 22,000×) | Verified from digest mirrors | Same abstract in the full-text mirror. §4: 6,930× at the 2023 frontier (1.3×10²² FLOP), of which 2,700× (89%) is scale-dependent; 2.23×/yr for their CEG multiplier vs Ho et al.'s 2.83×/yr ([mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md)) | **Confirmed** | "Small-scale <100× over 2012–2023" ⇒ <1.52×/yr stands | R1 |
| 3 | GPT-6 Astra on ARC-AGI-3: 62.7% vs 66%; ~100%; cost | 「冲突「未核实」」, "maybe different evaluation sets" | ARC Prize: **62.7% for ~$26K (Standard harness) and 99.9% for ~$19K (Provider Adapter harness), both on the Semi-Private set.** The Provider Adapter "preserves opaque reasoning state between requests and uses compaction". Chollet's post: "66% … standard harness, and nearly 100% with a continuous conversation harness and custom compaction, at … roughly $360 per game". ([ARC Prize blog](https://arcprize.org/blog/astra); [ARC Prize results](https://arcprize.org/results/openai-gpt-6-astra); [Techmeme](https://www.techmeme.com/260903/p40); [Chollet](https://x.com/fchollet/status/2095598451115614371); [AI/TLDR](https://ai-tldr.dev/releases/arcprize-astra-arc-agi-3/)) | **Confirmed** for 62.7% and 99.9%. **Corrected** on the explanation: the gap comes from the harness, not the evaluation set. **Unexplained**: Chollet's 66% vs the official 62.7% | Strengthens the red-team point that the frontier model has essentially saturated the Semi-Private set once the harness keeps reasoning state | R4 (≥4 independent results) |
| 4 | Astra uses "recurrent depth" | 「未核实」 | Still a single anonymously sourced report (The Information, 1 Sep). OpenAI's launch material names no looped or shared-layer architecture ([Raschka](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and); [atoms.dev](https://atoms.dev/blog/openai-astra-gpt-6-mewfour-release-date); [kingy.ai](https://kingy.ai/blog/recurrent-depth-openai-astra/)) | **Unverifiable** | Don't use it as evidence that looped depth has reached the frontier | R4 |
| 5 | Claude Opus 5.5 on ARC-AGI-1/2 (round 1) | 98.5% (v1), 93.3% (v2), $0.16 and $0.41 per task 「二手」 | ARC Prize (Verified, Semi-Private): ARC-AGI-2 93.3% at $0.41/task; ARC-AGI-1 98.5% at $0.16/task; 2.9 and 1.0 points above Opus 5 at about 80% lower cost ([ARC Prize X](https://x.com/arcprize/status/2102512140405866568); [ARC Prize results page](https://arcprize.org/results/anthropic-claude-opus-5-5)) | **Confirmed** | Can drop the 「二手」 tag | R4 (two primary-owned pages via snippets) |
| 6 | ARC-AGI-3 Kaggle Milestone #1 winner | Tufa Labs 1.21%, or "~1.6"; 「未核实」 | ARC Prize: 1st Tufa Labs **1.21%** ($25K); 2nd Reki 0.867% ($7.5K); 3rd Md Boktiar Mahbub Murad 0.864% ($5K). Tufa's "Duck harness" runs Qwen 3.6 27B FP8 locally as an agent that writes code in a REPL. Reki is a Gemma-4-31B vision-LLM policy. An earlier public-leaderboard jump went from 0.68% to 1.17%. No source gives "1.6" ([ARC Prize Milestone #1](https://arcprize.org/blog/arc-prize-2026-milestone-1); [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002); [digg](https://digg.com/ai/k8o9t7me); [duck-harness](https://github.com/Tufalabs/duck-harness)) | **Confirmed** at 1.21%. The "~1.6" is **unverified or stale** | The Kaggle top three sit at about 0.9–1.2%, not "1–3%" | R4 + R2 |
| 7 | Milestone #2 | Closes 30 Sep 2026; no results yet | Milestone #2 closes 30 Sep with prizes of $25K, $10K and $2.5K. Entry deadline 26 Oct, final submission 2 Nov, winners 4 Dec. The public Kaggle leaderboard is scored on about 50% of the test data ([ARC Prize competition page](https://arcprize.org/competitions/2026/arc-agi-3); search snippets) | **Confirmed**. No results yet | None yet | R4 |
| 8 | "Same method: 60–100 with a frontier API vs 1–3 on a single GPU, a 20–50× gap; the proposer prior does not fit on one GPU" | Stated as fact | The 58–100 figures are **public-set** scores; the Kaggle 1.21% is on **hidden** games. On the public 25 games, Polyphony with **Qwen3.6-27B**, the Kaggle winner's base model, scores **19.80% RHAE for $115**. Its reference setup serves the model at tensor-parallel 8. AERA reports 21.16% with Qwen2.5-0.5B and argues that "all 25 public ARC-AGI-3 games are reachable through non-intelligent strategies" ([Retrodict harness comparison](https://github.com/ryanbbrown/Retrodict/blob/main/docs/arc-agi-3-harness-comparison.md); [polyphony-arc-3](https://github.com/Mininglamp-AI/polyphony-arc-3); [AERA](https://github.com/farmountain/aera-arc3-paper)) | **Corrected.** The comparison mixes evaluation sets. The same model loses about 16× moving from public to hidden games. On the public set alone, frontier harnesses vs an open 27B are about 3–5× apart | Weakens "the proposer prior cannot fit in one GPU". Much of the gap is public-set familiarity or exploitability plus Kaggle's 12-hour, single-GPU budget. The fair comparison is Semi-Private frontier (Opus 5 30.2%, Astra 62.7%) vs Kaggle private (~1.2%): about 25–50×, but across different game pools | R3 + R2 + R1 |
| 9 | OPINE-World: 78.4, "self-reported total ~$800" | 「公开集自报」; costs conflict | 20/25 public games, **78.37% RHAE**, Claude Opus 4.8 (high). Submitted cost on the ARC Prize Community Leaderboard: **$1,040.00** ([submission.yaml](https://github.com/arcprize/ARC-AGI-Community-Leaderboard/blob/main/submissions/opine-world/submission.yaml); [author news post](https://github.com/chris91219/chris91219.github.io/blob/main/_news/2026-07-15-opine-world-arc3.md); [arXiv 2607.01531](https://arxiv.org/abs/2607.01531)) | Score **confirmed**. Cost **corrected** from ~$800 to $1,040 | Minor | R3 |
| 10 | Tycho: 88.49 (Opus 4.8), 100 (Opus 5, GPT-5.6 Sol); authors' estimate $12.4–15.2K; "about 2–2.6× cheaper than OPINE-World with Opus-class models" | 「公开集自报」 | README scorecards confirm 88.49, 100 and 100 on all 25 public games. Tycho's own `figure_data.json` gives mean realized cost per game at the $750 cap: Opus 5 **$95.46** (100 RHAE, ≈$2.4K per run); GPT-5.6 Sol **$179** (≈$4.5K); Opus 4.8 orchestrator **$231** (88.49 RHAE, ≈$5.8K). Retrodict lists $2,986 (API-equivalent) for the Opus 5 run ([Tycho](https://github.com/NIMI-research/Tycho); [artifacts](https://github.com/NIMI-research/Tycho/blob/main/artifacts/figure_data.json); [Retrodict](https://github.com/ryanbbrown/Retrodict/blob/main/docs/arc-agi-3-harness-comparison.md)) | Scores **confirmed**. Cost **corrected**. The "cheaper than OPINE-World" claim is **refuted**: with Opus 4.8, Tycho costs about $5.8K vs OPINE's $1.04K, roughly 5.5× more, for +10 RHAE | The "2–50× inference-side conversion rate" paragraph needs fixing. Cheaper near-perfect public-set runs exist: Retrodict 99.86% for $654; NOOA 85.13% for $332 under a two-hour cap | R2 + R3 |
| 11 | VibeThinker-3B: AIME26 94.3, GPQA 70.2, base Qwen2.5-Coder-3B; compute | 「自报」; teacher compute undisclosed | Abstract (arXiv 2606.16140): **94.3 on AIME26 (97.1 with claim-level test-time scaling)**, LiveCodeBench v6 Pass@1 80.2, IFEval 93.4, and the "Parametric Compression-Coverage Hypothesis". GPQA-Diamond 70.2 (VentureBeat and others). Base is Qwen2.5-Coder-3B. **No GPU-hours or total cost disclosed.** VibeThinker-1.5B disclosed about 3,900 H800 GPU-hours, under $8K ([abstract mirror](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/2026_papers/june_list.md); [VentureBeat](https://venturebeat.com/technology/why-weibos-tiny-vibethinker-3b-has-the-ai-world-arguing-over-benchmarks-again); [MarkTechPost](https://www.marktechpost.com/2026/06/19/vibethinker-3b-a-3b-dense-reasoning-model-built-on-qwen2-5-coder-3b-with-the-spectrum-to-signal-post-training-pipeline/); [VibeThinker-1.5B](https://arxiv.org/abs/2511.06221)) | Scores and base **confirmed** as author claims. Compute **still unverifiable** | No change. The full ledger for the #2 direction's showcase remains unknown | R1 (abstract) + R4 |
| 12 | Puro-2B: from scratch on RTX 5090s, 1.4T tokens, <$6.9K, "near Qwen2.5-1.5B", ≈1.7×10²² FLOP, **0% borrowed**, "the only truly clean ledger" | 「二手」 | Abstract confirms 1.4T tokens, FP8, RTX 5090s, best model under $6.9K, "approaches Qwen2.5-1.5B under our evaluation protocol". The title's "$5090" is the fitted cost law's estimate (~$4.4K) for Qwen2-1.5B parity. A secondary summary gives $6,891, 22,514 GPU-hours and 17.6 days, and excludes data acquisition, proxy runs and failed runs. The **data mix is Nemotron-CC, FineWeb-Edu-CN, MegaMath, SwallowMath-v2 and CoderForge Trajectories**, and a Qwen3-0.6B proxy was used for data audits ([abstract, zh-tw](https://github.com/shengwei-peng/awesome-ai-papers-zh-tw/blob/main/papers/2608.27370_Puro-2B_Poor_Lab's_Qwen2-1.5B_Trained_on_RTX_5090_within_$5090.md); [summary](https://github.com/ZhangCurosr/zhangcursor-papers-arxiv-cl-001/blob/main/2026-08-28/PuRo-2B-Poor-Lab-s-Qwen2-1-5B-Trained-on-RTX-5090-within-509_104b1911/summary.md); [arXiv 2608.27370](https://arxiv.org/abs/2608.27370)) | Capability and cost **confirmed**. 6ND gives 1.68×10²² FLOP, consistent with about 50% utilization of 22.5K RTX 5090 hours. "0% borrowed" is **refuted** under the report's own rule (see inferences) | The report's claim that "真正账本干净的只有Puro-2B" fails. No LLM-level system in the ledger is strictly borrow-free except DeepSeek-V3/R1 | R1 (abstract) + R4 |
| 13 | DeepSeek-R1 RL cost; DeepSeek-V3 GPU-hours | $294K, 147K H800 GPU-h; V3 2.788M | R1 v2 Table: R1-Zero 101K, SFT data 5K, R1 41K → **147K H800 GPU-h = $294K**, on 64×8 H800s. V3 README: **2.788M H800 GPU-h** (2.664M pretraining) ([R1 v2 text](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/deepseek-r1.txt); [DeepSeek-V3 README](https://github.com/deepseek-ai/DeepSeek-V3); [The Register](https://www.theregister.com/2025/09/19/deepseek_cost_train/)) | **Confirmed** | None | R1 + R2 |
| 14 | DiscoRL discovery compute and generalization | 1,024 TPUv3 cores × 64 h (Disco57); 2,048 × 60 h (Disco103) 「二手」; 754,778 params; Crafter human-level, near-MuZero Sokoban | Compute figures match Nature via multiple snippets. **Parameter count checked directly: `disco_103.npz` has 42 float32 arrays totalling 754,778 parameters.** Project-page source: evaluated on unseen Crafter, NetHack and Sokoban; "generalises when used to train agents with much more parameters and data"; Apache 2.0. Crafter human-level and near-MuZero Sokoban appear in Nature/36kr snippets ([Nature](https://www.nature.com/articles/s41586-025-09761-x); [disco_rl repo](https://github.com/google-deepmind/disco_rl); [36kr](https://eu.36kr.com/en/p/3527315416767366)) | **Confirmed** | Peak FLOP is ≈1.45×10²² (Disco57) and ≈2.7×10²² (Disco103), assuming ≈61.5 TFLOP/s per TPUv3 core. At 30–50% utilization, Disco103 is ≈0.8–1.4×10²². The report's "~10²²" stands | R2 (weights file) + R4 |
| 15 | WorldCoder "~50 interactions vs >1M steps" | Stated | Paper (via a structured reading note): the agent "builds a world model over its first 50 actions". Deep-RL baselines (PPO with 256 parallel envs; DreamerV3 defaults) need ">1M steps" for two-box Sokoban. GPT-4 is the synthesizer, capped at 50 requests per synthesis problem, front-loading ~400K tokens (~$15). No full compute accounting ([WorldCoder repo](https://github.com/haotang1995/WorldCoder); [reading note](https://github.com/beapirate/awesome-self-improving-agents/blob/main/wiki/sources/arxiv-2402.12275.md); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/820c61a0cd419163ccbd2c33b268816e-Paper-Conference.pdf)) | **Confirmed**, with the caveat that it is a symbolic, fully observed gridworld and the note says the numeric curves are incomplete | None | R1 (secondary structured) + R4 |
| 16 | PoE-World "<1,000 frames"; only positive Montezuma score | Stated | Demonstrations of "fewer than 1000 frames" per game; PoE-World + Planner is "the only method" with positive scores on Montezuma's Revenge (original and Alt). Its models run to 4,000+ lines of code vs WorldCoder's <100 ([MarkTechPost](https://www.marktechpost.com/2025/06/20/poe-world-outperforms-reinforcement-learning-rl-baselines-in-montezumas-revenge-with-minimal-demonstration-data/); [arXiv 2505.10819](https://arxiv.org/abs/2505.10819)) | **Confirmed** (two consistent secondary routes) | None | R4 |
| 17 | Dreamer 4: 2B params, 256–1,024 TPU v5p, 0.7% diamonds | Stated | Paper LaTeX: "2B parameters—400M for the tokenizer and 1.6B for the dynamics model—on 256 to 1024 TPU-v5p". Stone pickaxe >90%, iron pickaxe 29%, "obtains diamonds in 0.7% of episodes". Uses the 2,541 h VPT contractor dataset, "100× less data" than VPT ([Dreamer 4 text mirror](https://github.com/edwhu/dreamer4-jax/blob/main/docs/main.txt)) | **Confirmed**. Training duration not found, so the report's 4×10²¹–4×10²² FLOP stays an estimate | None | R1 |
| 18 | LeWorldModel: 15M params, single GPU, 48× faster planning; "AMI Labs' first output" | Attribution 「未核实」 | README abstract: "~15M parameters trainable on a single GPU in a few hours … plans up to 48× faster". Authors: Maes (Mila/UdeM), Le Lidec (NYU), Scieur (Mila/Samsung SAIL), LeCun (NYU), Balestriero (Brown); **no AMI Labs affiliation**. The later LeVJEPA (Aug 2026) does list "Advanced Machine Intelligence (AMI Labs)". Only an unofficial "ami.frenchtechjournal.com" page and blogs call LeWM an AMI output ([le-wm README mirror](https://github.com/aspamers/lewm); [affiliation note](https://github.com/Asada-Sinon/paper_reading/blob/main/2026-08/2026-08-15/leworldmodel.md); [LeVJEPA](https://github.com/MLO-lab/LeVJEPA/blob/main/paper.md)) | Specs **confirmed**. "AMI Labs output" is **refuted** at the affiliation level; it is a LeCun-circle academic paper | Removes LeWM as AMI evidence. AMI still shows no public language, math or reasoning system | R2 + R1 |
| 19 | AMI Labs funding | $1.03B seed | $1.03B seed announced 9–10 Mar 2026; $3.5B pre-money; Paris HQ; LeCun executive chairman, Alexandre LeBrun CEO ([TechCrunch](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/); [Silicon Republic](https://www.siliconrepublic.com/start-ups/yann-lecun-ai-start-up-ami-raises-seed-funding-world-model); [HPCwire](https://www.hpcwire.com/aiwire/2026/03/11/yann-lecuns-ami-secures-1b-seed-to-develop-ai-world-models/)) | **Confirmed** | None | R4 (≥3 outlets) |
| 20 | Continual-learning numbers: Dohare 89%→77%; plasticity-scaling study | Dohare peer-reviewed; the 5M–314M study 「二手」 | Dohare: "binary classification performance dropped from 89% … on an early task down to 77%, about the level of a linear network, on the 2000th task" ([arXiv v2](https://arxiv.org/html/2306.13812v2); [Nature](https://www.nature.com/articles/s41586-024-07711-7)). arXiv 2606.24752, "Can Scale Save Us From Plasticity Loss in Large Language Models?": plasticity loss appears from 5M to 314M non-embedding parameters and its onset "grow[s] sublinearly with model size" ([arXiv 2606.24752](https://arxiv.org/abs/2606.24752)) | **Confirmed** (both) | None | R4 |
| 21 | Oak Lab (Sutton and Javed, ~July 2026) | 「二手」 | Founded in July 2026 by Sutton and Khurram Javed, both ex-Keen, in Toronto. Goal: "an agent with one trillion parameters that learns and plans in real time while consuming 20 watts" ([TechTimes](https://www.techtimes.com/articles/320598/20260715/turing-award-winner-sutton-launches-oak-lab-calls-current-ai-fundamentally-broken.htm); [The Decoder](https://the-decoder.com/turing-award-winner-rich-sutton-founds-oak-lab-to-build-ai-agents-that-learn-on-their-own/); [BetaKit](https://betakit.com/ai-pioneer-richard-sutton-founds-new-research-lab/)) | **Confirmed** (three independent outlets) | None | R4 |
| 22 | Sparse memory finetuning: −11% vs −89% / −71% | 「二手」 | Abstract: "NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop" ([arXiv 2510.15103](https://arxiv.org/abs/2510.15103); [alphaXiv](https://www.alphaxiv.org/abs/2510.15103)) | **Confirmed** | None | R4 |
| 23 | Skill library: "−21 points at 202 skills; ~68% from wrong skill choice; inflection at ~100+ skills" | 「二手」 | Abstract: "performance degrades as libraries grow — by up to 21% when scaling from a small set of helpful skills to a 202-skill library". Skill-selection failure (shadowing), not context overhead, is the main cause; the context effect is "indistinguishable from zero" ([abstract mirror](https://github.com/JaredYe04/news-bot/blob/main/daily/2026-05-26-morning.md); [arXiv 2605.24050](https://arxiv.org/abs/2605.24050)) | **Partly corrected.** "Up to 21%" is confirmed, but "points" vs relative percent is ambiguous. The 68% share was not found. No "inflection at ~100+ skills" is stated: 202 is simply the largest library tested | The paper studies retrieval over human-authored skill libraries, not self-accumulated skills. It is indirect evidence against compounding and should not be cited as a measured inflection point | R1 (abstract) |
| 24 | Cursor real-time RL: "each ~5-hour round raises edit persistence by 2.28% and cuts dissatisfied follow-ups by 3.13%" | 「自报」 | Blog: each cycle takes "about five hours"; checkpoints are gated on evals including CursorBench. "We were able to improve Composer 1.5 via A/B testing behind Auto": +2.28% edits persisting, −3.13% dissatisfied follow-ups, −10.3% latency ([Cursor](https://cursor.com/blog/real-time-rl-for-composer); [blog mirror](https://github.com/kzinmr/ai-topics/blob/main/wiki/raw/articles/2026-06-05_cursor_real-time-rl-for-composer.md)) | **Corrected.** These are aggregate A/B gains for Composer 1.5, not per round; the number of cycles is not stated | Weakens "each round compounds" if anyone reads it that way. The loop still exists and is eval-gated | R1 |
| 25 | Budget-matched re-evaluation (arXiv 2606.15017) | 「二手」 | "Across three WebArena domains and three models (Gemini 3 Flash, GPT-5.4-mini, Qwen 3.6-27B), the vanilla baseline matches or surpasses all three augmentation methods [AWM, ASI, ReasoningBank] … while often using fewer total tokens" ([arXiv 2606.15017](https://arxiv.org/abs/2606.15017)) | **Confirmed** | None | R4 |
| 26 | BabyLM EWoK: best 58.4% in 2024; 2025 baselines 49.5–52.4; 2025 winners | Cited | 2024 (100M track): EWoK baseline 51.9, best **58.4**; 10M track: baseline 50.7, best 54.6; "most submissions near chance". 2025 README baselines: **49.47–52.44**. 2025 winners: Strict/NLP **Simple Diffusion** (NLP score 58.4 vs GPT-BERT-masked baseline 63.0); Strict/human-likeness CLASS-IT; Strict-Small/NLP AMLM-Hard-Decay; Strict-Small/human-likeness MoEP; Interaction BLM ([2024 structured note 1](https://github.com/borgr/paper-geo/blob/main/data/sidecars/findings-of-the-second-babylm-challenge-sample-efficient-pre.md); [2024 structured note 2](https://github.com/jeswr/kernel-of-truth/blob/main/docs/next/lit/EVAL.sources.jsonl); [2025 README](https://github.com/babylm/evaluation-pipeline-2025); [2025 Findings text](https://github.com/juand-r/claude-sandbox/blob/main/explorations/babylm2025/papers/p28.txt)) | **Confirmed.** The two "58.4"s are different quantities, both real | None | R1 + R2 |
| 27 | METR Jul 2026: "Anthropic contributors merge 8× code; uplift very likely >2×" | 「二手」, phrased as "METR measured" | METR note by Thomas Kwa: Anthropic *reported* 8× merged code per day vs pre-AI. Under Cobb-Douglas or CES assumptions this *implies* 2.33–2.91× researcher uplift from coding agents. Caveats: AI code may be more verbose, and low-stakes code may not have been written before AI ([METR](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/); [GreaterWrong mirror](https://www.greaterwrong.com/posts/ix5qEyW9BjGEb4d8k/because-8-e-anthropic-s-researcher-uplift-is-plausibly)) | **Corrected in wording**: model-based inference from a lab-reported figure, not a METR measurement. Numbers confirmed | ">2× uplift" is softer than "measured" | R4 (two mirrors) |
| 28 | AI Futures Project Q1 2026 update | Automated Coder median late 2029 → mid 2028 | Confirmed for Daniel. Eli's median moved from early 2032 to mid 2030; TED-AI moved about 1.5 years earlier. A later "Q2.5 2026 Timelines Update: Uplift and Revenue" exists ([AIFP Q1](https://blog.aifutures.org/p/q1-2026-timelines-update); [AIFP Q2.5](https://blog.aifutures.org/p/q25-2026-timelines-update-uplift)) | **Confirmed**. The newer Q2.5 medians were not retrieved | The report should cite both forecasters, or the newer Q2.5 update | R4 |
| 29 | Forethought: r≈1.2 (0.4–3.6); ~60% chance of compressing >3 years; ~20% chance of >10 years | Cited (round 1) | r = 1.2, range 0.4–3.6; ~60% chance of compressing >3 years into <1 year; ~20% for >10 years ([Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be); [EA Forum](https://forum.effectivealtruism.org/posts/eZ2hmSx2hYw6nfcwv/how-quick-and-big-would-a-software-intelligence-explosion-be)) | **Confirmed**. The "~50% self-sustaining" figure was not re-found | None | R4 |
| 30 | Cunningham et al. (Sep 2026): the loop is "not yet self-sustaining but strengthening" | 「二手」 | arXiv 2609.15802, submitted 14 Sep 2026, nine authors including Trammell and Halperin: "feedback loops are not currently strong enough to generate a self-sustaining acceleration, though they appear to be strengthening"; this is a back-of-the-envelope estimate ([arXiv](https://arxiv.org/abs/2609.15802); [METR note](https://metr.org/notes/2026-07-22-economics-of-recursive-self-improvement/)) | **Confirmed** | None | R4 |
| 31 | Chollet: Astra develops "its own shorthand DSL"; "harness capabilities are increasingly shifting into the model itself" | 「二手」 | Same wording across The Decoder, Zvi's roundup and Chollet's X thread ([The Decoder](https://the-decoder.com/benchmarks-disagree-on-gpt-6-astra-but-its-human-beating-efficiency-on-arc-agi-3-pulls-chollets-agi-forecast-forward/); [Zvi](https://thezvi.substack.com/p/gpt-6-astra-can-do-ambitious-things)) | **Confirmed** | The red team's central "function moves into big models" point holds | R4 |

### Inferences
- **No ranking-level conclusion is overturned.** Three supporting arguments need rewording:
  - ARC-AGI-3 "single-GPU 1–3% ⇒ the proposer prior doesn't fit on a GPU" (#8);
  - "the inference-side conversion rate is only 2–50×" (#10);
  - "Puro-2B is the only clean ledger" (#12).
- **Two corrections cut in opposite directions.**
  - The Gundlach restoration (#1) supports the round-1 claim that most measured algorithmic progress is scale-dependent, which hurts the ≤10²¹ tier. But the paper's own frontier is ~2×10²³ FLOP. Scale-dependent gains already realized at that scale are available to a 10²³–10²⁴ system, so the pessimism applies mainly to tier 3, not tiers 1–2.
  - The ARC-AGI-3 correction (#8) shrinks the apparent "model-size" gap between frontier API and open models to 3–5× on the public set. Much of the Kaggle collapse comes from hidden games and the 12-hour single-GPU budget. That modestly *helps* the #1 direction's world-model-harness story.
- **The report's full-attribution rule has a second-order problem.** Almost every modern open pretraining corpus is filtered or rewritten by LLMs, so almost nothing is borrow-free. The rule needs a de-minimis threshold, or an explicit statement that "borrowed" can be a small, unquantified fraction.

### Gaps
- Chollet's 66% vs ARC Prize's 62.7% for Astra's Standard harness is unexplained; a pre-final run is possible.
- The number of games in the Semi-Private set was not confirmed. $19K ÷ ~$360 ≈ 53 games, consistent with the 55-game private pool AERA mentions, but this is an inference.
- VibeThinker-3B compute and teacher models are undisclosed.
- Dreamer 4 training duration was not found.
- Forethought's "~50% self-sustaining" figure was not re-verified.
- The skill-library paper does not settle whether its 21% means percentage points or a relative change.

## Q1. Gundlach et al. (Nov 2025): what exactly does the paper claim about scale-dependence?

### Takeaway
The "91%" figure is genuine body text (intro and §4). It is a **log-share** at the paper's "2025 compute frontier" of about 2×10²³ FLOP, defined as a trend over Epoch notable models. The abstract's verified numbers stand: under 100× from small-scale ablations plus literature, and 6,930× of Ho et al.'s 22,000× reconstructed. The round-2 downgrade to 「未核实」 and its "contradictory label" diagnosis should be reversed.

### Cited Findings
- Abstract (v1): "…account for less than 10x … additional innovations … less than 10x, yielding a total under 100x … we account for 6,930x efficiency gains over the same time period, with the scale-dependent LSTM-to-Transformer transition accounting for the majority of gains. Our results indicate that algorithmic progress for small models has been far slower than previously assumed" — [v1 full-text mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md)
- Contributions list: scale-invariant innovations give "less than 10× … less than 10% of total improvements extrapolated to the 2025 compute frontier (2 × 10²³ FLOPs)". The two scale-dependent innovations "account for 91% of total efficiency gains when extrapolating to the 2025 compute frontier" — [same mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md)
- §4 decomposition: "between 2017 and 2025 … Of the total measured progress of 21,400× (relative to an LSTM), … 846× is achieved through LSTMs to Kaplan Transformers, and nearly 10× is attributable to Chinchilla rebalancing. Together, these comprise 91% of total relative efficiency gains". The paper also states that "the results remain log-proportional to the independent multipliers" — [same mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md)
- The "compute frontier" is defined via "the exponential relationship between time and compute among notable models [Epoch AI, 2023]". The 2023 frontier is 1.3×10²² FLOP; there, 725× comes from LSTM→Kaplan Transformer, 3.7× from Chinchilla and 2.6× from scale-invariant changes, for about 6,930× in total, "of which 2,700× (89%) is due to scale-dependent changes". The CEG multiplier grows about 2.23×/yr vs Ho et al.'s 2.83×/yr — [same mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md)
- The LSTM→Transformer gain grows with scale: "only 6× gains at 10¹⁵ FLOPs but more than 100× gains at 10²³ FLOPs"; "around 91×" at 2017 scale. Also: "limits to compute scaling pose obstacles not only to realizing efficiency gains but also to discovering them". §5 says LSTM→Transformer is "68% of measured efficiency gains at frontier scales" — [same mirror](https://github.com/bingran-you/bingran-you/blob/d6e0c6a2cc60c604e238b30c750c9b2017d186e6/papers/papers-in-zotero/large%20files/random%20papers/2511.21622v1.md)
- An Epoch newsletter paraphrases this as "most efficiency gains came from two scale-dependent innovations" — [Epoch substack mirror](https://github.com/shakir-fattani/ai-updates/blob/main/epochai.substack.com/the-least-understood-driver-of-ai/content.md)

### Inferences
- The shares are log shares, which I recomputed: ln(846×10)/ln(21,400) = 0.907 and ln(846)/ln(21,400) = 0.676, matching 91% and 68%. The 91% therefore does not mean that 91% of a multiplicative factor is explained; it is a share of orders of magnitude.
- The round-2 report called the "2×10²³ FLOPs" label contradictory because the real 2025 frontier is about 5×10²⁶. It is not a contradiction: it is the paper's definition, a trend line over notable models. But it means the paper's "frontier-scale" gains are measured at **about 10²³ FLOP, close to the report's tier 2 and below its 10²⁴ brain anchor**. A low-compute learner at 10²³–10²⁴ FLOP would already capture most of the historically measured scale-dependent gains. The "small models progress slowly" finding bites hardest at ≤10²¹, tier 3.
- The report's "if small-scale progress is <1.5×/yr, compressing 10²⁷→10²⁴ takes 16+ years" applies a rate for scale-invariant gains measured at ~10¹⁵–10¹⁸ FLOP to a 10²⁴ target. That is a questionable transfer and should be marked as a lower-bound scenario, not a central one.

### Gaps
- Whether a v2 revised any numbers could not be checked; the mirror is v1.

## Q2. ARC-AGI family: Astra, Opus 5.5, Kaggle milestones, OPINE-World, Tycho and the "20–50×" gap

### Takeaway
- **Astra.** The official figures are 62.7% (Standard harness, ~$26K) and 99.9% (Provider Adapter, ~$19K), both on the Semi-Private set. The 62.7-vs-~100 "conflict" is a harness difference, not a dataset difference. Chollet's own "66%" remains unexplained.
- **Opus 5.5.** 98.5% and 93.3% are ARC Prize–verified.
- **Kaggle.** Milestone #1 top three: 1.21%, 0.867%, 0.864%. Milestone #2 closes 30 Sep, with no results yet.
- **Public vs hidden games.** The report compares frontier harnesses on public games (58–100) with Kaggle hidden games (≈1). The same open model, Qwen3.6-27B, gets 19.8% on the public games, so the set difference alone is ~16×.

### Cited Findings
- Astra, Standard harness: 62.7% for ~$26K on Semi-Private. Provider Adapter harness: 99.9% for ~$19K on Semi-Private; it "preserves opaque reasoning state between requests and uses compaction". Across 167 game-reasoning pairs, Provider Adapter runs were about 3.66× faster and used 49% fewer tokens — [ARC Prize blog](https://arcprize.org/blog/astra); [AI/TLDR](https://ai-tldr.dev/releases/arcprize-astra-arc-agi-3/); [Techmeme](https://www.techmeme.com/260903/p40)
- Astra "used fewer actions than the median tested human on 96% of levels" — [ARC Prize blog via snippet](https://arcprize.org/blog/astra)
- Chollet: "scores 66% on ARC-AGI-3 using our standard harness, and nearly 100% with a continuous conversation harness and custom compaction, at a cost of roughly $360 per game" — [Chollet on X](https://x.com/fchollet/status/2095598451115614371). A search summary notes that the official best Standard score is 62.7%.
- OpenAI's 3 Sep launch post reportedly says Astra "saturates" ARC-AGI-3 at 99.9% — search snippet; primary not reached.
- Opus 5.5 (Verified): ARC-AGI-2 93.3% at $0.41/task; ARC-AGI-1 98.5% at $0.16/task — [ARC Prize X](https://x.com/arcprize/status/2102512140405866568); [ARC Prize results](https://arcprize.org/results/anthropic-claude-opus-5-5)
- Semi-Private leaderboard as of 24 Sep (aggregator): Astra 62.7%, Opus 5 30.2%, Gemini 3.8 Flash 10.4% — [BenchLM](https://benchlm.ai/benchmarks/arcagi3)
- Milestone #1 (Kaggle compute-limited track): Tufa Labs 1.21% ($25K); Reki 0.867% ($7.5K); Md Boktiar Mahbub Murad 0.864% ($5K). Tufa's Duck harness runs Qwen 3.6 27B FP8 locally and turns the game state into Python variables in a REPL; "hand-crafted tools actually hurt". Reki uses Gemma-4-31B as a vision-LLM policy — [ARC Prize Milestone #1](https://arcprize.org/blog/arc-prize-2026-milestone-1); [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002); [duck-harness repo](https://github.com/Tufalabs/duck-harness)
- An earlier Kaggle public-leaderboard step went from 0.68% to 1.17% — [digg](https://digg.com/ai/k8o9t7me)
- Milestone #2: closes 30 Sep 2026 with $25K/$10K/$2.5K; entry deadline 26 Oct, final 2 Nov, winners 4 Dec. The Kaggle leaderboard is scored on about 50% of the test data — [ARC Prize competition page](https://arcprize.org/competitions/2026/arc-agi-3) (snippet)
- Public-set harness comparison (25 games; costs per run):

  | Harness | Model | RHAE | Cost |
  |---|---|---:|---:|
  | Tycho | Opus 5 | 100% | $2,986 (API-equivalent) |
  | Retrodict | GPT-5.6 Sol | 99.86% | $654 |
  | NOOA | GPT-5.6 Sol | 85.13% | $332, two-hour cap |
  | OPINE-World | Opus 4.8 | 78.37% | $1,040 |
  | Polyphony | Qwen3.6 | 19.80% | $115 |
  | Continual Harness | Gemini 3.1 Pro | 20.54% | $774 |
  | AERA | Qwen2.5-0.5B | 21.16% | paper result; "evaluator behavior limits comparison" |

  — [Retrodict comparison](https://github.com/ryanbbrown/Retrodict/blob/main/docs/arc-agi-3-harness-comparison.md)
- Polyphony's default model is `Qwen/Qwen3.6-27B`, served with vLLM at tensor-parallel 8 in the README — [polyphony-arc-3](https://github.com/Mininglamp-AI/polyphony-arc-3)
- AERA claims "all 25 public ARC-AGI-3 games are reachable through non-intelligent strategies … the private 55-game set is the only genuine intelligence test". This is a single-author, unreviewed claim — [AERA repo](https://github.com/farmountain/aera-arc3-paper)
- OPINE-World submission: Opus 4.8 (high), public set, cost 1040.00, University of Toronto (Courtis, Li, Sanner) — [submission.yaml](https://github.com/arcprize/ARC-AGI-Community-Leaderboard/blob/main/submissions/opine-world/submission.yaml)
- Tycho scorecards on 25 public games:

  | Policy | Model | RHAE |
  |---|---|---:|
  | No world model | Opus 4.8 | 79.07 |
  | Single actor | Opus 4.8 | 85.36 |
  | Actor-controlled builder | Opus 4.8 | 88.49 |
  | Falsification-triggered builder | Opus 4.8 | 83.07 |
  | Actor-controlled builder | GPT-5.6 Sol | 100.00 |
  | Actor-controlled builder | Opus 5 | 100.00 |

  — [Tycho README](https://github.com/NIMI-research/Tycho)
- Tycho budget sensitivity, mean realized cost per game at a $750 cap: Opus 5 $95.46 (100 RHAE); GPT-5.6 Sol $179.00 (100); Orchestrator $231.23 (88.49); Single $290.70 (85.36); No-world-model $226.33 (79.07). At a $100 cap, Opus 5 already reaches 85.09 RHAE for $70.56/game — [Tycho figure_data.json](https://github.com/NIMI-research/Tycho/blob/main/artifacts/figure_data.json)
- Chollet: Astra does "on-the-fly symbolic world modeling … developing its own shorthand DSL"; "harness capabilities are increasingly shifting into the model itself" — [The Decoder](https://the-decoder.com/benchmarks-disagree-on-gpt-6-astra-but-its-human-beating-efficiency-on-arc-agi-3-pulls-chollets-agi-forecast-forward/); [Zvi](https://thezvi.substack.com/p/gpt-6-astra-can-do-ambitious-things)

### Inferences
- **Same model on two game pools.** Qwen3.6-27B scores 19.80% on public games with Polyphony and 1.21% on Kaggle hidden games with Tufa, a ~16× drop. The hardware and harness also differ (8-GPU serving vs one GPU for 12 hours).
- **A cleaner decomposition.** On the public set, frontier harnesses score 78–100 vs 20 for an open 27B model, a 4–5× model or harness gap. From public to hidden games, the drop is ≥10× for any model. Semi-Private frontier numbers (Opus 5 at 30.2%, Astra at 62.7%) also fall well below their public-set 100s.
- **"The proposer prior does not fit in one GPU" is only partly supported.** It needs to be separated from public-set overfitting or exploitability and from the throughput cap.
- **Tycho vs OPINE-World.** With Opus 4.8, Tycho's 88.49 run costs about 25 × $231 ≈ $5.8K against OPINE's $1,040, so Tycho is about 5.5× more expensive, not 2–2.6× cheaper. With Opus 5, Tycho's ≈$2.4K run is still about 2.3× dearer than OPINE-World, but it scores 100 vs 78.
- **Public-set play is cheap.** The cheapest near-perfect run is Retrodict at $654 per 25 games, about $26/game. The report's "$120–600 per game" is too high as a floor.
- **Astra cost per game.** $19K / $360 ≈ 53 games, which suggests Chollet's "$360 per game" refers to the 99.9% Provider Adapter run. This is an inference.

### Gaps
- The Semi-Private set size was not confirmed.
- The source of Chollet's 66% was not found.
- Tycho's "$12.4–15.2K" figure did not appear in the repo artifacts; it may be a whole-project total.
- Polyphony's hardware for the scored run (TP=8 vs a single GPU) is not stated in the comparison table.

## Q3. Compact kernels and the full ledger: VibeThinker-3B, Puro-2B, DeepSeek R1/V3

### Takeaway
- **VibeThinker-3B.** The headline scores are confirmed as author claims on a Qwen2.5-Coder-3B base, but the report discloses no compute.
- **Puro-2B.** Scale and cost are confirmed, and the 1.7×10²² FLOP figure is consistent. Its data mix, however, includes LLM-generated or LLM-rewritten corpora, so "0% borrowed" fails under the report's own rule.
- **DeepSeek.** The R1 and V3 compute figures are confirmed from primary text.

### Cited Findings
- VibeThinker-3B abstract: "94.3 on AIME26 (improving to 97.1 with claim-level test-time scaling), an 80.2 Pass@1 on LiveCodeBench v6 … 96.1% acceptance rate on recent unseen LeetCode contests … 93.4 on IFEval … Parametric Compression-Coverage Hypothesis" — [abstract via GitHub list](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/2026_papers/june_list.md); [arXiv 2606.16140](https://arxiv.org/abs/2606.16140)
- GPQA-Diamond is 70.2, vs 91.9 for Gemini 3 Pro and 87.0 for Claude Opus 4.5. Other reported scores: AIME25 91.4, HMMT25 89.3 — [VentureBeat](https://venturebeat.com/technology/why-weibos-tiny-vibethinker-3b-has-the-ai-world-arguing-over-benchmarks-again); [AI Weekly](https://aiweekly.co/alerts/vibethinker-3b-scores-943-on-aime26-matching-671b-deepseek-v32)
- The base model is Qwen2.5-Coder-3B. The pipeline includes a "data synthesis" stage — [MarkTechPost](https://www.marktechpost.com/2026/06/19/vibethinker-3b-a-3b-dense-reasoning-model-built-on-qwen2-5-coder-3b-with-the-spectrum-to-signal-post-training-pipeline/); [Raschka notes](https://sebastianraschka.com/blog/2026/vibethinker-3b-post-training.html) (snippet)
- The VibeThinker-3B report "does not disclose total GPU hours or a complete cost breakdown" (search summary). VibeThinker-1.5B used about 3,900 H800 GPU-hours, under $8K — [VibeThinker-1.5B](https://arxiv.org/abs/2511.06221)
- Puro-2B abstract: models trained from scratch "on up to 1.4 trillion tokens with FP8 precision on consumer-grade RTX 5090 GPUs". The best model costs "less than $6.9K" and "approaches Qwen2.5-1.5B … under our evaluation protocol". The fitted cost law puts Qwen2-1.5B parity at about $4.4K. Released under Apache 2.0 — [abstract (zh-tw)](https://github.com/shengwei-peng/awesome-ai-papers-zh-tw/blob/main/papers/2608.27370_Puro-2B_Poor_Lab's_Qwen2-1.5B_Trained_on_RTX_5090_within_$5090.md); [arXiv 2608.27370](https://arxiv.org/abs/2608.27370)
- The Puro-2B cost covers only one rerun of the final recipe; "data acquisition, proxy experiments, failed runs, and research labor [are] excluded" — search summary of [arXiv 2608.27370](https://arxiv.org/abs/2608.27370)
- Puro-2B, from an LLM-generated summary (low reliability):
  - $6,891, 22,514 GPU-hours, 17.6 days;
  - 15-benchmark mean 57.81 vs Qwen2-1.5B 55.14;
  - data: Nemotron-CC, FineWeb-Edu-CN, MegaMath, SwallowMath-v2, CoderForge Trajectories (438.8B + 960B tokens);
  - a Qwen3-0.6B proxy was used for data audits.

  — [summary](https://github.com/ZhangCurosr/zhangcursor-papers-arxiv-cl-001/blob/main/2026-08-28/PuRo-2B-Poor-Lab-s-Qwen2-1-5B-Trained-on-RTX-5090-within-509_104b1911/summary.md)
- DeepSeek-R1 v2 Table 7: R1-Zero 101K + SFT data 5K + R1 41K = 147K H800 GPU-hours, i.e. $202K + $10K + $82K = $294K, on 64×8 H800s for about 198 h and about 80 h — [R1 v2 text](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/deepseek-r1.txt)
- DeepSeek-V3: "requires only 2.788M H800 GPU hours for its full training", of which 2.664M is pretraining on 14.8T tokens — [DeepSeek-V3 README](https://github.com/deepseek-ai/DeepSeek-V3)

### Inferences
- **Puro-2B compute.** 6ND = 6 × 2×10⁹ × 1.4×10¹² = 1.68×10²² FLOP. Spread over 22,514 GPU-hours, that is about 2.1×10¹⁴ FLOP/s per GPU. That is plausible for FP8 on an RTX 5090, assuming a dense FP8 peak of roughly 4×10¹⁴ (my background figure, not verified here).
- **Puro-2B is not borrow-free.** The report's rule counts "anything the instance consumes that a large model generated (weights, data, labels, reward-model scores)" as borrowed compute. From background knowledge not re-verified in this sweep:
  - Nemotron-CC contains LLM-rephrased synthetic subsets and LLM-labelled quality classifiers;
  - SwallowMath is LLM-rewritten math;
  - MegaMath includes synthetic subsets;
  - agent "trajectories" are generated by LLM agents.

  Puro-2B therefore has nonzero, unquantified borrowed compute. The same logic likely applies to DCLM-7B, whose DCLM-Baseline filter was, from memory, trained on partly GPT-4-generated OpenHermes data. Under a strict reading, **DeepSeek-V3/R1 is the only LLM-level row the report can call essentially borrow-free.** Even that relies on DeepSeek's own disclosures.
- **VibeThinker-3B.** Its full ledger remains unknowable from public material. The report's 「未核实」 is appropriate.

### Gaps
- Exact synthetic fractions and generator models for Nemotron-CC, SwallowMath-v2, MegaMath and CoderForge were not checked.
- VibeThinker-3B's data-synthesis teacher models were not identified.

## Q4. Discovery and world-model evidence: DiscoRL, WorldCoder, PoE-World, Dreamer 4, LeWorldModel and AMI Labs

### Takeaway
All the quantitative claims hold:
- DiscoRL: compute, 754,778 parameters and unseen-environment generalization;
- WorldCoder: ~50 actions vs >1M steps;
- PoE-World: <1,000-frame demos and the only positive Montezuma score;
- Dreamer 4: 2B parameters, 256–1,024 TPU v5p, 0.7% diamonds;
- LeWM: 15M parameters, one GPU, 48× faster planning.

One attribution fails: LeWorldModel is not an AMI Labs paper by affiliation. AMI's $1.03B seed is confirmed.

### Cited Findings
- DiscoRL compute: "Disco57 was discovered using 1,024 TPUv3 cores for 64 hours, and Disco103 … 2,048 TPUv3 cores for 60 hours". Disco103 used 206 agents across Atari, ProcGen and DMLab-30 — [Nature](https://www.nature.com/articles/s41586-025-09761-x) (snippets from multiple results)
- `disco_rl/update_rules/weights/disco_103.npz` loads as 42 float32 arrays with **754,778** parameters in total; I verified this directly by loading the file — [disco_rl repo](https://github.com/google-deepmind/disco_rl)
- Project page source in the repo: "We evaluate both Disco57 and Disco103 on unseen domains: Crafter, Nethack, Sokoban"; "DiscoRL also generalises when used to train agents with much more parameters and data than those used for discovery"; released "under an open source Apache 2.0 licence" — [disco_rl docs/index.md](https://github.com/google-deepmind/disco_rl)
- Disco103 "achieved human-level performance on the Crafter benchmark and approached the state-of-the-art performance of MuZero on Sokoban" — [36kr summary](https://eu.36kr.com/en/p/3527315416767366); [Nature](https://www.nature.com/articles/s41586-025-09761-x) (snippet)
- WorldCoder "builds a world model over its first 50 actions". The deep-RL baseline "takes more than one million steps to learn two-box Sokoban". It uses GPT-4 with at most 50 requests per synthesis problem and front-loads about 400K tokens (about $15). ReAct solves only 15%±8% of basic levels — [structured reading note of arXiv 2402.12275](https://github.com/beapirate/awesome-self-improving-agents/blob/main/wiki/sources/arxiv-2402.12275.md); [WorldCoder repo](https://github.com/haotang1995/WorldCoder)
- PoE-World: demonstrations of "fewer than 1000 frames" per game; "the only method capable of achieving positive scores in Montezuma's Revenge". Its models exceed 4,000 lines of code vs WorldCoder's <100 — [MarkTechPost](https://www.marktechpost.com/2025/06/20/poe-world-outperforms-reinforcement-learning-rl-baselines-in-montezumas-revenge-with-minimal-demonstration-data/); [arXiv 2505.10819](https://arxiv.org/abs/2505.10819)
- Dreamer 4: "2B parameters---400M for the tokenizer and 1.6B for the dynamics model---on 256 to 1024 TPU-v5p"; "over 90% up to the stone pickaxe, … 29% for the iron pickaxe, and obtains diamonds in 0.7% of episodes"; uses the "2541 hours of contractor gameplay"; "100× less data" than VPT. The Gemma 3 VLA baseline reaches the iron pickaxe at 11% — [Dreamer 4 LaTeX mirror](https://github.com/edwhu/dreamer4-jax/blob/main/docs/main.txt)
- LeWM: "With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models" — [le-wm README mirror](https://github.com/aspamers/lewm); [official repo](https://github.com/lucas-maes/le-wm)
- LeWM authors and affiliations (v3, 3 Jun 2026): Maes (Mila & UdeM), Le Lidec (NYU), Scieur (Mila/Samsung SAIL), LeCun (NYU), Balestriero (Brown) — [paper reading note](https://github.com/Asada-Sinon/paper_reading/blob/main/2026-08/2026-08-15/leworldmodel.md). By contrast, LeVJEPA (Aug 2026) lists "Advanced Machine Intelligence (AMI Labs)" as an affiliation — [LeVJEPA paper.md](https://github.com/MLO-lab/LeVJEPA/blob/main/paper.md)
- AMI Labs: $1.03B seed announced 9–10 Mar 2026 at a $3.5B pre-money valuation. Investors include Bezos Expeditions, NVIDIA, Samsung, Toyota Ventures and Temasek. Paris HQ; CEO Alexandre LeBrun — [TechCrunch](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/); [Silicon Republic](https://www.siliconrepublic.com/start-ups/yann-lecun-ai-start-up-ami-raises-seed-funding-world-model); [HPCwire](https://www.hpcwire.com/aiwire/2026/03/11/yann-lecuns-ami-secures-1b-seed-to-develop-ai-world-models/)

### Inferences
- **DiscoRL discovery FLOP.** Taking about 61.5 TFLOP/s bf16 peak per TPUv3 core (half of the 123 TFLOP/s per chip; background figure), peak compute is about 1.45×10²² for Disco57 and 2.7×10²² for Disco103. At 30–50% utilization, Disco103 is about 0.8–1.4×10²² FLOP. This supports the report's "~10²²" for component-level discovery, excluding development and hyperparameter search.
- **Description length.** 754,778 × 32 bits ≈ 2.4×10⁷ bits (about 3 MB). The report's "about 5% of the functional-genome ceiling" arithmetic holds.
- **AMI Labs.** Absent a verified AMI byline on LeWM, the report should say "a LeCun-coauthored academic paper". The statement that AMI has released no language, math or reasoning system is consistent with everything found. It is not proven, only an absence of evidence.

### Gaps
- Dreamer 4 wall-clock training time was not located, so its FLOP stays an estimate.
- The DiscoRL Nature text itself was not read (blocked); the compute figures rest on consistent snippets.

## Q5. Continual learning: Dohare, plasticity scaling, Oak Lab, memory layers, skill libraries, Cursor, budget-matched memory

### Takeaway
The numbers are confirmed. Two framings are corrected:
- **Cursor.** Its +2.28% and −3.13% are cumulative A/B gains for Composer 1.5, not per 5-hour round.
- **Skill library.** The "up to 21%" drop comes from shadowing among human-authored skills at 202 skills. It is not a measured inflection "at ~100+ skills", and "points" vs percent is ambiguous.

### Cited Findings
- Dohare et al.: "binary classification performance dropped from 89% accuracy on an early task down to 77%, about the level of a linear network, on the 2000th task" — [arXiv 2306.13812v2](https://arxiv.org/html/2306.13812v2); [Nature](https://www.nature.com/articles/s41586-024-07711-7)
- Plasticity loss in GPT-style LMs: seen "across models ranging from 5M to 314M non-embedding parameters". Onset "follows a predictable scaling law, growing sublinearly with model size". It also occurs under stationary multilingual training — [arXiv 2606.24752](https://arxiv.org/abs/2606.24752)
- Oak Lab: co-founded in July 2026 by Sutton and Khurram Javed, formerly at Keen Technologies, in Toronto. Target: "an agent with one trillion parameters that learns and plans in real time while consuming 20 watts" — [TechTimes](https://www.techtimes.com/articles/320598/20260715/turing-award-winner-sutton-launches-oak-lab-calls-current-ai-fundamentally-broken.htm); [The Decoder](https://the-decoder.com/turing-award-winner-rich-sutton-founds-oak-lab-to-build-ai-agents-that-learn-on-their-own/); [BetaKit](https://betakit.com/ai-pioneer-richard-sutton-founds-new-research-lab/)
- Sparse memory finetuning: "NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop" — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103)
- Skill shadowing: "performance degrades as libraries grow -- by up to 21% when scaling from a small set of helpful skills to a 202-skill library … we formulate this performance degradation as the pass rate drop". Skill selection is the bottleneck; the context-overhead effect is "indistinguishable from zero" — [abstract mirror](https://github.com/JaredYe04/news-bot/blob/main/daily/2026-05-26-morning.md); [arXiv 2605.24050](https://arxiv.org/abs/2605.24050)
- Budget-matched web agents: the vanilla token-matched baseline "matches or surpasses" AWM, ASI and ReasoningBank across three WebArena domains and three models, often with fewer tokens. A similar trend holds on WorkArena-L1 with Qwen 3.6-27B — [arXiv 2606.15017](https://arxiv.org/abs/2606.15017)
- Cursor: each cycle collects "billions of tokens" of user interactions and runs evals "including CursorBench, to make sure there are no significant regressions". "This whole process takes about five hours." "We were able to improve Composer 1.5 via A/B testing behind Auto": +2.28% (agent edits persisting), −3.13% (dissatisfied follow-ups), −10.3% (latency) — [Cursor blog](https://cursor.com/blog/real-time-rl-for-composer); [mirror](https://github.com/kzinmr/ai-topics/blob/main/wiki/raw/articles/2026-06-05_cursor_real-time-rl-for-composer.md)

### Inferences
- The report's table row "生产环境实时RL: 每约5小时一轮，+2.28%编辑保留" should read "~5 h per cycle; +2.28% over some number of cycles in an A/B test (count not stated)". The per-cycle amortization factor is unknown.
- The skill-library result is about retrieval interference in curated libraries. It supports "adding skills without MDL-style compression hurts" but does not measure a learning agent's own library.

### Gaps
- The "~68% from wrong skill selection" figure and SkillFlow's 62.65→71.08 numbers were not re-verified.

## Q6. BabyLM world-knowledge evidence

### Takeaway
All BabyLM figures hold:
- 2024 EWoK: best 58.4% at 100M words and 54.6% at 10M;
- 2025 EWoK baselines: 49.47–52.44;
- 2025 Strict-track NLP winner: Simple Diffusion, NLP 58.4, below the GPT-BERT-masked baseline's 63.0.

The two "58.4" figures are different, independently confirmed quantities.

### Cited Findings
- 2024: "Most submissions … scored near the 50% chance level, and the maximum score was 58.4%" — [structured note of 2412.05149](https://github.com/borgr/paper-geo/blob/main/data/sidecars/findings-of-the-second-babylm-challenge-sample-efficient-pre.md). Also "10M-word track: … EWoK baseline 50.7, best 54.6. 100M-word: … EWoK baseline 51.9, best 58.4" — [second structured note](https://github.com/jeswr/kernel-of-truth/blob/main/docs/next/lit/EVAL.sources.jsonl); original [arXiv 2412.05149](https://arxiv.org/abs/2412.05149)
- 2025 EWoK baselines:
  - Strict-Small: GPT-BERT variants 49.47–50.23, GPT-2 Small 49.80;
  - Strict: 51.22–52.32;
  - Interaction (Preference Optimization): 52.44.

  — [evaluation-pipeline-2025 README](https://github.com/babylm/evaluation-pipeline-2025)
- 2025 winners:
  - Strict: human-likeness CLASS-IT; NLP Simple Diffusion (a "diffusion masked language model").
  - Strict-Small: human-likeness MoEP; NLP AMLM-Hard-Decay.
  - Interaction: BLM.

  Results table: Simple-Diffusion 12.6 / 58.4 / 35.5 (human-likeness / NLP / macro), vs baseline GPT-BERT-maskedMNTP 18.5 / 63.0 / 40.8. "The baselines (winning methods from previous years' challenges) remain strong" — [2025 Findings text](https://github.com/juand-r/claude-sandbox/blob/main/explorations/babylm2025/papers/p28.txt); [ACL Anthology](https://aclanthology.org/2025.babylm-main.28/)

### Inferences
- The report's claim that no model reaches world knowledge at ≤10⁹ words stands for the BabyLM regime. In 2025 no winner beat the baselines on the NLP macro score in the Strict track.

### Gaps
- 2025 per-submission EWoK scores for the winners were not extracted.

## Q7. Forecast and feedback-loop claims: METR, AI Futures Project, Forethought, Cunningham et al.

### Takeaway
All four are confirmed. METR's ">2×" is a model-based inference from Anthropic's self-reported 8× code-merge rate, not a measurement. The AI Futures Project has also published a newer Q2.5 2026 update.

### Cited Findings
- METR (Thomas Kwa, 8 Jul 2026): "Anthropic reported that contributors merge 8x as much code per day as before AI". Cobb-Douglas with β = 50% gives 2.83×; homogeneous CES gives [2.75, 2.91]; non-homogeneous CES gives [2.33, 2.66]. Caveats: verbosity, and "barely-useful code" — [METR](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/); [GreaterWrong](https://www.greaterwrong.com/posts/ix5qEyW9BjGEb4d8k/because-8-e-anthropic-s-researcher-uplift-is-plausibly)
- AIFP Q1 2026: "Daniel's Automated Coder (AC) median has moved from late 2029 to mid 2028, and Eli's … from early 2032 to mid 2030". TED-AI moved about 1.5 years sooner — [AIFP](https://blog.aifutures.org/p/q1-2026-timelines-update). A later Q2.5 update covers coding uplift (Daniel's median current uplift is 2×) and revenue — [AIFP Q2.5](https://blog.aifutures.org/p/q25-2026-timelines-update-uplift)
- Forethought (Davidson and Houlden 2025): r = 1.2, range 0.4–3.6; "~60%" chance of compressing >3 years of progress into <1 year; "~20%" for >10 years — [Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be); [EA Forum](https://forum.effectivealtruism.org/posts/eZ2hmSx2hYw6nfcwv/how-quick-and-big-would-a-software-intelligence-explosion-be)
- Cunningham, Althoff, Halperin, Jabarian, Koh, Ramani, Trammell, Whitfill and Wu, "The Economics of Recursive Self-Improvement" (arXiv 2609.15802, submitted 14 Sep 2026): "A back-of-the-envelope calculation suggests that feedback loops are not currently strong enough to generate a self-sustaining acceleration, though they appear to be strengthening" — [arXiv](https://arxiv.org/abs/2609.15802); [METR note](https://metr.org/notes/2026-07-22-economics-of-recursive-self-improvement/)

### Inferences
- The report's "实验室内研发提效已超过2倍" should be phrased as "plausibly >2× per a METR model applied to Anthropic's self-reported code volume". Its weight in the "discovery engine ≈60%" judgment is therefore slightly softer, but the direction is unchanged.

### Gaps
- The Q2.5 AIFP medians and Forethought's "~50% self-sustaining" figure were not re-verified.
