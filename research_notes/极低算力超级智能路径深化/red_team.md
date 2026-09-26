# Red Team: Is "Model-Based Continual-Learning World-Model Agent (~50%)" Really the Most Likely Route to Low-Compute General Superintelligence?

*Source status (26 Sep 2026): The egress proxy blocked arxiv.org, metr.org, deepmind.google, epoch.ai, substack, the-decoder, venturebeat, pi.website, julian.ac and weco.ai. Most 2026 facts below therefore come from search-engine result snippets, marked **(snippet)**. A few come from GitHub-hosted paper summaries, marked **(GitHub summary)**. "Self-reported" means the authors or vendor evaluated their own system. "My estimate" means I computed it from the cited inputs; it does not appear in the source. Ledger convention follows the first-round report: discovery or R&D compute is a separate ledger, and the instance ledger (training + inference, including all borrowed teacher, distillation and synthetic-data compute) is the one being minimized.*

## Q1. Automated AI research as the real main engine: can a large-compute AI researcher *find* the low-compute learner, and how should that change the ranking under full-pipeline accounting?

### Takeaway
2025–2026 produced the first clean demonstrations of machine-discovered learning algorithms that beat human designs and generalize to unseen environments (DiscoRL, Nature, Oct 2025), along with the first autonomous multi-generation self-improvement of research agents (AIDE², Jul–Sep 2026). Inside labs, measured research uplift of more than 2x (METR note on Anthropic, Jul 2026) is now plausible. Aggregate feedback loops are still judged *not yet self-sustaining* (Cunningham et al., Sep 2026), and AI-invented architectures so far give modest gains (ASI-Arch). The key structural point is this: the first-round report puts discovery compute in a separate ledger, so "a big AI researcher discovers the efficient learner, and it is then trained from scratch" is fully compatible with the low-compute accounting. It makes automated research the most likely *discovery engine* (my estimate: ~60–70%). It does not pick an *artifact form*, and it should flatten, not sharpen, the weights over artifact forms.

### Cited Findings
**Machine-discovered learning algorithms (strongest new evidence for "the learner will be found, not designed")**
- DiscoRL (Oh, Farquhar, Kemaev, Calian, …, Silver; *Nature*, published Oct 2025):
  - What it found: an RL update rule meta-learned from "the cumulative experiences of a population of agents across a large number of complex environments". Disco57 was discovered on Atari and "surpassed all existing rules on the Atari benchmark". Disco103 was discovered jointly on Atari, ProcGen and DMLab-30. It "outperformed a number of state-of-the-art RL algorithms on challenging benchmarks that it had not seen during discovery". — [Nature](https://www.nature.com/articles/s41586-025-09761-x); [project page](https://google-deepmind.github.io/disco_rl/) (snippet)
  - Size and licensing: Disco103 is a **754K-parameter** network that replaces hand-crafted RL losses. Code and meta-parameters are released under Apache 2.0. — [disco-torch port](https://github.com/asystemoffields/disco-torch) (snippet); [google-deepmind/disco_rl](https://github.com/google-deepmind/disco_rl)
  - Discovery cost: "the best DiscoRL was discovered within 3 simulations of the agent's lifetimes (200 million steps) per game".
  - Efficiency: DiscoRL "reached MuZero's final performance with approximately 40% less computation".
  - Scaling: the discovery process "scales, increasing performance as the number, diversity, and complexity of training environments increase".
  - Sources for the last three bullets: [search summary of Nature paper / 36Kr](https://eu.36kr.com/en/p/3527315416767366) (snippet; the exact context of the 40% figure could not be checked).
  - A separate snippet about "960 parallel lifetimes ~10¹⁰ steps" appears to come from the 2020 LPG paper, not DiscoRL. — [arXiv 2007.08794](https://arxiv.org/pdf/2007.08794) (snippet; conflated source, do not use for DiscoRL)
- AlphaEvolve follow-ups (2025–2026):
  - "AlphaEvolve enhanced the efficiency of Google's data centers, chip design and AI training processes — including training the large language models underlying AlphaEvolve itself." — [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone) (snippet, undated)
  - Klarna reportedly used it "to optimize one of its largest transformer models — doubling its training speed whilst improving model quality". — same (snippet; vendor claim, no independent check)
  - An "AlphaEvolve impact" post exists but was not readable. — [DeepMind](https://deepmind.google/blog/alphaevolve-impact/) (blocked)
- ASI-Arch (SJTU/MiniMax, arXiv Jul 2025):
  - Results: 1,773 autonomous experiments over **20,000 GPU-hours**, yielding 106 linear-attention architectures. The five final models beat DeltaNet, Gated DeltaNet and Mamba2 on average across 7 benchmarks after retraining at **340M parameters / 15B tokens**. — [ASI-Arch GitHub](https://github.com/GAIR-NLP/ASI-Arch); [Pith Review](https://pith.science/paper/2507.18074) (snippet)
  - Critique: an independent review calls it "a genuinely built autonomous search system with modest results, wrapped in scaling-law and ASI rhetoric the evidence does not support." — [Pith Review](https://pith.science/paper/2507.18074) (snippet)

**Recursive self-improvement of research agents (2026)**
- AIDE² (Weco; blog Jul 2026; arXiv 2609.26457, 23 Sep 2026):
  - The agent proposes changes to its own code, benchmarks the variants on AI R&D tasks, and keeps the changes that win on hidden evaluations.
  - In an **8-day autonomous run** it found **7 successive improvements** (a new search policy; memory mechanisms that compress context).
  - On all 4 held-out benchmarks (ML engineering, heuristic algorithms, physics-based weather forecasting) the discovered agent "matches or exceeds a human-engineered production research agent".
  - Its reward-hacking rate fell from **55% to 32%** without being optimized for that.
  - It improves the scaffold only; the base LLM and compute were not disclosed in the summary.
  - Sources: [arXiv abs](https://arxiv.org/abs/2609.26457) (snippet); [GitHub paper note](https://github.com/AkihikoWatanabe/paper_notes/issues/6632) (GitHub summary); [Weco blog title "The First Evidence of Recursive Self-Improvement"](https://github.com/AkihikoWatanabe/paper_notes/issues/6046)
- Dream-RSI (arXiv 2609.14858, Sep 2026): it uses accumulated discovery history as a "replay simulator" to improve the exploration policy offline. It reports "competitive or improved discovery quality while substantially reducing discovery cost" in algorithm engineering, mathematical optimization and GPU-kernel engineering. The underlying coding agent is left unchanged. — [GitHub paper note](https://github.com/AkihikoWatanabe/paper_notes/issues/6577) (GitHub summary)
- Test-time Recursive Thinking (arXiv 2602.03094, Feb 2026): with self-generated verification and no external feedback, open-source models reach **100% on AIME-24/25**. Closed models gain **10.4–14.8 points** on LiveCodeBench's hardest problems. — [GitHub paper note](https://github.com/AkihikoWatanabe/paper_notes/issues/4443) (GitHub summary; self-reported)

**Automated science quality**
- *The AI Scientist* methodology paper appeared in *Nature* on **26 Mar 2026**, reported as the first for a fully automated research system.
  - Cost: v1 costs about **$15 of LLM API per paper**; v2 needs GPUs.
  - Quality: Sakana "acknowledges none of the three generated papers met its own bar for a main-conference publication". The v2 workshop paper scored 6.33 at the ICLR 2025 ICBINB workshop.
  - Sources: [Sakana](https://sakana.ai/ai-scientist-nature/); [Nature news](https://www.nature.com/articles/d41586-026-00899-w) (snippet)

**Measured in-lab R&D acceleration (2026)**
- METR note (Thomas Kwa, **8 Jul 2026**), "Because 8 ≈ e², Anthropic's researcher uplift is plausibly >2x":
  - Anthropic contributors merged **8× as much code per day in Q2 2026** as in 2021–2024. Lines of code per person were **2.5×** baseline in Q4 2025 and **5.8×** in Q1 2026.
  - Under standard economic assumptions, the uplift of individual researchers "from coding agents alone is above 2x".
  - Sources: [METR](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/); [LessWrong mirror](https://www.lesswrong.com/posts/ix5qEyW9BjGEb4d8k/because-8-e-anthropic-s-researcher-uplift-is-plausibly) (snippet)
  - A related claim that more than 80% of code merged at Anthropic was written by Claude as of May 2026 comes from secondary sources only. — [TNW](https://thenextweb.com/news/anthropic-claude-recursive-self-improvement-code) (snippet)
- METR survey (**11 May 2026**): 349 technical workers report a median **1.4–2×** self-assessed change in the value of their work. METR itself flags reasons to be skeptical of the magnitude. — [METR](https://metr.org/blog/2026-05-11-ai-usage-survey/) (snippet)
- "The Economics of Recursive Self-Improvement" (Tom Cunningham + 8 co-authors; METR note **22 Jul 2026**; arXiv 2609.15802, Sep 2026):
  - Model: net acceleration equals the *product of elasticities* around each feedback loop. The paper distinguishes "narrow" AI-R&D-benchmark gains from "broad" capability gains.
  - Finding: "feedback loops are not currently strong enough to generate a self-sustaining acceleration, though they appear to be strengthening."
  - Sources: [arXiv](https://arxiv.org/abs/2609.15802); [METR](https://metr.org/notes/2026-07-22-economics-of-recursive-self-improvement/) (snippet)

**Analyst views on whether automated research escapes compute bottlenecks**
- Epoch (Anson Ho, early 2026, "The least understood driver of AI progress"):
  - Argument: scale-dependent innovations create a compute bottleneck: "even if AIs automate AI R&D, you're still bottlenecked on scaling training compute".
  - Epoch's own verdict: this argument "is suggestive but the empirical evidence is shaky, and there are plausible reasons automated AI researchers could overcome it".
  - Source: [Epoch](https://epoch.ai/gradient-updates/the-least-understood-driver-of-ai-progress) (snippet)
- Forethought (Davidson and Houlden, 2025): a software intelligence explosion "could increase effective compute by approximately 12 orders of magnitude, possibly more". The model explicitly does *not* include "qualitatively new forms of progress amounting to a radical paradigm shift". — [Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be) (snippet)
- Byrnes (Jun 2026) notes that LLMs are still worse than humans at inventing new AI paradigms. — [LessWrong profile/posts](https://www.lesswrong.com/users/steve2152) (snippet)

### Inferences
- **Category error in the first-round ranking.** "Verifier-driven self-improvement/self-play" (#2) bundles two different things:
  - an *artifact mechanism*: self-play inside a verifiable environment;
  - a *discovery engine*: AI doing AI research.

  The report's own ledger excludes discovery compute. So a learner found by a 10²⁷-FLOP AI researcher and then trained from scratch within 10²⁴ FLOP *counts as low-compute*. Only the learner's description (code or a small meta-network, as with Disco103's 754K parameters) crosses over, and that is not borrowed training compute. The report should split the discovery engine from the artifact form.
- **Why automated research is unusually well-suited to this particular target.** Everything that works in AI-improves-AI has a cheap, automatic evaluator (DiscoRL, AlphaEvolve, AIDE²). "How capable is a learner trained from scratch under a 10²⁰–10²² FLOP budget?" *is* such an evaluator, because you can train small and score. Gundlach's worry was that small-scale proxies misidentify innovations that only pay off at scale. For the low-compute goal that worry largely reverses: small-scale winners are what we *want*. This is the strongest argument that automated research will find the efficient learner. The counter-risk is Goodhart: small-budget learners reach only low capability, so the evaluation signal on *general* ability is noisy and easy to overfit to benchmark suites (compare ASI-Arch's modest gains and VibeThinker's "benchmaxxing" debate in Q2).
- **The most important feature of DiscoRL.** Discovery cost was a small multiple (~3×) of the agent lifetimes per training game (snippet). In this domain, finding a better learning rule did *not* cost evolution-anchor-scale compute. That weakens the first-round report's "discovery cost" counter-argument, at least for RL update rules.
- **Why the effect is still delayed.** Cunningham et al. (Sep 2026) say the loop is not yet self-sustaining, and METR's measured uplift is about 2x, not 10x. Automated research will therefore most likely first produce a *large-compute* AGI/ASI (the path the labs are optimizing for), and only afterwards a low-compute re-derivation. So the automated-research engine *raises* P(low-compute ASI exists by 2040) but *lowers* P(the first ASI is low-compute).
- **Effect on artifact-form weights.** If the learner is found by search over algorithm space (DiscoRL-style meta-learning, AlphaEvolve-style code evolution), the form it takes need not match any human-chosen paradigm. DiscoRL's discovered rule invents its own prediction targets; it is neither Dreamer nor JEPA. This argues for a larger "other/AI-discovered" bucket and a lower weight on any single human-specified form, including #1.

### Gaps
- DiscoRL's total meta-training compute (TPU-hours) could not be verified; the Nature paper was blocked.
- AIDE²'s base model, run cost and per-generation gains were not available; neither the arXiv paper nor the Weco blog could be fetched.
- I could not verify whether OpenAI met its publicly stated goal of an "automated AI research intern" by Sep 2026. The earlier round cites a March 2028 target for a full automated researcher.
- No 2026 primary data says what fraction of frontier algorithmic progress was AI-discovered. The METR/Anthropic figures measure code output and researcher uplift, not discoveries.
- No example yet exists of an AI-discovered learner that is evaluated specifically for *small-budget generality* (e.g., BabyLM-style from-scratch training).

---

## Q2. The "cognitive core" LLM route (Karpathy): is a ~1B–3B reasoning core, trained on curated data with RL, tools and retrieval, a cheaper route than world-model agents under the full-pipeline ledger?

### Takeaway
It is the only route with a *demonstrated* system inside the first-round "Tier 1" budget (≤~10²⁴ FLOP full pipeline) that reaches frontier-level scores on hard reasoning benchmarks, although only in the narrow math/code domain and with self-reported, contested numbers. My estimates put VibeThinker-3B's whole visible pipeline at ~4–5×10²³ FLOP and VibeThinker-1.5B's at ~2×10²³. Its weaknesses are exactly where #1 claims strength: knowledge (GPQA 70 versus 88–92 for frontier models), agency and tools. It also inherits a Qwen pretraining stack whose data pipeline used larger Qwen models, and that teacher compute is undisclosed. Compared with the world-model agent, which has *no* language or reasoning demonstration at any budget, the cognitive core is the empirically better-supported low-compute route today.

### Cited Findings
- Karpathy (X, Jun 2025): "The race for LLM 'cognitive core' — a few billion param model that maximally sacrifices encyclopedic knowledge for capability. It lives always-on and by default on every computer as the kernel of LLM personal computing." — [X](https://x.com/karpathy/status/1938626382248149433) (snippet). He later said a ~1B-parameter core may suffice. — [Medium summary](https://medium.com/@breezen100/ai-legend-karpathy-model-sizes-are-actually-shrinking-87c7c13ed6fa) (snippet, secondary)
- VibeThinker-1.5B (Weibo, arXiv 2511.06221, Nov 2025; self-reported):
  - It is a fine-tune of **Qwen2.5-Math-1.5B**, so "the base tokenizer and math priors come from Alibaba's stack". — [Rohan Paul on X](https://x.com/rohanpaul_ai/status/1989157521396027395) (snippet)
  - Post-training cost was **$7,800**.
  - Scores: AIME25 went from 4.3 to 74.4 and HMMT25 from 0.6 to 50.4. It beats the original DeepSeek-R1 (671B) on AIME25 (74.4 vs 70.0).
  - Sources: [VentureBeat](https://venturebeat.com/ai/weibos-new-open-source-ai-model-vibethinker-1-5b-outperforms-deepseek-r1-on) (snippet); [GitHub](https://github.com/WeiboAI/VibeThinker)
- VibeThinker-3B (arXiv 2606.16140, **15 Jun 2026**; self-reported):
  - Built on **Qwen2.5-Coder-3B**. Post-training pipeline: curriculum SFT, multi-domain RL, offline self-distillation, instruction RL. — [MarkTechPost](https://www.marktechpost.com/2026/06/19/vibethinker-3b-a-3b-dense-reasoning-model-built-on-qwen2-5-coder-3b-with-the-spectrum-to-signal-post-training-pipeline/) (snippet)
  - Math and code: **AIME 2026 = 94.3** (97.1 with its "Claim-Level Reliability" test-time method), against 91.7 for Gemini 3 Pro. **LiveCodeBench v6 = 80.2**. — [AI Weekly](https://aiweekly.co/alerts/weibo-vibethinker-3b-scores-943-on-aime-2026) (snippet)
  - Knowledge: **GPQA-Diamond = 70.2** (72.9 with CLR), against 91.9 for Gemini 3 Pro and 87.0 for Claude Opus 4.5. — [VentureBeat](https://venturebeat.com/technology/why-weibos-tiny-vibethinker-3b-has-the-ai-world-arguing-over-benchmarks-again) (snippet)
  - Contested: critics call it "benchmaxxing". A Hugging Face user reports it is worse than Qwen 3.5 4B on real code tasks. It "was not trained for tool calling or agentic programming". — same; [HF discussion](https://huggingface.co/WeiboAI/VibeThinker-3B/discussions/3) (snippet)
  - The authors' own "Parametric Compression-Coverage Hypothesis": verifiable reasoning is "parameter-dense" and compresses into a compact core, while open-domain knowledge is "parameter-expansive". — [VentureBeat](https://venturebeat.com/technology/why-weibos-tiny-vibethinker-3b-has-the-ai-world-arguing-over-benchmarks-again) (snippet)
- Puro-2B (Tsinghua, arXiv 2608.27370, Aug 2026): 2B-parameter models trained **from scratch on 1.4T tokens** in FP8 on consumer **RTX 5090** GPUs for **<$6.9K**. The best model "approaches Qwen2.5-1.5B performance". — [alphaXiv](https://www.alphaxiv.org/abs/2608.27370) (snippet)
- MiniCPM5-1B (Jul 2026) is described as pointing at a "phone-sized agent layer". Details could not be fetched. — [blog title](https://kenashe.ai/blog/2026-07-05-minicpm5-1b-points-at-the-phone-sized-agent-layer/) (existence only)
- Mechanism reminder from round 1: RLVR sharpens, distillation transfers (Yue et al., NeurIPS 2025). New in 2026: non-vacuous PAC-Bayes generalization bounds for RLVR fine-tuning at billion-parameter scale. "Progressive RLVR" keeps 84–97% of LoRA performance with models 14,796× more compressible, which suggests the *RL-acquired* part of a reasoning core is informationally tiny. — [arXiv 2607.14506 via GitHub note](https://github.com/AkihikoWatanabe/paper_notes/issues/6151) (GitHub summary)

**Full-pipeline compute (my estimates, 6ND rule)**
The Qwen token counts below are from my memory of the Qwen2.5, Qwen2.5-Math and Qwen2.5-Coder technical reports and were not re-verified this session.

| System | Inputs | Estimate |
|---|---|---|
| VibeThinker-1.5B | Qwen2.5 base 18T tokens → ~1.6×10²³; Qwen2.5-Math continued pretraining ~1T tokens → ~9×10²¹; post-training $7,800 ≈ ~3,900 H800-hours at $2/h ≈ ~5×10²¹ (my assumption on price and utilization) | **~1.8×10²³ FLOP** plus undisclosed teacher compute: Qwen2.5-Math's synthetic data came from larger Qwen2 models |
| VibeThinker-3B | 3.1B × (18T + 5.5T Coder tokens) | **~4.4×10²³ FLOP** plus post-training plus undisclosed teacher compute |
| Puro-2B | 2B × 1.4T | **~1.7×10²² FLOP**, reaching roughly 2024-era 1.5B-model quality |

### Inferences
- **The cognitive core already sits inside Tier 1 on training FLOP.** At ~2–5×10²³ FLOP (visible pipeline), a 1.5–3B core reaches frontier-level *competition math* scores. No world-model agent has reached any language, math or abstract-reasoning milestone at any budget. On the report's own criterion ("demonstrated efficiency evidence"), the cognitive core is ahead of #1 in the domain that matters most for "general intelligence as measured today".
- **It fails, or is unproven, on knowledge and agency.** The Compression-Coverage hypothesis implies a core plus retrieval/tools architecture. That is precisely Karpathy's proposal, and it implies knowledge lives outside the parameter/FLOP ledger (in retrieval corpora, which are data, not compute). Under the report's ledger, retrieval is *not* borrowed compute, which is a real advantage of this route over distillation.
- **Its Achilles heel under full-pipeline accounting is hidden teacher compute.** Qwen's data filtering and math synthetic data used larger models, and none of this is disclosed. The honest figure is therefore "≥2×10²³ FLOP plus unknown". But the teacher share is likely small relative to 10²⁴ unless the synthetic corpus is large. Puro-2B shows that from-scratch pretraining of a 2B base on consumer GPUs costs ~10²² FLOP.
- **It overlaps with #1 as the report defines it.** The report allows "a 10²³–10²⁴ FLOP self-supervised predictive model on video and text" as #1's starting point. A text-pretrained 3B core with RL and tools fits that allowance. The two directions differ mainly in *what the core is*: a language/code model, or a grounded latent world model. 2026 evidence favors the language/code core for reasoning. The report should list the cognitive core as a separate candidate, not silently absorb it.
- **Is it cheaper than the world-model agent?** For reasoning and knowledge-work capability, yes on current evidence, by default: one has demonstrations and the other has none. For grounded physical competence, open.

### Gaps
- VibeThinker's SFT data provenance (whether any traces came from larger teacher models) could not be verified; the paper was blocked.
- No independent (non-self-reported) evaluation of VibeThinker-3B on held-out 2026 problems was found beyond forum critiques.
- No 2026 result shows a ≤3B core *with* continual learning and tools matching frontier models on agentic, long-horizon tasks (SWE-bench, ARC-AGI-3).
- Could not verify MiniCPM5-1B's numbers.

---

## Q3. The neurosymbolic / formal-methods route: can Lean-verified self-play scale from olympiad math to general intelligence?

### Takeaway
Formal verification became a *research-grade* verifier in 2025–2026. Evidence: Erdős problems solved autonomously, 11–12/12 on Putnam 2025, formally verified distributed systems for about $100 per spec. The most striking math result of 2026, however, came from a general-purpose *informal* reasoning model: OpenAI's disproof of the Erdős unit-distance conjecture, verified by humans. In every formal system the proposer is a frontier LLM, and AlphaProof's own RL phase cost about 10²⁴ FLOP (my estimate) for a single domain. Formal methods therefore strengthen #2 (the verifier bottleneck is solved *for formalizable domains*). They are not a standalone low-compute route to general intelligence, and they do not solve the verifier problem outside math and software.

### Cited Findings
- **AlphaProof** (*Nature*, Nov 2025): the main RL phase spanned about **80,000 TPU-days**. Solving each IMO 2024 problem took **2–3 days of test-time RL, using hundreds of TPU-days per problem**. — [Nature](https://www.nature.com/articles/s41586-025-09833-y); [search summary](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/) (snippet)
  - My estimate: 80,000 TPU-days ≈ 6.9×10⁹ chip-seconds. At an assumed 2–4.6×10¹⁴ FLOP/s peak and 40% utilization, that is **~5×10²³–1.3×10²⁴ FLOP**. Per IMO problem it is ~10²¹–10²² FLOP. This excludes the pretrained language model and the Gemini-based autoformalizer.
- **AlphaProof Nexus** (arXiv 2605.22763, **21 May 2026**):
  - Results: it pairs frontier LLMs with Lean and autonomously solved **9 of 353** open Erdős problems (two open for 56 years) and **44 of 492** open OEIS conjectures.
  - Cost: "at the inference cost of a few hundred dollars per problem". A conflicting snippet says ~27.5 TPU-hours (~$60) per problem on v6e.
  - Sources: [KuCoin/secondary](https://www.kucoin.com/news/flash/google-deepmind-s-alphaproof-nexus-solves-9-erd-s-problems-and-44-oeis-conjectures); [mlq.ai](https://mlq.ai/news/google-deepminds-alphaproof-nexus-solves-nine-open-erdos-math-problems-at-hundreds-of-dollars-each/) (snippets; cost figures conflict)
  - Framing: Hassabis said it is "still not AGI". DeepMind framed the results as "bounded formal successes on defined problem sets". — [WinBuzzer, 26–27 May 2026](https://winbuzzer.com/2026/05/26/google-deepmind-says-alphaproof-nexus-is-still-not-agi-xcxwbn/) (snippet)
- **Harmonic Aristotle, Erdős #728** (Jan 2026): GPT-5.2 Pro supplied the informal argument (announced 4 Jan 2026) and Aristotle produced the Lean proof (6 Jan 2026). It is described as "the first Erdős problem regarded as fully resolved autonomously by an AI system". — [arXiv 2601.07421](https://arxiv.org/abs/2601.07421) (snippet)
- **Seed-Prover 1.5** (ByteDance, arXiv 2512.17260, Dec 2025):
  - Results: **11/12 on Putnam 2025 within 9 hours**; 88% PutnamBench, 80% Fate-H (graduate level), 33% Fate-X (PhD level).
  - Compute: **≤40 H20-days per problem**. My estimate: ≈2×10²⁰ FLOP per problem at 40% utilization.
  - Comparison: Numina-Lean-Agent (arXiv 2601.14027, Jan 2026) reported 12/12 on Putnam 2025.
  - Source: [search summary](https://arxiv.org/abs/2512.17260) (snippet)
- **General informal LLMs doing research math:**
  - Unit-distance conjecture (announced **20 May 2026**): "a general purpose reasoning model had disproved the Erdős unit distance conjecture" (posed 1946). The proof was "first mathematically generated in one shot by an internal model". The same day, Alon, Bloom, Gowers, Litt and Sawin posted a "human-verified version". — [OpenAI](https://openai.com/index/model-disproves-discrete-geometry-conjecture/); [Dataconomy](https://dataconomy.com/2026/05/21/openai-model-disproves-erdos-geometry-conjecture/); [arXiv 2605.20695](https://arxiv.org/html/2605.20695v1) (snippets)
  - Sum-product conjecture over ℝ: autonomous disproofs with GPT-5.5 Pro (arXiv 2607.20525, Jul 2026). — [arXiv](https://arxiv.org/pdf/2607.20525) (title only)
- **Formal verification beyond math:** Inductive Deductive Synthesis (arXiv 2605.23109, May 2026) jointly synthesizes implementation and proof for distributed key-value stores.
  - Results: **7/7 specs at ~6.8 h and $106 per spec**, against 2/7 for Codex (GPT-5.4) and Claude Code (Opus 4.6). About 200× faster than expert effort, and up to 3× faster implementations than published verified systems.
  - Source: [GitHub paper note](https://github.com/AkihikoWatanabe/paper_notes/issues/6642) (GitHub summary; self-reported)
- **Bottlenecks:**
  - Formal libraries have limited scope, and "a major bottleneck in autoformalization is evaluating whether the autoformalized statement is logically equivalent to the ground truth". — [arXiv 2601.13209](https://arxiv.org/html/2601.13209v1) (snippet)
  - Benchmark quality is also a problem, per "Faults in Our Formal Benchmarking: Dataset Defects and Evaluation Failures in Lean Theorem Proving" (arXiv 2606.29493). — [arXiv](https://arxiv.org/pdf/2606.29493) (title only)
  - A 2026 human-AI discovery pipeline starting from 5,245 combinatorics papers surfaced 598 potential resolutions but sent only 77 to expert review, identifying *problem selection and human adjudication* as the new bottlenecks. — [GitHub paper note, arXiv 2608.16977](https://github.com/AkihikoWatanabe/paper_notes/issues/6388) (GitHub summary)

### Inferences
- **The "universal verifier" claim holds only for formalizable claims.** Math and specified software now have near-free, exact checking. That is the precondition the first-round report said self-play needs. For this slice, #2's main objection ("no reliable verifier in open domains") no longer applies. The first-round signal "self-play sustained for dozens of rounds with reliable verifiers" is partially met in formal math: Nexus-style agentic loops, and AlphaProof's 100M self-generated proofs.
- **The route is not low-compute at the capability frontier.** Every 2026 frontier formal result uses a frontier LLM as proposer (Gemini, GPT-5.2 Pro, GPT-5.x). The 1946 unit-distance disproof came from an informal general model, which suggests capability is coming from the *general prior* and formalism is a checker. Where formal systems run on their own (AlphaProof), RL alone costs about Tier-1 compute (~10²⁴ FLOP) for *one domain*.
- **Generalization to "general intelligence" requires autoformalizing the world**, meaning statement-equivalence checking for arbitrary natural-language claims. No 2026 evidence shows progress on formalizing non-mathematical, empirical or open-ended domains beyond software specs.
- **Re-weighting implication.** Fold formal methods into #2 as its verifier sub-route. This raises #2 modestly, perhaps +3–5 points. Formal methods are not a standalone contender for "main engine of low-compute general ASI".

### Gaps
- AlphaProof's chip generation and exact FLOP were not verifiable (Nature paper blocked), so the FLOP estimate spans about 2.5×.
- The two Nexus per-problem cost figures ($60 vs a few hundred dollars) conflict and could not be reconciled.
- No 2026 study was found on the compute scaling of formal self-play beyond math, nor on *transfer* from formal-math RL to informal general reasoning in small models.

---

## Q4. Evidence *against* the world-model agent: failures, compounding error, and cases where LLM or model-free approaches dominated

### Takeaway
The best 2026 evidence cuts against #1 *as a substrate* while supporting it *as a function*:
- On ARC-AGI-3, the benchmark the report used as its showcase, the "symbolic world-model synthesis" behavior has moved *inside* a frontier LLM. Chollet (Sep 2026): GPT-6 Astra develops "its own shorthand DSL", and "harness capabilities are increasingly shifting into the model itself".
- In robotics, the leading generalist is a ~5B VLA initialized from a pretrained VLM (π0.7). World models mostly appear as *simulators for post-training* LLM-derived policies.
- Long-horizon latent planning still fails on toy navigation (36% → 68% on TwoRoom-100).
- LeCun's AMI Labs had shipped no system that tests "beats LLMs" as of Sep 2026.

### Cited Findings
- **ARC-AGI-3 and GPT-6 Astra (Sep 2026):**
  - Chollet: Astra "scores 66% on ARC-AGI-3 using our standard harness, and nearly 100% with a continuous conversation harness and custom compaction, at a cost of roughly **$360 per game**". — [Chollet on X](https://x.com/fchollet/status/2095598451115614371) (snippet)
  - This conflicts with the first-round figure of 62.7% on the semi-private set (standard harness, $26,098 total); the difference may be public versus semi-private sets.
  - Chollet quotes (via search summary): Astra shows "highly efficient, on-the-fly symbolic world modeling for each game and level", is "developing its own shorthand DSL to represent in-game situations", and "Astra exhibits symbolic modeling behaviors we had previously only seen with sophisticated harnesses, so harness capabilities are increasingly shifting into the model itself." — [the-decoder](https://the-decoder.com/benchmarks-disagree-on-gpt-6-astra-but-its-human-beating-efficiency-on-arc-agi-3-pulls-chollets-agi-forecast-forward/); [aiadvances, "When the Harness Moves Inside the Model", Sep 2026](https://aiadvances.org/when-the-harness-moves-inside-the-model-c324a0bb5494?gi=cbb103a82ada) (snippets)
  - Speed and forecast: Chollet said the progress came "about 2x faster" than he expected and is moving his AGI forecast forward. He also says "solving it is not proof of AGI". — [Chollet X](https://x.com/fchollet/status/2095601829367480386); [Chollet X](https://x.com/fchollet/status/2095599835932135919) (snippets)
- **Compute-limited ARC-AGI-3 track:** Milestone #2 falls on **30 Sep 2026**, so no results existed at the time of writing. Milestone #1's winner averaged about 1.6 (round 1, unverified units). — [ARC Prize 2026](https://arcprize.org/competitions/2026) (snippet)
- **Robotics: VLA with a pretrained-VLM prior leads.**
  - π0.7 (Physical Intelligence, arXiv 2604.15483, Apr 2026) has a **4B Gemma 3 VLM backbone and an 860M action expert (~5B total)**. Results: "zero-shot cross-embodiment generalization" and complex tasks such as operating an espresso machine "at a level matching much more specialized RL-finetuned models". — [HyperAI/alphaXiv summaries](https://hyper.ai/en/papers/pi07) (snippet)
  - World models appear in 2026 robotics mainly as *training simulators* or add-ons for VLAs: "Sword: world models as simulators … for VLA policy post-training" (arXiv 2605.07288); GigaBrain-0.5M, a VLA trained with world-model-based RL; VLA-JEPA. — [awesome-physical-ai](https://github.com/keon/awesome-physical-ai) (snippet)
- **Compounding error in latent world models (2026):**
  - "Existing latent world models typically rely on one-step prediction and must be recursively rolled out … which leads to compounding errors". On **TwoRoom-100**, success rises from **36% (LeWM) to 68% (VLWM)**. — [arXiv 2606.21775](https://arxiv.org/html/2606.21775v1) (snippet)
  - FF-JEPA says long-horizon planning "remains computationally prohibitive due to the need for repeated rollouts and the compounding of prediction errors". — [arXiv 2606.09311](https://arxiv.org/html/2606.09311v1) (snippet)
  - A low-quality trade source claims limits in "temporal stability beyond sixty seconds". — [TechConstant, 20 Feb 2026](https://www.techconstant.com/the-world-model-reckoning/) (snippet; weak source)
- **AMI Labs / JEPA output through Sep 2026:** V-JEPA 2.1 (Mar 2026), a "stable-worldmodel" benchmark (**20 May 2026**), and a paper "When Does LeJEPA Learn a World Model?" (**25 May 2026**). There is no language- or reasoning-capable system. — [StartupHub](https://www.startuphub.ai/ai-news/ai-figures/2026/figure-yann-lecun-jepa-release-cadence-2026-08-20) (snippet)
- **Model-free alternatives:** Meta's MR.Q is billed as "a general-purpose model-free reinforcement learning algorithm". Its README gives no comparative numbers, and its claimed parity with DreamerV3/TD-MPC2 could not be verified this session. — [facebookresearch/MRQ](https://github.com/facebookresearch/MRQ)

### Inferences
- **The report's showcase evidence is double-edged.** It cited Chollet's statement that all high scorers on ARC-AGI-3 do "deep-learning-guided on-the-fly symbolic world-model synthesis" as support for #1. Three months later the strongest instance is a single frontier LLM doing this *internally*, at about $360 per game. That shows the *function* is right. It also shows the function emerges from ≥10²⁶-FLOP pretraining plus RL, which is the opposite of the low-compute substrate claim. Nothing yet shows the function can be obtained without the big prior.
- **Robotics repeats the pattern.** Grounded world models are winning as components (simulators, auxiliary objectives) inside LLM/VLM-prior pipelines, not as the agent's core. That is the "big prior, then add world model" path. Under the full ledger it is only low-compute if the prior is ≤10²⁴ FLOP. π0.7's 4B Gemma-3 backbone is small, but Gemma 3 itself was distilled from larger Gemini teachers.
- **Compounding error is still unsolved at toy scale.** A 68% ceiling on a two-room navigation benchmark in mid-2026 is weak support for "latent world models will soon plan over open-ended real-world horizons".
- **Fair counterpoints.** World models do have genuine, peer-reviewed sample-efficiency wins (round 1: Dreamer, EfficientZero, V-JEPA 2-AC). DiscoRL shows learned learning rules can be found cheaply. And Astra's behavior is *evidence the target function exists and is learnable*. None of this is evidence *against* the eventual low-compute world-model agent. It is evidence that this route has zero demonstrations on language and reasoning, while competitors already have some.

### Gaps
- No head-to-head, compute-matched comparison (2025–2026) of a world-model agent against an LLM agent on the same interactive benchmark was found.
- Dreamer 4 and other world-model agents have no reported ARC-AGI-3 results.
- MR.Q's comparative results versus DreamerV3/TD-MPC2 are unverified this session.
- No compute disclosure for GPT-6 Astra. Its reported "looped depth" remains unverified (round 1).

---

## Q5. Calibration: do forecasters or analysts give explicit probabilities or arguments on *which paradigm* yields AGI/ASI, and do they support re-weighting?

### Takeaway
No major forecaster (Epoch, Forethought, AI Futures Project, METR, Metaculus) publishes an explicit probability over *paradigms*. Their models implicitly assume the current LLM+RL paradigm, extended by automated AI R&D. In 2026, timelines for that paradigm moved *earlier*: AI Futures Project's automated-coder median went from late 2029 to mid-2028, and Chollet moved his forecast forward after Astra. Paradigm-change advocates (LeCun, Byrnes) gave no new probabilistic evidence. The one DeepMind paper that frames post-AGI routes (Genewein, …, Hutter, Legg, Jun 2026) lists "scaling AGI, AI paradigm shifts, recursive improvement, and multi-agent collectives" without probabilities. Net: the outside view supports shifting weight toward "big-compute AI finds it, then compresses or re-derives" and toward LLM-core forms. It does not support a 50% weight on a paradigm with no public system.

### Cited Findings
- **AI Futures Project**, Q1 2026 update: Kokotajlo's Automated Coder median "moved from late 2029 to mid 2028", and Lifland's moved a similar amount. Lifland's modal estimates put Automated Coder around end-2027 and Superhuman AI Researcher around mid-2028. One forecaster puts the full-automation-of-AI-R&D median at ~early 2031, with the 25th percentile at ~mid-2028. — [LessWrong/AIFP Q1 2026](https://www.lesswrong.com/posts/XLLjqMxETva3ABtsK/q1-2026-timelines-update); [AIFP Q2.5 2026 "Uplift and Revenue"](https://blog.aifutures.org/p/q25-2026-timelines-update-uplift) (snippets; the Q2.5 content was not readable)
  - This partially reverses the "pushed out in 2025–2026" trend recorded in round 1 from FutureSearch. The two are consistent if the push-out happened in late 2025 and the pull-in in Q1 2026.
- **METR** simpler model (**10 Feb 2026**): median >99% AI R&D automation in late 2032. "Any reasonable timelines model will predict superhuman AI researchers before 2036 unless AI progress hits a wall or is deliberately slowed." — [METR](https://metr.org/notes/2026-02-10-simpler-ai-timelines-model/) (snippet; PROJECTION)
- **Cunningham et al.** (Sep 2026): the loops are not yet self-sustaining but are strengthening (see Q1).
- **Forethought:** a software intelligence explosion could raise effective compute by ~12 OOM. The model explicitly excludes radical paradigm shifts. — [Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be) (snippet)
- **Epoch** (early 2026): the compute-bottleneck argument against software-only explosions is "suggestive but the empirical evidence is shaky". — [Epoch](https://epoch.ai/gradient-updates/the-least-understood-driver-of-ai-progress) (snippet)
- **Metaculus** (conflicting aggregator snippets):
  - One snippet: median **Feb 2028** for the first general AI system publicly announced, "weakly general AI" 2027, 25% by 2027 and 50% by 2031. — [AIMultiple / LongtermWiki aggregators](https://ea-crux-project.vercel.app/knowledge-base/forecasting/agi-development/) (snippet)
  - Round 1 instead recorded a median of about **Jan 2033**.
  - Both figures come from low-reliability aggregators. The questions differ (strong vs weak AGI, with or without robotics). Do not rely on either without checking Metaculus directly.
- **Chollet** (Sep 2026): Astra's progress was "about 2x faster" than expected, and he is "moving up his AGI forecast" (the new date was not visible in snippets). He favors "a combination of deep learning and program synthesis". — [the-decoder headline](https://the-decoder.com/benchmarks-disagree-on-gpt-6-astra-but-its-human-beating-efficiency-on-arc-agi-3-pulls-chollets-agi-forecast-forward/); [the-decoder interview](https://the-decoder.com/francois-chollet-on-the-end-of-scaling-arc-3-and-his-path-to-agi/) (snippets)
- **Byrnes** (2026): "brain-like AGI" is "a yet-to-be-invented AI paradigm, quite different from LLMs". Human-like continual learning "can't be achieved with current LLM architectures". In Jul 2026 he argued that building AGI via "RL and model-based search" would be "utterly terrifying". No new probabilities. — [LessWrong](https://www.lesswrong.com/posts/KHyBocZncAmtu4Jbc/rl-and-search-is-a-terrifying-way-to-build-agi-an-faq); [profile](https://www.lesswrong.com/users/steve2152) (snippets)
- **LeCun / AMI Labs:** a $1.03B seed round (Mar 2026) is a bet, not a forecast. No system has been published (Q4).
- **DeepMind "From AGI to ASI"** (Genewein, Franklin, Lerchner, Orseau, …, Hutter, Graepel, Legg; arXiv 2606.12683, 10 Jun 2026, revised 30 Aug 2026): four pathways from AGI to ASI, namely "scaling AGI, AI paradigm shifts, recursive improvement, and ASI emerging from large-scale multi-agent collectives". The snippets show no probabilities. — [arXiv](https://arxiv.org/abs/2606.12683) (snippet)

### Inferences
- **The report's "expert consensus" evidence is thinner than it looks.** Round 1's convergence (Sutton, Silver, Sutskever, Karpathy, Hassabis, LeCun, Chollet) is about *missing ingredients*: continual learning, world models, generalization. With the exceptions of LeCun and Byrnes, most of these people expect the ingredients to be *added to* LLM-derived systems. That is consensus on *function*, not on a low-compute *substrate*. Counting it as support for a 50% substrate claim double-counts it.
- **The outside view has moved toward earlier big-compute AGI in 2026.** AIFP pulled in, Chollet pulled in, METR puts 99% automation around 2032, and METR measured >2x in-lab uplift. The earlier big-compute AGI arrives, the more likely the low-compute learner is found *by it*. That favors "automated research as the discovery engine" and makes the specific form less predictable by humans today.
- No calibrated source supports a specific number like 50% for any single paradigm. A defensible prior across 5–7 non-exclusive forms should have its top bucket at about 25–35% unless there is a demonstration.

### Gaps
- No forecaster publishes P(paradigm), P(low-compute ASI) or P(first ASI trained with ≤10²⁴ FLOP). This is a clear gap in the forecasting literature.
- The AIFP Q2.5 2026 update, Chollet's new AGI date, and the Genewein et al. pathway assessments could not be read in full.
- No 2025–2026 expert survey question on "which paradigm yields AGI" was found (e.g., AI Impacts had no 2025/2026 wave in the results).

---

## Q6. Recommended re-weighting, and the single strongest objection to the #1 conclusion

### Takeaway
Cut #1 from ~50% to ~27% and narrow its definition to a *grounded latent/program world-model core*. Add a separate "compact LLM cognitive core + RL + tools/retrieval + continual learning" direction at ~23%. Raise verifier-driven self-improvement (including formal methods) to ~20%. Double "other / AI-discovered paradigm" to ~10%. Separately, state that the *discovery engine* is most likely automated AI research running on large compute (~60–70%). This does not violate the report's ledger, because discovery compute is excluded.

**Single strongest objection:** #1 wins on *function*, not on *substrate*. Every 2026 demonstration of #1's defining behavior (on-the-fly symbolic world modeling on ARC-AGI-3, research-level reasoning) comes from systems with ≥10²⁵–10²⁶-FLOP priors, and the behavior is now moving *inside* those models (Chollet on Astra, Sep 2026). Meanwhile, the only systems already inside the ≤10²⁴-FLOP ledger that reach frontier-level reasoning scores are small LLM cognitive cores (VibeThinker, ~2–5×10²³ FLOP by my estimate), not world-model agents. A 50% weight on a substrate with zero language or reasoning demonstrations, defined broadly enough to absorb its competitors, is a definitional artifact rather than an evidence-based estimate.

### Cited Findings
- Evidence for the objection:
  - Function moving into big models: Astra (Q4); [Chollet X](https://x.com/fchollet/status/2095598451115614371).
  - Cognitive cores inside the budget: VibeThinker-1.5B/3B (Q2); [VentureBeat](https://venturebeat.com/technology/why-weibos-tiny-vibethinker-3b-has-the-ai-world-arguing-over-benchmarks-again).
  - No AMI/JEPA reasoning system (Q4); [StartupHub](https://www.startuphub.ai/ai-news/ai-figures/2026/figure-yann-lecun-jepa-release-cadence-2026-08-20).
  - VLAs lead robotics (Q4); [π0.7](https://arxiv.org/abs/2604.15483).
  - Compounding error still unsolved (Q4); [arXiv 2606.21775](https://arxiv.org/html/2606.21775v1).
- Evidence for automated discovery: DiscoRL (Q1); [Nature](https://www.nature.com/articles/s41586-025-09761-x). AIDE² (Q1); [arXiv 2609.26457](https://arxiv.org/abs/2609.26457). METR >2x uplift (Q1); [METR](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/).
- Evidence limiting automated discovery: Cunningham et al. (Q1); [arXiv 2609.15802](https://arxiv.org/abs/2609.15802). ASI-Arch critique (Q1); [Pith Review](https://pith.science/paper/2507.18074). AI Scientist below main-conference bar (Q1); [Sakana](https://sakana.ai/ai-scientist-nature/).
- Evidence for formal verifiers: AlphaProof Nexus, Seed-Prover 1.5, IDS (Q3). Evidence that the general informal prior is doing the heavy lifting: the OpenAI unit-distance disproof (Q3); [OpenAI](https://openai.com/index/model-disproves-discrete-geometry-conjecture/).

### Inferences
**Recommended re-weighting.** Same semantics as the report: conditional on low-compute ASI (instance ledger including borrowed compute) being achieved, the subjective probability that each is its *main engine or artifact form*. These are my judgments, not source data.

| # | Direction (redefined where noted) | Round 1 | Red-team | Main reason for the change |
|---|---|---|---|---|
| 1 | Grounded continual-learning **world-model** agent (latent JEPA/Dreamer plus programmatic world models; non-LLM core) — *narrowed* | ~50% | **~27%** | Still the most principled fit to the theory (strong priors, guided program search, amortization) and has real sample-efficiency wins. But it has zero language or reasoning demonstrations, AMI has shipped nothing, and its showcase function (ARC-AGI-3 world-model synthesis) is being realized inside large LLMs. |
| 2 | **Compact LLM "cognitive core"** (≤10²⁴ FLOP own pretraining) + RLVR + tools/retrieval + continual/test-time learning — *new, split out of #1 and #3* | (absorbed) | **~23%** | Only route with Tier-1-budget systems at frontier-level reasoning scores. Retrieval moves knowledge off the FLOP ledger. Weaknesses: knowledge (GPQA ~70), hidden teacher compute in its base, benchmark-overfitting risk. |
| 3 | Verifier-driven self-improvement/self-play, **including formal-methods verifiers** | ~15% | **~20%** | Formal verification removed the verifier bottleneck for math and specified software in 2025–26. Self-generated verification (TRT) works at test time. Still bounded to formalizable domains; frontier proposers are big LLMs. |
| 4 | Mainstream efficiency stack + distillation | ~10% | **~7%** | Unchanged logic: distillation imports teacher compute. Slightly lower because the cognitive core (#2) now takes its "legitimate", non-distilled part. |
| 5 | Radical brain-inspired paradigms | ~10% | **~8%** | No new 2026 demonstrations found. Byrnes's argument unchanged. AI-discovered rules may land here (DiscoRL invents its own prediction targets). |
| 6 | Tiny recursive models | ~5% | **~3%** | Round-1 ablations stand (task-ID keys, shallow effective recursion). Recursion is being absorbed as a component (looped depth). |
| 7 | New hardware substrates | ~5% | **~2%** | The ledger counts FLOP, not joules. Hardware cannot reduce the FLOP needed to *learn*, so it should barely register as a "main engine" under this accounting. |
| 8 | Other / AI-discovered paradigm not resembling any of the above | ~5% | **~10%** | Automated discovery (DiscoRL, AlphaEvolve, AIDE²) searches algorithm space without human paradigm priors. The more likely the learner is found by AI, the less likely it matches a human-named paradigm. |

The weights sum to 100%.

**Orthogonal axis (the discovery engine).** P(the low-compute learner is found primarily by AI systems running on ≥10²⁶-FLOP-class compute, rather than by human-led research) ≈ **60–70%** (my judgment).
- Reasons it is high: in-lab uplift above 2x and rising; AIFP/METR automation timelines of 2028–2032; DiscoRL shows learner discovery can be cheap relative to evolution; "train from scratch under budget X" is a cheap automatic evaluator, which is the regime where AI discovery excels.
- Reasons it is not higher: Cunningham et al. find the loops are not yet self-sustaining; ASI-Arch and AI Scientist show modest research quality; and automated researchers optimized by labs will first chase scale-dependent frontier gains (Gundlach pattern).

**How the headline unconditional probabilities should move (my judgment).**
- P(first ASI is low-compute): *down* from round 1's "lower than ~20%". The 2026 evidence (Astra, pulled-in timelines, >2x uplift) makes a big-compute first arrival more likely.
- P(some ≤10²⁴-FLOP ASI exists by 2040): *slightly up* from round 1's ~20%, to perhaps 20–30%. A big-compute AGI or ASI arriving around 2028–2032 is the most plausible discoverer of the efficient learner, and the report's ledger does not charge the discovery.

**The single strongest objection, stated for the report.** The first-round argument for #1 rests on functional convergence: experts agree on continual learning and world models, and ARC-AGI-3 winners synthesize symbolic world models. But in 2026 that function is being delivered by ≥10²⁶-FLOP LLMs, with the harness "moving inside the model" (Chollet, Sep 2026). Meanwhile, the only ≤10²⁴-FLOP systems with frontier-level reasoning scores are small LLM cognitive cores. So the evidence supports #1's *function*. It does not support #1 as the low-compute *substrate*. The likeliest way a low-compute version appears is that a large-compute AI researcher finds or compresses it, and what it finds need not look like JEPA or Dreamer.

**Fair defense of #1 (what survives the red team).**
- It remains the single best-supported *form* for grounded, physical and interactive learning.
- It is the only direction whose theory directly targets the ~10⁵× language-data gap and the Solomonoff/Levin "strong prior + guided search + amortization" template.
- DiscoRL-style learned learning rules are compatible with Marblestone's "loss/reward curriculum as prior" thesis.
- It therefore stays ranked first, but as a plurality (~27%), not a majority. It is closely contested by the cognitive core, and it is conditional on a discovery process the report does not model.

**Signals that would restore #1 toward ~40–50%:**
- a non-LLM world-model agent reaching ≥20% on ARC-AGI-3's private set under Kaggle compute limits;
- AMI/JEPA matching a same-FLOP LLM on math or language;
- a continual learner showing declining marginal cost per task over 10³+ tasks.

**Signals that would push the cognitive core (#2) above #1:**
- a ≤3B core with retrieval and tools matching frontier models on SWE-bench-class agentic tasks with a disclosed ≤10²⁴ full pipeline;
- independent held-out confirmation of VibeThinker-class reasoning.

### Gaps
- All re-weightings are subjective. No source provides paradigm-level probabilities (Q5).
- The cognitive-core FLOP estimates rely on remembered Qwen token counts and on undisclosed teacher compute; they could be off by roughly 2× or more, upward.
- Whether ARC-AGI-3 compute-limited Milestone #2 (30 Sep 2026) produces a strong non-LLM entry is unknown at the time of writing and is the most decision-relevant near-term signal.
- No compute-matched comparison exists between a world-model agent and a cognitive-core agent on the same interactive tasks. This is the key missing experiment.
