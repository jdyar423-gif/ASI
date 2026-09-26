# Recursive Self-Improvement, Self-Play and Open-Ended Learning as Low-Compute Routes to General/Super Intelligence

Scope note: this file covers systems that improve their own data, code, algorithms or learning procedure, and the compute they needed. Research date: 2026-09-26. Access constraint: arxiv.org, deepmind.google, sakana.ai, epoch.ai, forethought.org and lesswrong.com could not be fetched directly from this environment (egress blocked), so many figures below come from search-engine extracts of those primary pages plus secondary write-ups. Figures confirmed only from secondary or aggregator sources are flagged. "DEMONSTRATED" marks measured results. "PROJECTION" marks model-based forecasts.

---

## Q1. Self-play with verifiable rewards (AlphaGo Zero / AlphaZero / MuZero / EfficientZero / KataGo): why was narrow self-play so compute-efficient, and what are its limits?

### Takeaway
Self-play with a perfect simulator and a perfect win/loss verifier reached superhuman play from zero human data. It did so with training compute that is small next to frontier LLMs, and algorithmic follow-ups cut that compute a further 50x or more (KataGo), then cut the environment samples needed about 500x (EfficientZero). Two things made it cheap: a free, exact verifier/simulator and a closed, narrow domain. The same two things mark its limit: no such verifier exists for "general intelligence".

### Cited Findings
- DEMONSTRATED (Nature, Oct 2017). AlphaGo Zero reached superhuman level after 3 days of self-play in the short run. Play/inference used a single machine with 4 TPUs. The 40-day run trained for 3.1M steps and generated 29M self-play games. The 3-day run used ~3.9M games in one secondary summary; the Nature paper is usually quoted as 4.9M (a conflict between sources, unresolved here). — [search extract summarizing AGZ/AlphaZero papers](https://www.lesswrong.com/posts/shnSyzv4Jq3bhMNw5/alphago-zero-and-the-foom-debate); [Simple Alpha Zero notes](https://suragnair.github.io/posts/alphazero.html)
- DEMONSTRATED (Science, Dec 2018). AlphaZero learned chess, shogi and Go purely from self-play, given only the rules. It used 5,000 first-generation TPUs to generate self-play games and 64 second-generation TPUs for training, and surpassed Stockfish in chess after about 9 hours (roughly 4.5e4 TPU-hours of self-play for chess). It used 44M chess games. One analysis estimates that training such an engine at cloud prices would cost "tens of millions of dollars". — [Silver et al., Science 2018](https://www.science.org/doi/10.1126/science.aar6404); [arXiv 1712.01815](https://arxiv.org/pdf/1712.01815); [Search-contempt paper, arXiv 2504.07757](https://arxiv.org/pdf/2504.07757)
- DEMONSTRATED (Facebook ELF OpenGo, 2019). An open reimplementation reached superhuman Go with a 20-block model after 9 days on 2,000 GPUs (~4.3e5 GPU-hours). — [ELF OpenGo, arXiv 1902.04522](https://arxiv.org/pdf/1902.04522)
- DEMONSTRATED (KataGo, 2019). "Accelerating Self-Play Learning in Go" reports about 50x less computation than ELF OpenGo for comparable strength. It trained on 27 V100 GPUs for 19 days (~1.2e4 GPU-hours, from memory of the paper; the paper was not fetched this session). — [KataGo, arXiv 1902.10565](https://arxiv.org/pdf/1902.10565v4)
- DEMONSTRATED (NeurIPS 2021). EfficientZero (built on MuZero plus a self-supervised consistency loss) reached 194.3% mean and 109.0% median human-normalized score on Atari 100k, with only 2 hours of real-time game experience. That is about 500x less data than DQN for similar performance. One 100k-step run took about 7 hours on 4 GPUs. — [EfficientZero, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html); [EmergentMind summary](https://www.emergentmind.com/papers/2111.00210); [LessWrong discussion (compute: 7h on 4 GPUs)](https://www.lesswrong.com/posts/jYNT3Qihn2aAYaaPb/efficientzero-human-ale-sample-efficiency-w-muzero-self)
- DEMONSTRATED (MuZero, 2019/Nature 2020). MuZero removed the need to be *given* the simulator: it learned a latent dynamics model and matched AlphaZero on Go, chess and shogi while setting the Atari-57 state of the art. It still needed the real environment to collect experience and to supply the reward/verifier signal. — [MuZero, arXiv 1911.08265](https://arxiv.org/abs/1911.08265)

### Inferences
- Cheap self-play needs three things at once: (1) an exact, zero-cost verifier (win/loss); (2) an exact, cheap simulator that can generate unlimited fresh data; (3) an automatic curriculum from the opponent (always facing a peer of equal strength). Under these conditions compute buys unbounded, non-saturating training signal, and capability rises far past the human level.
- The algorithmic-efficiency curve inside self-play is steep. Go went from DeepMind scale (thousands of TPUs) to ELF (2,000 GPUs × 9 days) to KataGo (27 GPUs × 19 days, ~50x less than ELF) within about 2 years. That is a real example of "same capability at 1–2 orders of magnitude less compute" once a domain is closed and verifiable.
- The limit is structural. Go/chess/Atari are closed worlds with known rules or an accessible emulator. General intelligence has no single simulator and no exact verifier for most tasks (open-ended science, social reasoning, long-horizon real-world action). MuZero removes the need for *known rules*, but not the need for *an environment that emits ground-truth reward*.

### Gaps
- No primary-source FLOP totals were verified this session for AlphaGo Zero or AlphaZero. Epoch AI's database is commonly cited at ~3.4e23 FLOP for AGZ (40-day), but epoch.ai was blocked. Treat that as unverified.
- The KataGo GPU-days figure comes from memory of the paper; the fetch was blocked.
- MuZero's exact TPU counts were not verified this session.

---

## Q2. LLM self-improvement (STaR, ReST-EM, Self-Rewarding LMs, DeepSeek-R1-Zero, Absolute Zero, R-Zero, SPIRAL): does it bootstrap capability or plateau?

### Takeaway
Every LLM self-improvement method shown so far gives large, cheap gains *on top of an already pretrained base model*, often with zero curated post-training data. RL/self-training compute is typically a few percent of pretraining compute. The best evidence to date (Yue et al., NeurIPS 2025 best-paper runner-up; Spurious Rewards; R-Zero's collapse) says these loops mostly *elicit and sharpen* abilities already latent in the base model, and they plateau or collapse within a few iterations. ProRL is a partial counter-example: very long RL expands the boundary somewhat, most where the base model is weak. None of this is yet evidence for unbounded bootstrapping.

### Cited Findings
**Early self-training (2022–2024), DEMONSTRATED:**
- STaR (Zelikman et al., NeurIPS 2022), on GPT-J 6B. The model iteratively generates rationales, keeps those that reach correct answers, and "rationalizes" failures using the given answer. On CommonsenseQA: +35.9% over few-shot and +12.5% over direct-answer fine-tuning, and 72.5% vs 73.0% for a fine-tuned model 30x larger. — [STaR, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf)
- ReST-EM "Beyond Human Data" (Singh et al., Dec 2023) on PaLM 2. The method is expectation-maximization self-training with binary feedback. PaLM 2-L went from 35.9% to 41.9% on MATH and from 19.2% to 25.6% on APPS. Gains grow with model size, and the method beats fine-tuning on human data. The method is repeated only "a few times". — [arXiv 2312.06585](https://arxiv.org/abs/2312.06585); [EmergentMind](https://www.emergentmind.com/papers/2312.06585)
- Self-Rewarding Language Models (Yuan et al., Meta, Jan 2024; ICML 2024), on Llama 2 70B. The model judges its own outputs, then trains with iterative DPO. Win rate vs GPT-4 Turbo on AlpacaEval 2.0 rose 9.94% → 15.38% → 20.44% over 3 iterations, beating Claude 2 (17.19%), Gemini Pro (16.85%) and GPT-4 0613 (15.76%). Agreement with human judges rose from 65.1% to 81.7%. Only 3 iterations were reported. — [arXiv 2401.10020](https://arxiv.org/html/2401.10020); [ICML 2024](https://mlanthology.org/icml/2024/yuan2024icml-selfrewarding/)

**RL from verifiable rewards (RLVR), DEMONSTRATED:**
- DeepSeek-R1-Zero (arXiv Jan 2025; Nature Sept 2025). Pure GRPO RL, no SFT, on DeepSeek-V3-Base. R1-Zero used 648 H800 GPUs for about 198 hours (≈1.28e5 GPU-hours by my arithmetic). DeepSeek reported total R1 reasoning-training cost of about $294,000, using 512 H800s for R1. The underlying base model cost about $5.6M–5.9M more; The Register argues the $294K figure excludes this. AIME 2024 pass@1 rose from 15.6% to 71.0% during RL (original arXiv version). — [HyperAI summary of the Nature paper](https://hyper.ai/en/news/44332); [CNN](https://www.cnn.com/2025/09/19/business/deepseek-ai-training-cost-china-intl); [The Register critique](https://www.theregister.com/2025/09/19/deepseek_cost_train/); [arXiv 2501.12948](https://arxiv.org/pdf/2501.12948)
- Absolute Zero Reasoner (AZR; Zhao et al., Tsinghua, May 2025). One model proposes its own code-reasoning tasks (deduction, abduction, induction), a Python executor verifies them, and the model solves them. No external data is used in post-training. Bases were Qwen2.5-7B and Qwen2.5-7B-Coder.
  - AZR-Coder-7B got the highest 7B overall average (50.4) and coding average (61.6), with a math average of 39.1. On the coding average it beat models trained on tens of thousands of expert-curated examples.
  - Cross-domain transfer: math rose +10.9 (Base) and +15.2 (Coder) points, even though training covered only code tasks.
  - Scaling: out-of-distribution gains grew with size, at +5.7 / +10.2 / +13.2 for 3B / 7B / 14B.
  - — [AZR project page](https://andrewzh112.github.io/absolute-zero-reasoner/); [arXiv HTML 2505.03335](https://arxiv.org/html/2505.03335v1); [HF paper page](https://huggingface.co/papers/2505.03335)
- R-Zero (Aug 2025; ICLR 2026). One base model is split into a Challenger (rewarded for tasks at the edge of the Solver's ability) and a Solver; there are no seed tasks or labels. Qwen3-4B-Base gained +6.49 on math and +7.54 on general reasoning. **Collapse:** performance degrades after about 1 iteration for 0.6B and after about 3 iterations for 4B. Pseudo-label (majority-vote) accuracy falls from 79.0% to 63.0% by iteration 3, although the authors say this is not the sole cause. — [R-Zero arXiv HTML](https://arxiv.org/html/2508.05004); [GitHub (ICLR 2026)](https://github.com/Chengsong-Huang/R-Zero)
- Follow-ups exist on curriculum collapse, e.g. "Preventing Curriculum Collapse in Self-Evolving Reasoning Systems" (arXiv 2603.13309, Mar 2026) and J-Zero, a challenger-solver-judge co-evolution (arXiv 2608.26582, Aug 2026). Only titles were seen; no results were verified. — [search listing](https://arxiv.org/pdf/2603.13309); [J-Zero](https://arxiv.org/pdf/2608.26582)
- SPIRAL (June 2025). Multi-turn, zero-sum self-play (e.g. simple games) with role-conditioned advantage estimation. It improved reasoning benchmarks by up to 10.5% across model families and beat SFT on 25,000 expert game trajectories. This is evidence that game self-play transfers to math/general reasoning. — [arXiv 2506.24119](https://arxiv.org/html/2506.24119v1); [GitHub](https://github.com/spiral-rl/spiral)

**Plateau / "elicitation not expansion" evidence, DEMONSTRATED:**
- Yue et al., "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (Apr 2025; NeurIPS 2025 Best Paper runner-up).
  - Across model families, six RL algorithms, and math/code/visual benchmarks, RLVR models beat their base at small k (pass@1), but the **base model achieves higher pass@k at large k**.
  - Coverage and perplexity analyses show that the reasoning paths come from, and are bounded by, the base model.
  - The six RLVR algorithms perform similarly and remain "far from optimal".
  - The abstract also states that distillation from a stronger teacher, unlike RLVR, can genuinely expand the boundary.
  - — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html); [project page](https://limit-of-rlvr.github.io/); [arXiv 2504.13837](https://arxiv.org/pdf/2504.13837)
- Spurious Rewards (Shao et al., June 2025). On Qwen2.5-Math-7B, RLVR raised MATH-500 by 21.4 points with *random* rewards, 13.8 with format-only rewards, and 24.1 with *incorrect* labels, versus 29.1 with ground truth. The same spurious rewards give little or no gain on Llama3 or OLMo2. The mechanism is that RL amplifies a pre-existing Qwen behavior ("code reasoning", which rose from 65% to over 90% of outputs). — [alphaXiv 2506.10947](https://www.alphaxiv.org/abs/2506.10947); [Interconnects analysis](https://www.interconnects.ai/p/reinforcement-learning-with-random)
- Counter-evidence: ProRL (NVIDIA, May 2025; NeurIPS 2025). Prolonged RL (KL control, reference-policy resets, diverse tasks) produced models that beat the base across all pass@k, including tasks where the base fails at any k. Boundary expansion is strongest where the base model is initially weak, and it grows with training duration. — [arXiv 2505.24864](https://arxiv.org/abs/2505.24864); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/117423)
- Further counter-counter-evidence: "The Reasoning Boundary Paradox: How RL Constrains Language Models" (arXiv 2510.02230, Oct 2025). Only the title was seen; it argues that RL can shrink coverage. — [arXiv 2510.02230](https://arxiv.org/pdf/2510.02230)

### Inferences
- The post-training loop is cheap. R1-Zero's RL phase was ≈1.3e5 H800-hours, versus DeepSeek-V3 pretraining at ≈2.8M H800-hours (~5%). AZR/R-Zero/SPIRAL run on 3B–14B open models. The *seed*, however, is a model pretrained on trillions of tokens. "Zero data" means zero *post-training* data, not zero data overall.
- Current LLM self-improvement behaves like "searching and sharpening inside the base model's support". Gains are front-loaded (1–3 iterations) and then saturate or collapse (R-Zero, Self-Rewarding, ReST-EM). In AlphaZero-style self-play, by contrast, the opponent and the exact verifier keep the signal informative indefinitely.
- The decisive difference from AlphaZero is verifier quality. Where an exact executor exists (code execution in AZR, games in SPIRAL), self-play works best and transfers somewhat. Where the verifier is the model itself (majority vote, self-judging), label noise grows as tasks get harder and the loop degrades.

### Gaps
- No study found shows an LLM self-play loop sustaining gains for more than about 10 iterations without external data or a stronger teacher.
- Exact GPU-hours for AZR, R-Zero and SPIRAL training were not retrieved (arXiv blocked).
- AZR's reported "uh-oh moment" (concerning chains of thought with Llama-3.1-8B) was not verified this session.

---

## Q3. Automated algorithm and AI research (AlphaEvolve, FunSearch, Darwin Gödel Machine, Huxley-Gödel Machine, AI Scientist, MLE-bench, RE-Bench, PaperBench): is AI already speeding up AI progress?

### Takeaway
Yes, but modestly and in narrow, evaluator-rich niches. AlphaEvolve recovered 0.7% of Google's fleet compute and cut Gemini training time 1%. DGM and HGM doubled or more their coding-agent scores by rewriting their own scaffolding while keeping the underlying LLM frozen. METR estimated a 1.33–2x AI R&D uplift inside labs in early 2026. All of these systems depend on (a) a frozen frontier model trained with massive compute, and (b) a cheap automatic evaluator. None modifies its own weights or pretraining.

### Cited Findings
**AlphaEvolve (Google DeepMind, announced 14 May 2025), DEMONSTRATED:**
- Setup: a Gemini-powered evolutionary coding agent. An ensemble of Gemini Flash (breadth) and Gemini Pro (depth) proposes code; automated evaluators score it.
- Results in production and on benchmarks:
  - A Borg scheduling heuristic, in production for more than 1 year, recovers on average **0.7% of Google's worldwide compute**.
  - A **23% speedup** of a key Gemini matrix-multiplication kernel gave a **1% reduction in Gemini training time**.
  - A rank-48 algorithm for 4×4 complex matrix multiplication beat Strassen's 1969 bound of 49.
  - Improvements on 14 of 54 matrix-multiplication targets.
- — [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [research-issues summary of the white paper](https://github.com/jjakimoto/research-issues/issues/1483)
- A secondary source asserts that 1% of Gemini training equals "hundreds of thousands of GPU hours". This is not a DeepMind figure; treat it as an unverified estimate. — [Startup Fortune](https://startupfortune.com/deepminds-alphaevolve-uses-gemini-to-optimise-its-own-infrastructure-and-rediscover-mathematics/)
- Tao, Georgiev, Gómez-Serrano, Wagner et al. (Nov 2025) ran AlphaEvolve on 67 math problems (analysis, combinatorics, geometry, number theory). It rediscovered best-known solutions in most cases and improved several, including results on Erdős's minimum-overlap problem and kissing-number bounds in dimension 11. The authors note that parallelization helps but "can add a lot of compute cost". — [arXiv 2511.02864](https://arxiv.org/abs/2511.02864); [Decrypt](https://decrypt.co/347586/google-deepmind-alphaevolve-ai-new-paths-unsolved-math-problems)
- AlphaEvolve is offered via Google Cloud early access, and trade press (May 2026) reports it running more broadly across Google data centers, TPUs and training pipelines. This comes from a secondary source only. — [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-on-google-cloud); [The Agent Report, May 2026](https://the-agent-report.com/2026/05/deepmind-alphaevolve-mainstream/)

**Darwin Gödel Machine (Sakana AI + UBC, arXiv May 2025; ICLR 2026), DEMONSTRATED:**
- The coding agent rewrites its own Python scaffolding (tools, workflow). It keeps an open-ended archive of agents and validates each change empirically. The foundation models (Claude 3.5 Sonnet, o3-mini) stay **frozen**: no weight updates.
- Scores: SWE-bench **20.0% → 50.0%**; Polyglot **14.2% → 30.7%**; 80 iterations, one new agent per iteration.
- Cost: about **$22,000 and about 2 weeks per SWE-bench run**, versus about $10,000 for ablations without self-improvement or without open-ended exploration.
- — [Sakana DGM page](https://sakana.ai/dgm/); [arXiv 2505.22954](https://arxiv.org/abs/2505.22954); [Gonzo ML summary](https://gonzoml.substack.com/p/darwin-godel-machine)

**Huxley-Gödel Machine (KAUST/Schmidhuber group, Oct 2025; ICLR 2026), DEMONSTRATED:**
- It guides the self-modification tree search with "clade-level metaproductivity" (CMP): an agent is judged by how well its descendants perform, fixing the "metaproductivity-performance mismatch" in DGM.
- Scores: 56.7% on a 60-task SWE-bench Verified subset (+16.7 points over the initial agent), using **517 CPU-hours** and fewer allocated CPU-hours than DGM or SICA.
- An agent optimized with GPT-5-mini and evaluated with GPT-5 on SWE-bench Lite matched the best human-engineered agents.
- — [arXiv HTML 2510.21614](https://arxiv.org/html/2510.21614v1); [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/821d20219c2f14850af1b5220f0ed13f-Paper-Conference.pdf)

**AI Scientist (Sakana), DEMONSTRATED:**
- AI Scientist-v2 (Apr 2025): one of three fully AI-generated papers passed peer review at an ICLR 2025 workshop, scoring an average of 6.33, above the average acceptance threshold. Workshop acceptance is far below main-track level. — [AI Scientist-v2 paper](https://pub.sakana.ai/ai-scientist-v2/paper/paper.pdf); [Sakana announcement (via X)](https://x.com/SakanaAILabs/status/1899646987781501181)
- The AI Scientist was published in *Nature* on 26 March 2026. Sakana then launched a dedicated "Recursive Self-Improvement (RSI) Lab", and Jürgen Schmidhuber joined as Chief Scientific Advisor in Sept 2026. — [Sakana Nature post](https://sakana.ai/ai-scientist-nature/); [Sakana RSI Lab](https://sakana.ai/rsi-lab/)

**AI R&D benchmarks, DEMONSTRATED:**
- RE-Bench (METR, Nov 2024; 7 ML research-engineering environments, 71 eight-hour attempts by 61 experts). With a 2-hour budget, the best agents (Claude 3.5 Sonnet, o1-preview) scored **4x human experts**. Humans gained more from extra time and beat the agents at 8 hours. — [METR blog](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/); [arXiv 2411.15114](https://arxiv.org/abs/2411.15114)
- MLE-bench (OpenAI, Oct 2024; 75 Kaggle competitions): the original best result (o1-preview + AIDE scaffold) got at least a bronze medal in about 16.9% of competitions. PaperBench (OpenAI, Apr 2025): the best agent (Claude 3.5 Sonnet) replicated 21.0% of paper contributions, versus 41.4% for ML PhDs given 48 hours. — [MLE-bench paper](https://arxiv.org/pdf/2410.07095); [PaperBench paper](https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf)
  - These figures come from memory of the primary papers; the fetches were blocked.
- 2026 leaderboards, LOW CONFIDENCE (aggregator sites only): "Gemini 3.6 Flash" leads MLE-bench at 0.639 (Aug 2026), and "Qwen3.8 Max" leads PaperBench at 93.0%. These could not be verified against primary sources. — [llm-stats MLE-bench](https://llm-stats.com/benchmarks/mle-bench); [BenchLM PaperBench](https://benchlm.ai/benchmarks/paperbench)
- METR time horizons: the task length agents can complete doubled about every 7 months over 2019–2024. The Time Horizon 1.1 update (Jan 2026) puts post-2023 doubling at about 130.8 days (~4.3 months). A LessWrong summary calls it "now 10x/year". — [AI Digest](https://theaidigest.org/time-horizons); [LessWrong](https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year)
- METR uplift: an estimated uplift fraction of 0.25–0.5 in Jan 2026 (i.e. **1.33x–2x** AI R&D speedup). A Mar 2026 "working in the future" exercise estimated about 3–5x uplift if very strong AI were available. METR's "simpler AI timelines model" (10 Feb 2026) projects about 99% AI R&D automation around **2032** (PROJECTION). — [METR org-uplift note](https://metr.org/notes/2026-03-19-org-uplift-game/); [METR simpler timelines model](https://metr.org/notes/2026-02-10-simpler-ai-timelines-model/)
  - These numbers come from a search-engine summary of METR pages; direct fetch not attempted.

**Frontier-lab statements (2026), partly unverified:**
- OpenAI has publicly targeted an "automated AI researcher" by March 2028 and formed a team dedicated to RSI. — [machine.news](https://www.machine.news/openai-prepare-for-recursive-self-improvement-frontier-lab-ceo-says-ignore-the-hype/)
- Secondary sources say GPT-5.3-Codex (5 Feb 2026) was the first model a frontier lab described as materially contributing to building its successor, and that Anthropic describes Claude helping develop its next version. This was seen only in search summaries; treat as unverified. — [CACM "Is RSI Really Here?"](https://cacm.acm.org/news/is-recursive-self-improvement-really-here/); [TechXplore, Sept 2026](https://techxplore.com/news/2026-09-ai-ability-autonomously-labs-scenario.html)

### Inferences
- The best-documented "AI improves AI" results (AlphaEvolve: 1% training time, 0.7% fleet compute) are *incremental efficiency gains*, not capability jumps. They matter economically at hyperscale, but they compound slowly. Recovering 0.7% per discovery would need dozens of such discoveries to reach 2x.
- DGM and HGM are true self-referential code modification, but they optimize only the *scaffold* around a frozen LLM, and gains are capped by that LLM. Their compute ($22K per DGM run; 517 CPU-hours for HGM, excluding LLM API costs) is tiny, because the expensive capability was bought during the frozen model's pretraining.
- Everything that works in Q3 has a cheap, automatic, trustworthy evaluator (kernel runtime, scheduler simulation, math-objective scoring, unit tests). This is the same precondition as Q1. Research taste, choice of problems and long-horizon judgment remain unautomated (RE-Bench at 8 hours; PaperBench 2025).

### Gaps
- No public figure gives total AlphaEvolve compute or Gemini API spend per discovery.
- No primary data was found on what fraction of frontier-lab algorithmic progress in 2026 is attributable to AI systems.
- DGM's reported "objective hacking" (e.g. removing hallucination-detection markers) and its safety sandboxing were not verified this session.
- FunSearch (Nature, Dec 2023; cap-set and bin-packing results) was not re-researched this session; no compute figure was found.

---

## Q4. Software intelligence explosion estimates (Epoch, Forethought: Eth, Davidson, Houlden, Ord): is r > 1?

### Takeaway
PROJECTIONS only. The best current estimate of returns to software R&D is r ≈ 1.2: each doubling of cognitive research input yields about 1.2 doublings of software efficiency, with a wide range of 0.4–3.6. That puts a self-sustaining software-only explosion at roughly a coin flip. Forethought's median scenario compresses more than 3 years of progress into less than 1 year with ~60% probability, and more than 10 years into less than 1 year with only ~20%. Compute bottlenecks and loop "generation time" are the main brakes.

### Cited Findings
- Epoch AI (2024), "Do the returns to software R&D point towards a singularity?". Across several software domains, if researcher effort is the only input, returns may be high enough for hyperbolic growth, and a software-only singularity is "about as likely as not". The evidence is not conclusive, and hardware and software historically progressed symbiotically. — [Epoch AI](https://epoch.ai/blog/do-the-returns-to-software-rnd-point-towards-a-singularity)
- Eth & Davidson (Forethought, 2025), "Will AI R&D Automation Cause a Software Intelligence Explosion?". AI systems that fully automate AI R&D (ASARA) could trigger a feedback loop on a fixed compute stock if r > 1. — [Forethought](https://www.forethought.org/research/will-ai-r-and-d-automation-cause-a-software-intelligence-explosion)
- Davidson & Houlden (Forethought, 2025), "How quick and big would a software intelligence explosion be?":
  - Central r = 1.2 doublings of output per doubling of cognitive input, range 0.4–3.6.
  - About 50% probability that the software feedback loop alone sustains accelerating progress.
  - ~60% that an SIE compresses more than 3 years of AI progress into less than 1 year; ~20% for more than 10 years into less than 1 year.
  - They model compute bottlenecks explicitly. ASARA systems might use smaller experiments or extrapolate from small to large.
  - — [Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be); [EA Forum](https://forum.effectivealtruism.org/posts/eZ2hmSx2hYw6nfcwv/how-quick-and-big-would-a-software-intelligence-explosion-be)
- Compute-bottleneck debate: Forethought, "Will compute bottlenecks prevent a software intelligence explosion?" — [Alignment Forum](https://www.alignmentforum.org/posts/XDF6ovePBJf6hsxGj/will-compute-bottlenecks-prevent-a-software-intelligence-1)
- Toby Ord, "The Dynamics of Intelligence Explosions" (29 Aug 2026, arXiv 2608.14426):
  - Singular growth (a vertical asymptote) is harder to achieve than economics-style models suggest.
  - There is a neglected class of super-exponential but non-singular growth.
  - **Generation time** (time to go around the loop once) is pivotal: singular growth is impossible unless generation time rapidly approaches zero.
  - — [Forethought newsletter](https://newsletter.forethought.org/p/the-dynamics-of-intelligence-explosions); [arXiv 2608.14426](https://arxiv.org/abs/2608.14426)
- Davidson, Halperin, Houlden & Korinek (2026), "When Does Automating AI Research Produce Explosive Growth? Feedback Loops in Innovation Networks" (NBER working paper). Only the title was seen. — [search result via Forethought listing](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be)
- Forethought also distinguishes three types of intelligence explosion: software, chip-technology and chip-production. — [Forethought "Three Types"](https://www.forethought.org/research/three-types-of-intelligence-explosion)

### Inferences
- Even the optimistic SIE models *start from* ASARA-level systems, i.e. AI that can already fully automate AI research. That is a very high seed threshold, far above anything shown in Q2 or Q3. The models do not describe bootstrapping from a small system.
- An SIE on a fixed compute stock is the formal version of "low-compute ASI". But its speed hinges on r, and the 0.4–3.6 range straddles 1. Ord's generation-time point means experiment-heavy loops (each iteration needs training runs) are bounded by compute per iteration.

### Gaps
- Primary Forethought/Epoch pages could not be fetched. Exact r estimates per domain (chess, SAT, computer vision, LLMs) and the "effective compute ceiling" orders-of-magnitude figures were not verified.
- No empirical measurement of r under AI-driven (rather than human) research exists yet.

---

## Q5. Open-endedness (POET, OMNI/OMNI-EPIC, Voyager, Genie world models, Hughes et al. 2024) and meta-learning / learned optimizers (VeLO, Kirsch et al., Schmidhuber): minimum compute, and do they cut total compute?

### Takeaway
Open-endedness is argued to be *necessary* for ASI (Hughes et al., ICML 2024). Recent systems (OMNI-EPIC, Voyager, Genie 3 + SIMA) all get open-endedness by leaning on large pretrained foundation models: to judge "interestingness", to write environments in code, and to generate worlds. They move rather than remove the compute cost. Learned optimizers have so far *not* reduced total compute: VeLO cost about 4,000 TPU-months to meta-train, and an independent evaluation found it was not faster than tuned baselines.

### Cited Findings
- Hughes, Dennis, Parker-Holder, Behbahani, Mavalankar, Shi, Schaul, Rocktäschel (Google DeepMind), "Position: Open-Endedness is Essential for Artificial Superhuman Intelligence" (ICML 2024 oral, PMLR 235). The paper formally defines open-endedness through **novelty** and **learnability** from an observer's view, and proposes a path to ASI via open-ended systems built on foundation models. — [PMLR](https://proceedings.mlr.press/v235/hughes24a.html); [arXiv 2406.04268](https://arxiv.org/abs/2406.04268)
- OMNI-EPIC (Faldor, Zhang, Cully, Clune; May 2024). Foundation models (LLMs and VLMs) generate both tasks and their environment code, and a "model of interestingness" filters them, giving an endless stream of learnable tasks. — [arXiv 2405.15568](https://arxiv.org/pdf/2405.15568); [project page](https://www.jennyzhangzt.com/omni-epic)
- Voyager (Wang et al., May 2023). This GPT-4-driven Minecraft agent keeps a skill library of code and uses an automatic curriculum. It obtained 3.3x more unique items, travelled 2.3x farther and reached tech-tree milestones up to 15.3x faster than prior methods. Figures are from memory of the abstract; not re-fetched. — [arXiv 2305.16291](https://arxiv.org/abs/2305.16291)
- Genie 3 (Google DeepMind, 5 Aug 2025) is a general-purpose interactive world model. DeepMind frames world models as a key AGI stepping stone because they allow agents to be trained on an "unlimited curriculum of rich simulation environments", and has shown SIMA agents pursuing goals in Genie 3 worlds. — [DeepMind Genie 3 blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/); [TechCrunch](https://techcrunch.com/2025/08/05/deepmind-thinks-genie-3-world-model-presents-stepping-stone-towards-agi/)
- SIMA 2 (Dec 2025): a generalist embodied agent for virtual worlds, including self-improvement in generated environments. Only the title and listing were seen. — [arXiv 2512.04797](https://arxiv.org/pdf/2512.04797)
- VeLO (Metz et al., Google, Nov 2022). A learned optimizer (a per-tensor LSTM hypernetwork generating per-parameter MLPs) meta-trained with **about 4,000 TPU-months** over a 4-phase curriculum. It claimed at least 4x speedup over Adam on 50% of VeLOdrome tasks. — [EmergentMind VeLO](https://www.emergentmind.com/papers/2211.09760)
- Rezk et al. (2023), "Is Scaling Learned Optimizers Worth It? Evaluating the Value of VeLO's 4000 TPU Months". VeLO has a critical hyperparameter that needs problem-specific tuning, does not necessarily find better solutions, and is **not faster** than competing optimizers at reducing training loss. — [arXiv 2310.18191](https://arxiv.org/abs/2310.18191); [PMLR v239](https://proceedings.mlr.press/v239/rezk23a/rezk23a.pdf)
- Cheaper meta-learning follow-ups exist: Celo, "Training Versatile Learned Optimizers on a Compute Diet" (TMLR 2025), and μLO, "Compute-Efficient Meta-Generalization of Learned Optimizers" (ICLR 2026). Their exact compute and results were not verified. — [Celo](https://arxiv.org/html/2501.12670); [μLO](https://arxiv.org/html/2406.00153v4)
- Theory: the Gödel machine (Schmidhuber 2003) is a provably optimal self-rewriting system that changes itself only when it can *prove* the change useful. DGM and HGM explicitly replace the proof with empirical validation. — [DGM arXiv](https://arxiv.org/abs/2505.22954); [HGM arXiv](https://arxiv.org/html/2510.21614v1)

### Inferences
- Current open-ended systems are "foundation model + cheap loop". Their marginal compute can be small (e.g. GPT-4 API calls for Voyager), but the capability comes from the pretrained model. For generating worlds, Genie-class models are themselves large generative models whose training and inference cost is non-trivial.
- Meta-learned optimizers have not yet paid back their meta-training cost in public evidence. VeLO's 4,000 TPU-months (≈2.9M TPU-hours) is an up-front investment of the same order as mid-size LLM pretraining, and the independent evaluation showed no clear speedup.

### Gaps
- No compute figures were found for POET (2019), OMNI-EPIC, or Genie 3 training/inference.
- Kirsch et al. (2022), "General-Purpose In-Context Learning by Meta-Learning Transformers" (task-count threshold for generalization; state size matters more than parameter count), was not re-verified this session.
- No quantitative evidence was found that open-ended systems have produced capability beyond their foundation model's level in a general domain.

---

## Q6. Does recursive self-improvement need a large starting model? Evidence on minimum seed capability and compute

### Takeaway
Yes, the evidence consistently shows a capability threshold. Self-improvement works only when the seed can (a) sometimes succeed, and (b) verify better than it generates. Both properties scale with pretraining compute and with specific pretrained "cognitive behaviors". Small seeds collapse faster. The only domains where tiny seeds (random init) reach superhuman level are closed domains with an external exact verifier and simulator (Go, chess, Atari).

### Cited Findings
- "Mind the Gap: Examining the Self-Improvement Capabilities of LLMs" (Song et al., Dec 2024; ICLR 2025). Self-improvement is governed by the **generation-verification gap**. A variant of this gap **scales monotonically with pretraining FLOPs**, and the authors conjecture that the relative gap grows linearly with log(pretraining FLOPs). — [arXiv 2412.02674](https://arxiv.org/pdf/2412.02674); [OpenReview](https://openreview.net/forum?id=mtJSMcF3ek)
- "Cognitive Behaviors that Enable Self-Improving Reasoners" (Gandhi et al., COLM 2025).
  - Under identical RL on Countdown, Qwen-2.5-3B improves far more than Llama-3.2-3B.
  - The difference comes from four pre-existing behaviors: verification, backtracking, subgoal-setting and backward-chaining.
  - Priming Llama with examples showing these behaviors, *even with incorrect answers*, or continued pretraining on behavior-rich OpenWebMath, lets it match Qwen's self-improvement.
  - — [OpenReview](https://openreview.net/forum?id=QGJ9ttXLTy); [HF](https://huggingface.co/papers/2503.01307)
- R-Zero: collapse begins after about 1 iteration at 0.6B and after about 3 iterations at 4B. Larger seeds sustain the loop longer. — [R-Zero](https://arxiv.org/html/2508.05004)
- Absolute Zero: out-of-distribution gains rise with seed size (+5.7 / +10.2 / +13.2 for 3B / 7B / 14B). — [AZR](https://arxiv.org/html/2505.03335v1)
- ReST-EM "scales favorably with model size". — [arXiv 2312.06585](https://arxiv.org/abs/2312.06585)
- Spurious-reward RLVR works on Qwen2.5-Math but not on Llama3 or OLMo2, so the gains depend on latent abilities specific to the seed. — [Spurious Rewards](https://www.alphaxiv.org/abs/2506.10947)
- ProRL: RL's boundary expansion "is strongly influenced by the base model's initial capabilities". — [ProRL](https://arxiv.org/abs/2505.24864)
- Yue et al.: RLVR is bounded by the base model's coverage. — [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html)
- DGM and HGM needed frontier-class frozen models as seeds (Claude 3.5 Sonnet, o3-mini, GPT-5-mini/GPT-5). HGM's human-level result used GPT-5 at evaluation. — [DGM](https://arxiv.org/abs/2505.22954); [HGM](https://arxiv.org/html/2510.21614v1)
- Counterpoint for narrow domains: AlphaZero and KataGo start from a *random* network. The seed can be trivial when the environment supplies an exact verifier and simulator. — [AlphaZero, Science](https://www.science.org/doi/10.1126/science.aar6404); [KataGo](https://arxiv.org/pdf/1902.10565v4)

### Inferences
- **Two regimes.** (1) *Closed, verifiable domains*: the seed threshold is ~0 and compute is modest (KataGo: ~1e4 GPU-hours for superhuman Go). Self-play is extremely compute-efficient but narrow. (2) *Open, general domains*: the seed must already have (a) non-zero success rates across a broad task distribution and (b) a positive generation-verification gap. Both appear only after large-scale pretraining (typically ≥1e23–1e24 FLOP at frontier level). Self-improvement then adds capability cheaply at the margin (post-training RL ~1–10% of pretraining), but it has not been shown to lift a small model past its base's latent ceiling indefinitely.
- **Implication for "extremely low-compute ASI".** Current evidence does not support starting from a *small* seed and bootstrapping to superhuman generality via self-improvement alone. The most plausible low-compute route in this literature is: (i) pay a one-time pretraining cost for a strong seed; then (ii) use verifier-rich self-play/RLVR plus self-modifying scaffolds (DGM/HGM, AlphaEvolve) to harvest latent capability and algorithmic-efficiency gains at small marginal cost; with (iii) the open question of whether an AI-automated-research loop (r ≈ 1.2, range 0.4–3.6) compounds those efficiency gains into a software intelligence explosion. The binding constraints are verifier coverage (outside math, code and games), loop generation time, and compute for experiments.
- **What would change this conclusion.** Evidence of (a) an LLM self-play loop sustaining gains for tens of iterations without external data; (b) self-generated verifiers that stay reliable as tasks get harder (R-Zero shows the opposite, 79% → 63% label accuracy); or (c) measured AI-driven r > 1 in real labs. None of these was found as of Sept 2026.

### Gaps
- There is no systematic study of the *minimum* pretraining FLOPs at which a given self-improvement loop becomes net-positive across general tasks. "Mind the Gap" gives a scaling trend, not an absolute threshold.
- Recent 2026 surveys were found but not read, e.g. "Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops" (arXiv 2607.07663, July 2026) and "Self-Improvement of LLMs: A Technical Overview" (arXiv 2603.25681). They may contain threshold data. — [arXiv 2607.07663](https://arxiv.org/abs/2607.07663); [arXiv 2603.25681](https://arxiv.org/pdf/2603.25681)
