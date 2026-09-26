# World-model agents (latent-predictive and program-based): newest evidence on generality and full-pipeline compute (status as of 26 Sept 2026)

Method note: This session's egress proxy blocked arxiv.org, ar5iv, alphaxiv, openreview, huggingface, papers.neurips.cc, emergentmind, pith.science, danijar.com, basis.ai, tufalabs.ai, docs.arcprize.org, techtimes, implicator.ai and medium.com. GitHub READMEs were readable. Most numbers below therefore come from search-engine extracts of primary pages; these are cited to the primary URL and marked **(snippet)**. Numbers read directly from a GitHub README are marked **(README)**.

Evidence tags:
- **[peer-reviewed]**: conference or journal paper.
- **[preprint]**: arXiv only.
- **[self-reported]**: authors' own eval, usually the ARC-AGI-3 public set.
- **[verified]**: ARC Prize hidden-set test.
- **[est.]**: my own arithmetic.

This file does not repeat background already in `brain_inspired_alternatives.md` §1, `small_models_program_synthesis.md` KQ1 or the round-1 report. That background covers: the basics of DreamerV3, Dreamer 4 (100x less data), V-JEPA 2 and V-JEPA 2-AC, I-JEPA, LeJEPA and SIGReg; the AMI Labs $1.03B seed; the Genie 3 demo; the ARC-AGI-3 format and RHAE; frontier ARC-AGI-3 scores; Chollet's "DL-guided on-the-fly synthesis of symbolic world models" quote; and the basic Milestone #1 winners list.

---

## 1. Latent-predictive world models (Dreamer 4, TD-MPC2, EfficientZero V2, BBF, transformer WMs, Genie 3/SIMA 2, V-JEPA 2, LeWorldModel, AMI Labs, World Labs): what have they achieved, and at what full-pipeline compute?

### Takeaway
Latent world models keep producing large gains in **environment-data efficiency** and **planning speed**, but only in perception and control domains. Dreamer 4's headline result shows the limits clearly:
- The offline Minecraft agent obtains diamonds in only **0.7% of episodes**.
- Its 2B-parameter model was trained on a **256–1,024-chip TPU v5p** slice. That is data-cheap but not compute-cheap.

The genuinely low-compute systems are narrow control agents, with no language or abstract reasoning:
- LeWorldModel: 15M parameters, a few hours on one GPU.
- BBF: about 10 GPU-hours per Atari game.
- TD-MPC2: 317M parameters for 80 control tasks.

The most general embodied agent, SIMA 2, gets its generality from **Gemini**, not from a learned world model. Genie 3 is used there as a training "dojo". Its compute is undisclosed.

### Cited Findings

**Dreamer 4 (Hafner, Yan et al., Google DeepMind; arXiv 29 Sept 2025) [preprint]**

Compute:
- Training uses about **2B-parameter** models on **256–1,024 TPU v5p chips**. Inference runs interactively at **≥20 FPS on a single H100**, with a 9.6 s context (6x longer than prior models) — [arXiv 2509.24527](https://arxiv.org/abs/2509.24527) (snippet via EmergentMind/Harold Benoit notes); [Dreamer 4 notes](https://haroldbenoit.com/notes/ml/llms/multi-modality/video/dreamer-4) (snippet).
- Training duration and total FLOPs were not found.

Tokenizer and pipeline:
- Tokenizer: 384×640 frames, 960 patch tokens per frame, compressed to 256 spatial tokens per frame; 192-frame context — [arXiv 2509.24527](https://arxiv.org/pdf/2509.24527) (snippet).
- Three phases:
  1. Pretrain the causal tokenizer and dynamics model on video.
  2. Insert agent tokens and fine-tune to predict actions and rewards.
  3. Run RL entirely inside imagination, with no online environment interaction — [InfoQ, Oct 2025](https://www.infoq.com/news/2025/10/dreamer-4-minecraft-agent/) (snippet).

Capability:
- Success rate: >90% up to the stone pickaxe, **29% iron pickaxe**, **0.7% diamonds**.
- This beats all prior offline agents, including fine-tuned VPT and a Gemma-3 VLA baseline, which score 0% on diamonds.
- Data: about 100x less than VPT's 2,500 h of contractor play plus 270,000 h of pseudo-labelled web video — [EmergentMind summary of 2509.24527](https://www.emergentmind.com/papers/2509.24527) (snippet); [implicator.ai](https://www.implicator.ai/dreamer-4-mines-diamonds-in-an-imagined-minecraft/) (snippet).

My estimate of training compute [est.]:
- One epoch over 2,500 h of video at 20 FPS with 256 latent tokens per frame is about 4.6×10^10 tokens.
- With 6·N·D and N = 2B, that is about 5.5×10^20 FLOP per epoch.
- One day on 256–1,024 v5p chips at 40% utilisation is roughly 4×10^21–1.6×10^22 FLOP.
- So total training is plausibly around 10^21–10^22 FLOP. The frame rate and number of epochs are unverified.

**Model-based RL on standard sample-efficiency benchmarks (2023–2025)**

- **BBF** (ICML 2023) [peer-reviewed]. BBF is model-free but serves as the reference point.
  - It reaches superhuman IQM on Atari 100k in **about 10 A100 GPU-hours per game**, versus more than 40 for model-based EfficientZero.
  - That is a runtime reduction of at least 4x at similar performance — [BBF, arXiv 2305.19452](https://arxiv.org/pdf/2305.19452) (snippet).
- **EfficientZero V2** (ICML 2024) [peer-reviewed]: Atari 100k mean human-normalised score **2.428**, median **1.286**, above BBF and EfficientZero. GPU-hours were not found — [arXiv 2403.00564](https://arxiv.org/html/2403.00564v2) (snippet).
- **TD-MPC2** (ICLR 2024) [peer-reviewed]:
  - One **317M-parameter** agent handles **80 tasks** (50 Meta-World and 30 DMControl) across embodiments and action spaces.
  - Normalised score rises from **16.0 at 1M parameters to 70.6 at 317M**, roughly log-linear, with no saturation and the same hyperparameters at every size — [arXiv 2310.16828](https://arxiv.org/abs/2310.16828); [tdmpc2.com](https://www.tdmpc2.com/) (snippet).
- **Transformer world models on Craftax-classic** (Dedieu et al., DeepMind; ICML 2025) [peer-reviewed]:
  - Reward **69.66% after only 1M environment steps**, versus **53.2% for DreamerV3** and **65.0% for human experts**.
  - It is the first method to beat human performance on Craftax-classic with limited data.
  - The method combines Dyna with warmup, a nearest-neighbour patch tokenizer and block teacher forcing — [arXiv 2502.01591](https://arxiv.org/pdf/2502.01591); [ICML 2025 poster](https://icml.cc/virtual/2025/poster/45728) (snippet).

**JEPA line, 2026 (LeCun/Balestriero group; AMI Labs)**

- **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero; arXiv 2603.19312, Mar 2026) [preprint]. Described as the first JEPA trained stably end-to-end from raw pixels, with two losses: next-embedding prediction plus a Gaussian regulariser. This cuts tunable loss hyperparameters from 6 to 1.
  - Size and cost: **about 15M parameters, trainable on a single GPU in a few hours** (README).
  - Planning cost: each frame is **one 192-dimensional token**, about **200x fewer tokens than DINO-WM**, so planning takes **about 1 s versus 47 s (48x faster)**.
  - Results: it "significantly outperforms DINO-WM on Push-T and OGBench-Cube".
  - Tasks: PushT, Cube, TwoRooms, Reacher.
  — [LeWM GitHub](https://github.com/aspamers/lewm) (README); [arXiv 2603.19312](https://arxiv.org/html/2603.19312v1) (snippet)
- Secondary sources call LeWM "the first public proof point from AMI Labs", released about two weeks after the $1.03B raise. The README does not list affiliations, so the AMI Labs attribution is **unconfirmed** — [Medium/B. Vishal](https://medium.com/@bathri06vishal/leworldmodel-how-yann-lecun-solved-the-hardest-problem-in-world-models-on-a-single-gpu-555829cf9ab1) (snippet); [Taskade](https://www.taskade.com/blog/ai-world-models) (snippet).
- Same secondary source: a "stable-worldmodel" benchmark was released on **20 May 2026**, and "When Does LeJEPA Learn a World Model?" followed on **25 May 2026**. Both are unverified beyond the snippet — [Taskade](https://www.taskade.com/blog/ai-world-models) (snippet).
- Follow-ups visible by title only: "Fast LeWorldModel" ([arXiv 2606.26217](https://arxiv.org/pdf/2606.26217)) and UniJEPA, a task-agnostic visual world model ([arXiv 2608.07409](https://arxiv.org/pdf/2608.07409)). Contents were not read.
- **V-JEPA 2 compute** (June 2025) [preprint]:
  - ViT-g, about 1B parameters, trained for **252K iterations** (extended from 90K).
  - Progressive resolution: 16 frames at 256² for most of training, 64 frames at 384² only in the cooldown. This **cut GPU time 8.4x**.
  - A follow-up paper states **240K steps at batch size 3,072** — [arXiv 2506.09985](https://arxiv.org/html/2506.09985v1) (snippet); [Rethinking JEPA, arXiv 2509.24317](https://arxiv.org/pdf/2509.24317) (snippet).
  - Estimate [est.]: about 7.7×10^8 clips × about 2,048 tokens × about 6×10^9 FLOP per token gives roughly **10^22 FLOP**. That is small next to frontier LLMs (10^25–10^26) but far from "tiny".
- **AMI Labs by Sept 2026**: apart from the LeWM, benchmark and LeJEPA papers above, I found **no AMI Labs system that addresses language, mathematics or general reasoning** — [TechCrunch, 9 Mar 2026](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/); [Wikipedia: AMI Labs](https://en.wikipedia.org/wiki/Advanced_Machine_Intelligence_Labs) (snippet).

**Generative world models plus agents (Genie 3, SIMA 2, World Labs)**

- **SIMA 2** (Google DeepMind; blog 13 Nov 2025, arXiv 2512.04797 Dec 2025) [preprint and company claim]:
  - A **Gemini-powered** agent. It completes about **65%** of a complex task suite, versus **31% for SIMA 1** and about **71% for humans**.
  - Much higher task completion than SIMA 1 on held-out games (ASKA, MineDojo).
  - Self-improvement: after an initial human-gameplay phase, a separate Gemini proposes tasks and a Gemini reward model scores attempts; trajectories go into a self-generated data bank.
  - This was demonstrated inside **Genie 3-generated worlds**, described as the "first preliminary working example of an agent learning within a universal world model".
  — [DeepMind SIMA 2 blog](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) (snippet); [arXiv 2512.04797](https://arxiv.org/pdf/2512.04797) (snippet); [TechCrunch, 13 Nov 2025](https://techcrunch.com/2025/11/13/googles-sima-2-agent-uses-gemini-to-reason-and-act-in-virtual-worlds/) (snippet)
- **Genie 3** public launch: **29 Jan 2026**. Waymo built a "Waymo World Model" variant to simulate robotaxi edge cases. No "Genie 4" was found as of Sept 2026, and parameters and compute remain undisclosed — [Wikipedia: Genie](https://en.wikipedia.org/wiki/Genie_(world_model)) (snippet).
- A June 2026 article reports "SIMA practices inside Genie 3" for robot training — [TechTimes, 6 Jun 2026](https://www.techtimes.com/articles/317932/20260606/deepmind-world-models-train-robots-imagined-worlds-sima-practices-inside-genie-3-model.htm) (title only; page blocked).
- **World Labs** (Fei-Fei Li):
  - Marble, which generates editable 3D worlds, entered beta in Nov 2025 and launched commercially in Feb 2026.
  - It raised **$1B on 18 Feb 2026**, from AMD, Autodesk ($200M), NVIDIA and others.
  - RTFM, a real-time frame-generation preview, came out in Oct 2025.
  - No agent-training or reasoning results were found.
  — [TechCrunch, 12 Nov 2025](https://techcrunch.com/2025/11/12/fei-fei-lis-world-labs-speeds-up-the-world-model-race-with-marble-its-first-commercial-product/) (snippet); [The AI Insider, 19 Feb 2026](https://theaiinsider.tech/2026/02/19/fei-fei-lis-world-labs-raises-1b-in-fresh-funding-to-advance-development-of-world-models/) (snippet)

### Inferences
- **Data efficiency versus compute efficiency diverge.** Dreamer 4 cuts *action-labelled data* by about 100x, but the absolute capability is weak (0.7% diamonds). The training slice (up to 1,024 TPU v5p) is closer to mid-size LLM training than to "one GPU". The genuinely one-GPU systems (DreamerV3 online, LeWM, BBF) are narrow control or game agents.
- **The best "general" embodied agent is LLM-centred.** SIMA 2 is Gemini inside a Genie 3 world, so the world model supplies a curriculum and training ground, not the intelligence. This is the opposite of the round-1 hypothesis that a compact agent learns its own abstract world model.
- **Compact latent state is the lever for inference-time savings.** LeWM uses 1 token per frame against DINO-WM's hundreds and plans 48x faster. V-JEPA 2-AC similarly beat Cosmos (round-1 notes). Latent compactness is a robust, repeatable 1–2 order-of-magnitude saving at plan time.
- **TD-MPC2's clean log-linear parameter scaling (1M to 317M) is evidence that world-model agents obey scaling laws too.** More capability costs more compute. There is no sign of a discontinuous "small is enough" regime.

### Gaps
- Dreamer 4 training duration and total FLOPs: only the chip count (256–1,024 TPU v5p) was found.
- V-JEPA 2 GPU-hours: not stated in accessible text; only my estimate is given.
- Genie 3 and SIMA 2 compute and parameters: undisclosed.
- Whether LeWM is formally an AMI Labs output: unconfirmed.
- AMI Labs had no model release beyond research code and papers found by Sept 2026.
- DreamerV3 follow-ups from 2025–26 (e.g., "Simulus", [arXiv 2502.11537](https://arxiv.org/pdf/2502.11537)): titles only, no numbers.

---

## 2. Program-based (symbolic/code) world models: WorldCoder, PoE-World, AutumnSynth, TheoryCoder, Meta CWM and ARC-AGI-3 executable-world-model agents. Cost per environment and sample efficiency versus deep RL

### Takeaway
Program world models are the most **sample-efficient** agents measured:
- About **50 interactions** in Sokoban, versus **more than 1M steps** for PPO or DreamerV3.
- **Under 1,000 non-successful demo frames** for Montezuma's Revenge.
- **2,000 actions per game budget** on ARC-AGI-3.

Every strong instance, however, requires three things:
1. A **frontier LLM** as the program proposer.
2. **Object-centric symbolic input**, not pixels.
3. Mostly **deterministic, fully observed** environments.

On ARC-AGI-3's public set, they reach **58–100 RHAE** at roughly **$120–$600 per game** in LLM API cost. Meta's "Code World Model" is a different thing entirely: a standard 32B LLM trained on about 13T tokens, i.e., LLM-scale compute.

### Cited Findings

**WorldCoder** (Tang, Key, Ellis; NeurIPS 2024) [peer-reviewed]:
- The agent writes a Python world model over object-oriented MDP states. It is "more sample-efficient compared to deep RL, more compute-efficient compared to ReAct-style agents", and transfers by editing its code (README).
- Sokoban: basic competence in **about 50 interactions**, while **PPO and DreamerV3 need more than 1,000,000 steps**.
- ReAct, with the same pretrained knowledge, succeeds on only **15% ± 8%** of basic levels.
- Once a model is learned, new tasks need **O(1) LLM calls** because planning is local.
- AlfWorld models exceed 250 lines of code.
— [WorldCoder GitHub](https://github.com/haotang1995/WorldCoder) (README); [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/820c61a0cd419163ccbd2c33b268816e-Paper-Conference.pdf) (snippet)

**PoE-World** (Piriyakulkij, Ellis et al.; NeurIPS 2025) [peer-reviewed]:
- The world model is an exponentially weighted **product of LLM-synthesised programmatic experts**.
- It learns from a **demonstration of fewer than 1,000 frames** per game. The demos are not successful plays; in Montezuma's Revenge the demo never gets reward.
- PoE-World + planner was **the only method to score positive on Montezuma's Revenge**, getting the key in both the original and an altered version.
- It predicts better than WorldCoder, and its Montezuma model runs to **4,000+ lines of code**.
— [arXiv 2505.10819](https://arxiv.org/abs/2505.10819) (snippet); [NeurIPS 2025 PDF](https://papers.neurips.cc/paper_files/paper/2025/file/262dd62fd1bbb30d6a6b4d578f5e65ff-Paper-Conference.pdf) (snippet)
- It requires **object-centric observations via OC_Atari** and the **OpenAI API**. Only Pong, PongAlt, Montezuma and MontezumaAlt are supported. No compute or cost figure is given (README) — [PoE-World GitHub](https://github.com/topwasu/poe-world).

**AutumnSynth / Autumn** (Das, Tenenbaum, Solar-Lezama et al.; POPL 2023) [peer-reviewed]:
- Pure symbolic synthesis, with no neural network, of reactive programs with latent state (functional plus automata synthesis) for Atari-like grid worlds.
- Results: **27/30** benchmark programs and **21/27** third-party grid games synthesised — [ACM DL 10.1145/3571249](https://dl.acm.org/doi/pdf/10.1145/3571249) (snippet); [Basis blog](https://www.basis.ai/blog/autumn/) (snippet).
- In 2025 Basis released **AutumnBench**, with a human baseline, AI comparisons and the "MARA" interactive protocol — [Basis AutumnBench blog](https://www.basis.ai/blog/autumn-platform-2025/) (title and snippet only; page blocked, numbers not obtained).

**TheoryCoder** (TMLR 2025) and **TheoryCoder-2** (arXiv 2602.00929, Feb 2026) [peer-reviewed / preprint]:
- A bilevel world model: general abstractions (e.g., "move to") on top of an LLM-synthesised low-level Python transition model, used with bilevel planning in Baba Is You.
- A world model learned in KekeCompetition **transferred to Baba is AI and solved all 40 levels without additional learning**.
- TheoryCoder-2 learns reusable abstractions and is "significantly more sample-efficient than … WorldCoder" and LLM-planning baselines.
— [arXiv 2503.20124](https://arxiv.org/pdf/2503.20124) (snippet); [arXiv 2602.00929](https://arxiv.org/html/2602.00929) (snippet)

**Meta CWM "Code World Model"** (Meta FAIR; 24–25 Sept 2025) [preprint and company release]:
- A **32B dense** decoder-only LLM with 131k context.
- Training data:
  - **8T tokens** of pretraining.
  - **5T tokens** of mid-training on about **200M Python memory/execution traces** and about **3M agentic trajectories** in **30,000+ containerised environments**.
- Post-training: multi-task RL.
- Results: **SWE-bench Verified 65.8%** (with test-time scaling), **LiveCodeBench 68.6%**, **Math-500 96.6%**, **AIME 2024 76.0%**.
- Quantised inference fits on one 80GB H100.
— [arXiv 2510.02387](https://arxiv.org/abs/2510.02387) (snippet); [MarkTechPost, 25 Sept 2025](https://www.marktechpost.com/2025/09/25/meta-fair-released-code-world-model-cwm-a-32-billion-parameter-open-weights-llm-to-advance-research-on-code-generation-with-world-models/) (snippet); [CWM GitHub](https://github.com/facebookresearch/cwm) (README)
- [est.] 6 × 32×10^9 × 13×10^12 ≈ **2.5×10^24 FLOP**, excluding RL. This is an LLM-scale "world model of code execution", not a low-compute agent.

**ARC-AGI-3 executable-world-model agents (May–Aug 2026), all [self-reported] on the 25 public games unless noted**

- **"Executable World Models for ARC-AGI-3 in the Era of Coding Agents"** (arXiv 2605.05138, May 2026; also a Springer chapter):
  - Method: the agent maintains an executable Python world model, verifies it against past observations, refactors toward simpler abstractions as an MDL-like proxy, and plans through the model. It has no game-specific code.
  - **GPT-5.5 (high): 15 games fully solved, mean RHAE 58.12%.**
  - **GPT-5.4 (high): 8 games, 41.29%.**
  — [arXiv 2605.05138](https://arxiv.org/html/2605.05138v1) (snippet)
- **Ablation follow-up, "Do Coding Agents Need Executable World Models, Simplification, and Verification…?"** (arXiv 2607.15439, Jul 2026):
  - Setup: four nested Codex agents, ranging from a textual baseline to an executable model with simplification and exact-replay verification.
  - "Gains from model capability and reasoning effort often exceed variant differences."
  - At max effort, "the three imposed mechanisms are not required for action-efficient public-set completion". Verification "nevertheless scores higher and succeeds at lower effort".
  — [arXiv 2607.15439](https://arxiv.org/abs/2607.15439) (snippet)
- **OPINE-World** (arXiv 2607.01531, Jul 2026):
  - Two cooperating **Claude Opus 4.8** agents (actor and model-synthesiser) with counterexample-guided synthesis, exact-replay verification, model-based planning and "ontology-error"-guided exploration.
  - **20/25 games, 160/183 levels, RHAE 78.4** (92.1 claimed if five rate-limited games had finished), within a **2,000-action budget per game**.
  - The authors claim that on ARC-AGI-3 "program-synthesis and neural latent world models solve none".
  - Future work: stochastic and partially observed environments, and raw-pixel perception.
  — [arXiv 2607.01531](https://arxiv.org/abs/2607.01531) (snippet); [OPINE-World GitHub](https://github.com/david-courtis/opine-world) (README)
- **Cost conflict on OPINE-World**:
  - The authors report about **$800 total** (four Claude Max subscriptions at $200 per month) (README).
  - Tycho's authors estimate OPINE-World at **$12.4–15.2k** at API rates — [Tycho, arXiv 2607.28287](https://arxiv.org/pdf/2607.28287) (snippet).
- **Tycho** (Lehmann et al., NIMI; arXiv 2607.28287, Jul 2026):
  - Games are modelled as rendered deterministic Moore machines, which the coding agent builds, tests and repairs.
  - Public-set RHAE: **Opus 4.8 88.49; Opus 5 100; GPT-5.6 Sol 100** (README).
  - Cost: about **$5.78k** for the Opus 4.8 comparison run and about **$2.99k** for the Opus 5 run (snippet). That is roughly **$120–230 per game** [est.].
  — [Tycho GitHub](https://github.com/NIMI-research/Tycho); [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- Other program-world-model papers seen by title only:
  - Mind-Studio, executable world models for partially observable games ([arXiv 2606.16070](https://arxiv.org/pdf/2606.16070)).
  - PatchWorld, gradient-free optimisation of executable world models ([arXiv 2605.30880](https://arxiv.org/pdf/2605.30880)).
  - Object-Centric Environment Modeling for agentic tasks ([arXiv 2607.02846](https://arxiv.org/pdf/2607.02846)).

### Inferences
- **Sample-efficiency ratios.** Program world models beat deep RL by about **4 orders of magnitude** in environment interactions (about 50 against more than 10^6 in Sokoban) and solve problems where model-free and latent-model RL score 0 (Montezuma from under 1K non-rewarding frames).
- **The cost moves into LLM inference.** It is about $100–600 per ARC-AGI-3 game at frontier API prices. Behind that is the proposer's pretraining (10^24–10^26 FLOP), which the "full-pipeline" ledger must count.
- **The ablation paper cuts against "structure substitutes for scale".** On ARC-AGI-3, base-model capability and reasoning effort matter more than the world-model scaffolding. The scaffolding mainly lowers the *effort* needed to reach a given score.
- **Pure symbolic synthesis without an LLM (AutumnSynth) works only inside a hand-designed DSL of grid games.** OPINE-World reports that non-LLM program synthesis and latent world models "solve none" of ARC-AGI-3. This is the sharpest evidence that the "DL guidance" in "DL-guided program synthesis" currently has to be a frontier-scale model.
- All the strong program-world-model results assume **object-centric, symbolic state** (OC_Atari, OO-MDP, grid cells) and deterministic dynamics. The perception problem that latent models solve is simply skipped. **No system in this evidence set combines a learned latent perception model with a synthesised program world model end-to-end.**

### Gaps
- WorldCoder and PoE-World per-environment LLM token or dollar cost: not found (READMEs silent; papers blocked).
- AutumnBench human-versus-AI numbers: page blocked.
- The identity of the baselines behind OPINE-World's "solve none" claim (which latent world model? AutumnSynth? PoE-World?) was not in the snippet or README.
- No independent (ARC Prize-verified, semi-private) score was found for Tycho, OPINE-World or the Executable-World-Models agent. All are public-set self-reports.

---

## 3. ARC-AGI-3 compute-limited Kaggle track (2026): status, top entrants' approaches, and what it shows about low-compute world-model induction

### Takeaway
Under Kaggle's constraints the best open solutions score **about 1–3%**, against **58–100%** for the same kind of program-world-model harness driven by frontier API models on the public set, and **62.7%** for GPT-6 Astra on the semi-private set.

Kaggle's constraints are:
- one GPU;
- no internet;
- a 12-hour notebook.

The Milestone #1 winner was a lightweight harness around **Qwen 3.6 27B** that "encodes the game as a programming problem". Its reported scores are **1.21%** (Tufa's X post, set unspecified) and **about 1.6** as a mean self-measured on public games. Milestone #2 closes **30 Sept 2026**, and no results were available at the time of writing.

### Cited Findings
- **Kaggle constraints:**
  - Notebooks must run in under **12 hours** with **no internet**, so there are no GPT, Claude or Gemini API calls.
  - The **RTX 6000** accelerator is reserved for ARC-AGI-3 notebooks.
  - All prize-eligible solutions must be open-sourced.
  — [Kaggle ARC-AGI-3 rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules) (snippet); [ARC Prize 2026 docs](https://docs.arcprize.org/arc-prize-2026) (snippet)
  - A $10,000 runtime-cost cap appears in the same snippet. It is probably the ARC Prize verified-leaderboard policy, not a Kaggle rule — [ARC Prize policy](https://arcprize.org/policy) (snippet; ambiguous).
- **Milestone #1 winner, Tufa Labs "Duck harness"** (open-sourced July 2026):
  - "Built on Qwen 3.6 27B … We hit 1.21% with our lightweight harness" — [Tufa Labs on X](https://x.com/tufalabs/status/2072336849465417747) (snippet).
  - Like other solutions it "encode[s] the game as a programming problem" to use the LLM's coding ability. "Because evaluation on the semi-private/private test set on Kaggle is constrained to a single GPU, [they] are restricted to use small open-source models and small token throughput" — [Tufa Labs research page](https://tufalabs.ai/research/duck-harness/) (snippet); [duck-harness GitHub](https://github.com/Tufalabs/duck-harness) (README: "tool-using solver" on the TAAF framework).
  - Score note: the "about 1.6 mean score" is self-measured on public games; the 1.21% is from Tufa's X post, set unspecified. See the conflict note below.
- **Milestone #1, 2nd place, "Reki"**: renders frames to images, feeds them to a **locally served Gemma-4-31B**, keeps a reflection memory, and emits structured JSON actions with repair and legal-action guards. No explicit world-model synthesis is described — [ARC Prize Milestone #1 blog](https://arcprize.org/blog/arc-prize-2026-milestone-1) (snippet).
- **Open-weight models outside the Kaggle limits:**
  - "Polyphony Agent" reached **19.8%** on the *community* public leaderboard with self-hosted Qwen 3.6.
  - Closed models in specialised harnesses reach 95–100 RHAE on the public set: Tycho 100.0, Retrodict 99.86, NVIDIA AVO 100.
  - An open project (arc3cb) is porting the "retrodiction-first" executable-`step(state, action)`-simulator harness to gpt-oss-120b and Gemma-4-31B via Cerebras, at an estimated **$25/game** for Gemma-4-31B. No scores yet.
  — [avo-qwen-arcagi3 GitHub](https://github.com/criticaldata/avo-qwen-arcagi3) (README)
- **Late-Sept 2026 participant datapoint (weak source):** one competitor's Kaggle notebook (RTX Pro 6000, forked from a public "B81" notebook) logged a first hidden-set score of **3.41** on 25 Sept 2026. Smoke tests cleared 3/7, 1/7 and 1/9 levels on three games — [GitHub PR, sehajvir-singh](https://github.com/sehajvir-singh/ml-from-scratch/pull/1) (README/PR text; individual, unverified).
- **Milestone #2** closes 30 Sept 2026, with prizes of $25K, $10K and $2.5K. No results yet — [ARC Prize 2026 ARC-AGI-3](https://arcprize.org/competitions/2026/arc-agi-3) (snippet).
- **Verified frontier leaderboard, 24 Sept 2026 (aggregator):** GPT-6 Astra 62.7%, Claude Opus 5 30.2%, Gemini 3.8 Flash 10.4% — [BenchLM ARC-AGI-3](https://benchlm.ai/benchmarks/arcagi3) (snippet); [llm-stats](https://llm-stats.com/benchmarks/arc-agi-3) (snippet).

### Inferences
- **About a 20–50x score gap.** Under realistic low compute (one GPU for 12 h, a 27–31B open model), the *same* approach family scores about 1–3%. With frontier APIs costing about $3–26K per run, it scores 60–100%. The world-model-induction *structure* transfers, but the *proposal prior* does not fit on one GPU. That makes the gap a direct measure of how far the "compact agent" in the round-1 ranking is from reality today.
- Tufa's lesson that "hand-crafted tools hurt, letting it improvise worked better" (round-1 notes) and the ablation in §2 point the same way: extra structure helps less than a stronger base model.
- Nothing non-LLM appears among the Kaggle leaders. The 2025 preview's small CNN+RL winner (StochasticGoose, 12.58% on preview games) has not been carried forward as a competitive 2026 approach, going by the sources found.

### Gaps
- The Kaggle public-leaderboard top score as of late Sept 2026 was not retrievable: the Kaggle, Medium and arcprize pages were blocked and the search budget ran out.
- Score conflict for Tufa: 1.21% (X post) versus about 1.6 "mean score across all public games". These are likely different sets (hidden versus public) and possibly different scales. Unresolved.
- Third-place (Md Boktiar Mahbub Murad) approach: not found.

---

## 4. Hybrids: LLM as proposal prior plus world-model verification. Does a world model measurably reduce search or inference compute?

### Takeaway
Yes, but the savings are modest and conditional. Measured effects:
- Fewer LLM calls once a model is learned: WorldCoder needs O(1) calls per new task, against a per-step ReAct loop.
- Faster planning with compact latents: 48x for LeWM, about 15x for V-JEPA 2-AC versus Cosmos.
- Lower reasoning effort for the same ARC-AGI-3 success when verification is used.
- Large reliability gains from small learned transition models gating LLM planners: hallucinated-state rate cut about 2.5–5x.

No clean "exchange rate" between world-model quality and search was found for 2025–26 LLM agents. The best cross-system comparison (Tycho versus OPINE-World) shows about a **2–2.6x cost difference** from harness design at similar or better scores.

### Cited Findings
- **Hybrid-WM** (arXiv 2606.27806, Jun/Jul 2026) [preprint]:
  - The LLM stays the planner. A **small parametric transition model** predicts validity, state deltas, risk and value, and a consistency gate triggers revision only on disagreement.
  - Overall success **0.668 → 0.838**; long-horizon success **0.471 → 0.758**; hallucinated-state rate **0.205 → 0.079**.
  - On real GPT-4o-mini episodes, the hallucinated-state rate fell **0.176 → 0.035**.
  — [arXiv 2606.27806](https://arxiv.org/abs/2606.27806) (snippet)
- **WALL-E 2.0** (NeurIPS 2025) [peer-reviewed]:
  - Training-free. It learns symbolic action rules, knowledge graphs and scene graphs as code that regulates an LLM agent, with MPC in which the LLM is the look-ahead optimiser.
  - Mars (Minecraft-like): reward **+16.1% to +51.6%**, task score **+61.7% or more**.
  - ALFWorld: **98% success after only 4 iterations**.
  — [arXiv 2504.15785](https://arxiv.org/abs/2504.15785v1) (snippet); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119191); [code](https://github.com/elated-sawyer/WALL-E)
- **WorldCoder**: once the code model explains the data, the agent "can solve new tasks with O(1) LLM calls by planning locally". It is "more compute-efficient than ReAct-style agents" — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/820c61a0cd419163ccbd2c33b268816e-Paper-Conference.pdf) (snippet); [GitHub](https://github.com/haotang1995/WorldCoder).
- **ARC-AGI-3 ablation**: verification "scores higher and succeeds at lower effort", but model capability and effort dominate variant differences — [arXiv 2607.15439](https://arxiv.org/abs/2607.15439) (snippet).
- **Harness-level cost differences at fixed task (ARC-AGI-3 public, Opus-class models):** Tycho about $5.78k against OPINE-World about $12.4–15.2k at API rates, with Tycho scoring higher (88.49 versus 78.4 RHAE) — [Tycho arXiv 2607.28287](https://arxiv.org/pdf/2607.28287) (snippet).
- **Latent planning cost:** LeWM plans in about 1 s versus 47 s for DINO-WM (48x) by using about 200x fewer tokens per frame — [arXiv 2603.19312](https://arxiv.org/html/2603.19312v1) (snippet). V-JEPA 2-AC versus Cosmos (16 s versus about 4 min) is in the round-1 notes.
- **Classic train-versus-test compute trade-off:** Jones (2021) showed with AlphaZero on Hex that "test-time and train-time compute … can be traded off while maintaining performance". Perfect play on 9×9 Hex takes a little under 3 hours on one RTX 2080 Ti — [arXiv 2104.03113](https://arxiv.org/abs/2104.03113) (snippet).

### Inferences
- The best-supported compute benefit of a world model in LLM agents is **amortisation**. After an up-front synthesis cost (hundreds of dollars per ARC-AGI-3 game), acting is cheap and needs no per-step LLM calls. That favours long-lived agents in stable environments and helps little for one-shot open-ended tasks.
- A **small learned transition model can audit a big LLM planner** and roughly halve to fifth its state-hallucination errors (Hybrid-WM). That is a reliability gain, not a capability gain, and the LLM still does the semantic reasoning.
- None of this shows a world model *replacing* a large proposal prior. The measured exchange rates are about 2–50x at inference time, far short of the 10^3–10^5x needed to bring frontier-prior costs down to a compact agent.

### Gaps
- The specific Jones (2021) exchange-rate figure (roughly "N× less test-time compute per 10× train-time compute") could not be verified because arXiv was blocked. I have not quoted a number.
- RAP ("Reasoning via Planning", Hao et al. 2023), which pairs an LLM world model with MCTS: its numbers could not be re-verified in this session, so they are omitted.
- No 2025–26 paper was found that systematically varies world-model accuracy and measures the search budget needed, for LLM agents.

---

## 5. Language and abstract reasoning: can latent-predictive (non-autoregressive) world models match LLMs on language or math per FLOP, and do program-world-model agents handle open-ended language tasks?

### Takeaway
**No. As of Sept 2026 there is no evidence that a latent-predictive / JEPA-style model matches LLMs on language or math per FLOP at scale.** All JEPA-for-language results are **auxiliary losses on top of an autoregressive LLM, applied during fine-tuning**:
- LLM-JEPA: up to +14 points on narrow fine-tuning tasks, at about 1.5–2x step cost.
- STP: matches full-data fine-tuning with 16x less data.

Standalone latent language models underperform:
- Meta's Large Concept Model loses to a same-size Llama on instruction following.
- CALM's 44% FLOP saving is at 281–371M scale and is still autoregressive (next-vector).

Program-world-model agents have been shown only on games, grids and household-text simulators, always with an LLM doing the language.

### Cited Findings
- **LLM-JEPA** (Huang, LeCun, Balestriero; arXiv 2509.14252 Sept 2025; ICLR 2026) [peer-reviewed]:
  - Adds a JEPA embedding-prediction loss between paired "views" (e.g., text and code) to standard LLM training.
  - Reported "outperform[s] the standard LLM training objectives by a significant margin" on NL-RX, GSM8K, Spider and RottenTomatoes across Llama3, OpenELM, Gemma2 and OLMo.
  - Example: **Llama-3.2-1B-Instruct on NL-RX-SYNTH, 71.46% versus 57.29%** with standard fine-tuning.
  - Cost: needs an extra forward pass, **about 1.5–2x per-step training cost**. The "loss dropout" trick reduces this.
  — [arXiv 2509.14252](https://arxiv.org/abs/2509.14252) (snippet); [ML Anthology ICLR 2026](https://mlanthology.org/iclr/2026/huang2026iclr-llmjepa/); [Liner review](https://liner.com/review/llmjepa-large-language-models-meet-joint-embedding-predictive-architectures) (snippet, cost figure)
- **Semantic Tube Prediction, STP** (arXiv 2602.22617; ICML 2026) [peer-reviewed]:
  - "Geodesic hypothesis": hidden-state trajectories are held to a tube around a locally linear path, so the JEPA predictor becomes the identity. Overhead is negligible and nothing is added at inference.
  - It **matches full-dataset fine-tuning accuracy with 16x less training data**.
  - The snippets describe fine-tuning; no pretraining-scale result was found — [arXiv 2602.22617](https://arxiv.org/abs/2602.22617) (snippet); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64567).
- **VL-JEPA** (Meta; arXiv 2512.10942, Dec 2025) [preprint]:
  - Predicts continuous embeddings of target text instead of generating tokens.
  - At **1.6B parameters** it is comparable to InstructBLIP and QwenVL on four VQA datasets.
  - It has **50% fewer trainable parameters** than token-space VLM training with the same encoder and data.
  - **Selective decoding cuts decoding operations 2.85x**.
  - It beats CLIP, SigLIP2 and Perception Encoder on the averages over 8 video-classification and 8 retrieval sets.
  - These are perception and VQA tasks, not open-ended language or math — [arXiv 2512.10942](https://arxiv.org/abs/2512.10942) (snippet).
- **Large Concept Models** (Meta, Dec 2024) [preprint]:
  - Sentence-level "concept" prediction in SONAR space, at 1.6B parameters / 1.3T tokens and 7B / about 7.7T tokens.
  - LCMs "generate less fluent summaries than LLMs".
  - At 1.6B, a same-size "SmaLLaMA" LLM performs better on instruction following.
  — [Meta AI publication](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/) (snippet); [LessWrong distillation](https://www.lesswrong.com/posts/7Dtyhdkp5m6p4mquC/distillation-of-meta-s-large-concept-models-paper) (snippet)
- **CALM, Continuous Autoregressive LMs** (Shao et al., arXiv 2510.27688, Oct 2025) [preprint]:
  - An autoencoder compresses K tokens into one vector, and the model predicts the next vector.
  - **371M CALM-M matches a 281M Transformer-S on BrierLM with 44% fewer training FLOPs and 34% fewer inference FLOPs.**
  - This is small scale, and it is still autoregressive generation — [arXiv 2510.27688](https://arxiv.org/pdf/2510.27688) (snippet); [CALM GitHub](https://github.com/shaochenze/calm).
- **DLLM-JEPA** (arXiv 2606.00091, Jun 2026): a JEPA loss for masked-diffusion LMs, seen by title only — [awesomepapers](https://awesomepapers.io/generative-models/papers/2606.00091).
- **Program-world-model agents and language:**
  - The closest to open-ended language are WALL-E 2.0 (ALFWorld text household, 98%) and SIMA 2 (natural-language instructions in 3D games). Both put the **LLM (or Gemini) in charge of language**, with the world model as a constraint or training ground.
  - The Meta CWM "world model" gets strong math (Math-500 96.6%, AIME24 76.0%), but it is a conventional 32B autoregressive LLM at about 13T tokens.
  — [WALL-E 2.0](https://arxiv.org/abs/2504.15785v1) (snippet); [SIMA 2 blog](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) (snippet); [CWM](https://arxiv.org/abs/2510.02387) (snippet)

### Inferences
- The JEPA-for-language evidence supports **"embedding-space objectives are a useful regulariser and data-efficiency multiplier (up to 16x) for LLM fine-tuning"**. It does not support "a non-autoregressive world model can replace the LLM". Every language result keeps a token-level decoder.
- Standalone latent language models (LCM, and CALM as a partial case) show that **moving prediction to a latent space costs fluency and precision**. The only FLOP saving at matched quality is 44%, at sub-billion scale. That is within the range of ordinary architecture tweaks, not an order-of-magnitude route.
- For the round-1 hypothesis, the "language/math" part of a compact world-model agent currently has **no demonstrated non-LLM mechanism**. It would have to be either (a) an LLM, which imports its pretraining FLOPs, or (b) program synthesis over language, which has not been shown for open-ended text.

### Gaps
- No pretraining-scale (≥1B parameters, ≥100B tokens) head-to-head of a JEPA or latent LM against an autoregressive LM on language or math benchmarks per FLOP was found.
- STP's exact models, datasets and whether it was tested in pretraining: not confirmed (paper blocked).
- LLM-JEPA pretraining results: the abstract mentions them, but numbers were not obtained.

---

## 6. Bottom line: how general and how cheap are world-model agents, and what is the biggest demonstrated gap to LLM-based agents?

### Takeaway
The newest evidence **supports world models and program synthesis as the right inference-time structure**. Every top ARC-AGI-3 system synthesises an executable world model, and latent models cut data and planning cost by 1–4 orders of magnitude in their domains.

It **undercuts the "compact / low-compute" part** of the round-1 ranking. The biggest demonstrated gap is this:
- Under a genuinely low compute budget (one GPU, 12 h, a 27–31B open model), world-model-induction agents score **about 1–3% on ARC-AGI-3**.
- Driven by frontier LLMs, the same approach scores **58–100%** (public set) and 62.7% (Astra, semi-private).
- Non-LLM latent or program-synthesis world models reportedly score **0**.
- No world-model agent without a large LLM has shown competence on language, math or open-ended tasks.

### Cited Findings
- Across ARC-AGI-3 program-world-model agents, **base-model capability and reasoning effort dominate the gains from world-model scaffolding** — [arXiv 2607.15439](https://arxiv.org/abs/2607.15439) (snippet).
- The Kaggle single-GPU winner reached 1.21% with Qwen 3.6 27B — [Tufa Labs on X](https://x.com/tufalabs/status/2072336849465417747) (snippet). Frontier-harness public-set scores are 58–100 RHAE — [Executable World Models](https://arxiv.org/html/2605.05138v1) (snippet); [Tycho GitHub](https://github.com/NIMI-research/Tycho); [OPINE-World GitHub](https://github.com/david-courtis/opine-world).
- The claim that "program-synthesis and neural latent world models solve none" of ARC-AGI-3 — [OPINE-World, arXiv 2607.01531](https://arxiv.org/abs/2607.01531) (snippet).
- Dreamer 4, the most capable offline latent-world-model agent: 0.7% diamonds on 256–1,024 TPU v5p — [arXiv 2509.24527](https://arxiv.org/abs/2509.24527) (snippet); [EmergentMind](https://www.emergentmind.com/papers/2509.24527) (snippet).
- SIMA 2's generality comes from Gemini, with Genie 3 as a training world — [DeepMind](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) (snippet).
- The JEPA language results are fine-tuning add-ons — [LLM-JEPA](https://arxiv.org/abs/2509.14252); [STP](https://arxiv.org/abs/2602.22617) (snippets).

**Summary table: scope, compute and the key limit per system**

| System (date, status) | Domain | Params / proposer | Full-pipeline compute | Data or sample efficiency | Key limit |
|---|---|---|---|---|---|
| Dreamer 4 (Sep 2025, preprint) | Minecraft, offline | ~2B | 256–1,024 TPU v5p; ~10^21–10^22 FLOP [est.] | ~100x less data than VPT | 0.7% diamonds |
| TWM on Craftax (ICML 2025) | 2D open-world game | small | not found | Beats human (69.66% vs 65.0%) at 1M steps | Single game family |
| TD-MPC2 (ICLR 2024) | 80 control tasks | 317M | not found | Log-linear scaling from 1M to 317M | Control only |
| BBF (ICML 2023) | Atari 100k | model-free reference | ~10 A100-h per game | Superhuman IQM at 100k steps | Narrow |
| LeWM (Mar 2026, preprint) | 2D/3D control | 15M | one GPU, a few hours | Planning 48x faster than DINO-WM | Toy control |
| V-JEPA 2 / 2-AC (Jun 2025) | Video, robot arm | ~1B + 300M | ~10^22 FLOP [est.] | 62 h of robot video | No language generation |
| SIMA 2 + Genie 3 (Nov 2025) | 3D games | Gemini | undisclosed | Self-improves in generated worlds | LLM-centred |
| WorldCoder (NeurIPS 2024) | Sokoban, MiniGrid, AlfWorld | GPT-4-class LLM | LLM API cost (not found) | ~50 interactions vs >1M for PPO/DreamerV3 | Symbolic state input |
| PoE-World (NeurIPS 2025) | Atari (Pong, Montezuma) | OpenAI LLM | not found | <1K non-rewarding frames | Object-centric input |
| Meta CWM (Sep 2025) | Code, SWE, math | 32B LLM | ~2.5×10^24 FLOP [est.] | – | It is an LLM |
| Tycho / OPINE / ExecWM (May–Jul 2026, self-reported) | ARC-AGI-3 public set | Opus 4.8–5, GPT-5.5/5.6 | ~$120–600 per game in API cost + frontier pretraining | 2,000-action budget per game | Needs a frontier proposer |
| Duck harness (Jul 2026, Kaggle) | ARC-AGI-3 hidden set | Qwen 3.6 27B | 1 GPU, ≤12 h | – | ~1.2–1.6% |

### Inferences
- **Generality:**
  - Latent world models generalise *within* perception and control (many tasks per model: TD-MPC2's 80, SIMA 2's held-out games).
  - Program world models generalise *across* grid and game mechanics, and transfer by code editing (TheoryCoder: 40/40 levels zero-shot).
  - Neither class, on its own, has crossed into language, math or open-ended reasoning.
- **Cost:**
  - The "cheap" parts are the environment interactions and the planning step.
  - The expensive part is the **prior**. For latent models that means video pretraining, about 10^22 FLOP. For program models it means the LLM proposer, 10^24–10^26 FLOP of pretraining plus about $100s per environment in inference.
  - A compact agent that must supply its own proposal prior has **no demonstration** beyond about 1–3% on ARC-AGI-3 with a 27B model.
- **Implication for the round-1 ranking:** the direction still looks like the right *architecture* for skill-acquisition efficiency. The evidence now suggests its near-term form is "frontier LLM + synthesised world model", not "compact agent".
  - The low-compute version depends on an unsolved sub-problem: a **cheap, general proposal prior for programs**, including for language.
  - Signals worth tracking:
    - Kaggle ARC-AGI-3 Milestone #2 (30 Sept 2026) and the final 2026 results.
    - Any JEPA or latent language model at ≥1B parameters from pretraining, compared per FLOP.
    - Any end-to-end system combining learned latent perception with program synthesis, working on raw pixels.

### Gaps
- There is no apples-to-apples full-pipeline FLOP accounting (pretraining + synthesis + acting) for any ARC-AGI-3 world-model agent. Frontier model sizes and training FLOPs are undisclosed.
- ARC Prize-verified semi-private scores for program-world-model harnesses using open-weight models are unavailable.
- Whether the 20–50x Kaggle-versus-frontier gap is mostly model capability or mostly the token-throughput limit (12 h on one GPU) was not separated in any source found.
