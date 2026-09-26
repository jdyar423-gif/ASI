# Low-Compute Approaches to General Reasoning and Abstraction: Tiny Recursive/Latent-Reasoning Models, Program Synthesis, Neurosymbolic Methods, Test-Time Adaptation (ARC-AGI as the key benchmark). Status as of Sept 2026

Research notes compiled 2026-09-26. Access caveat: arcprize.org, arxiv.org, huggingface.co, several Substack blogs, poetiq.ai, pathway.com, tufalabs.ai and developer.nvidia.com were blocked by the session's egress proxy. Many figures below therefore come from search-engine snippets of those primary pages (cited by the primary URL), from ARC Prize's own X posts, or from GitHub READMEs, which I fetched directly. Where a number comes only from an aggregator or blog, I say so. "Verified" means ARC Prize tested it on a hidden set (semi-private or Kaggle private). "Self-reported" means the authors ran it on the public evaluation set.

## Key Question 1: ARC-AGI-1/2/3 status as of 2026: top scores, cost per task, ARC Prize 2024/2025 winners and methods, and frontier LLMs vs small specialized systems

### Takeaway
By Sept 2026, frontier LLMs have almost saturated ARC-AGI-1 and ARC-AGI-2 at very low inference cost per task. The verified figures for Claude Opus 5.5 are 98.5% at $0.16 per task on ARC-AGI-1 and 93.3% at $0.41 per task on ARC-AGI-2. Compute-capped open systems trail far behind: the Kaggle 2025 winner scored 24% on ARC-AGI-2 at about $0.20 per task, using a 4B model with test-time training. ARC-AGI-3, the interactive version launched in March 2026, went from under 1% for frontier models at launch to 62.7% verified (GPT-6 Astra, Sept 2026, about $26K total run cost). The top ARC-AGI-3 systems are frontier LLMs inside harnesses that synthesize programmatic world models. Small models do not lead there. The low-compute advantage of specialized systems now lies almost entirely in **training** cost and parameter count. On **inference $/task**, frontier models caught up during 2026.

### Cited Findings

**ARC-AGI-1 (static grid puzzles, 2019)**
- Dec 2024: OpenAI o3-preview scored 75.7% on the ARC-AGI-1 Semi-Private eval within the $10k public-leaderboard compute limit. A high-compute configuration using 172× the compute scored 87.5%, which was not eligible for the leaderboard. OpenAI said the tested o3 had been trained on 75% of the public training set. — [ARC Prize on X](https://x.com/arcprize/status/1870169260850573333); [ARC Prize o3 blog](https://arcprize.org/blog/oai-o3-pub-breakthrough)
- Sources disagree on o3-preview's cost per task. One snippet says about $20/task for low compute ([ARC Prize o3 blog](https://arcprize.org/blog/oai-o3-pub-breakthrough), via search snippet). Eric Pang says "o3-preview scored 75.7% … while spending $200/task on low setting" ([Eric Pang on X](https://x.com/_eric_pang_/status/1968009342508208589)). An R40 headline says "The Same 87.5% on ARC-AGI-1 Cost $4,560 in 2024 and 30 Cents in 2026" ([R40](https://r40.io/blog/arc-agi-verified-cost-per-task-price-step/), headline only because the page was blocked). The gap likely reflects ARC Prize's later re-pricing of o3-preview, but I could not confirm this.
- 2026, verified: Claude Opus 5.5 scored ARC-AGI-1 **98.5% at $0.16/task** and ARC-AGI-2 **93.3% at $0.41/task**. ARC Prize reports that it "outscored Opus 5 by 2.9 pp on v2 and 1.0 on v1, at roughly 80% lower evaluation cost." — [ARC Prize on X](https://x.com/arcprize/status/2102512140405866568)
- Aggregator only, not verified in this session: DeepSeek V4 Flash at 89.0% on ARC-AGI-1 for $0.02/task — [BenchLM](https://benchlm.ai/benchmarks/arcagi1)
- Living survey (Mar 2026): program synthesis, neuro-symbolic and neural approaches **all drop 2–3× from ARC-AGI-1 to ARC-AGI-2**, which the authors read as a limit on compositional generalization. — [ARC living survey, arXiv 2603.13372](https://arxiv.org/html/2603.13372v1)

**ARC Prize 2024 (ARC-AGI-1 private set, Kaggle compute limits)**
- The ARChitects won first place with 53.5% using test-time training (TTT). MindsAI scored 55.5% but could not win a prize because it did not open-source its solution. The private-set SOTA rose from 33% to 55.5%, driven by "deep learning-guided program synthesis and test-time training." MindsAI pioneered TTT for ARC in 2023. — [ARC Prize 2024 Technical Report](https://arxiv.org/html/2412.04604v1); [ARC Prize 2024 winners](https://arcprize.org/blog/arc-prize-2024-winners-technical-report)

**ARC-AGI-2 (2025) and ARC Prize 2025**
- ARC Prize 2025 ran from Mar 26 to Nov 3, 2025, with 1,455 teams, 15,154 entries and 90 papers (47 in 2024). The Grand Prize (85% on the private set within Kaggle limits) went unclaimed. — [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis); [ARC Prize 2025 Technical Report, arXiv 2601.10904](https://arxiv.org/html/2601.10904v1)
- Kaggle private-set winners (ARC-AGI-2):
  1. **NVARC** (Ivan Sorokin, Jean-François Puget, NVIDIA): **24.03%**. It builds on the 2024 ARChitects TTT pipeline and relies heavily on synthetic data.
  2. **The ARChitects**: **16.53%** with a "2D-aware masked-diffusion LM with recursive self-refinement and perspective-based scoring."
  3. **MindsAI**: **12.64%** with a "heavily-engineered test-time-training pipeline."

  Paper Prize 1st place: **TRM (Jolicoeur-Martineau)**. — [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- NVARC details:
  - Base model: Qwen3 4B, fine-tuned with LoRA.
  - Synthetic data: 103,000 synthetic puzzles plus 3.2M augmented puzzles.
  - The repo also contains an improved TRM variant, but its contribution to the score is not quantified. — [NVARC GitHub](https://github.com/1ytic/NVARC)
  - Compute: "maximum 12 hours on four small GPUs with a budget of approximately 20 cents per task". Public-LB score 27.64%.
  - Puzzle summaries used to seed the synthetic data were generated partly with code and partly with GPT-4o. — [NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/), via search snippet
- Kaggle ARC-AGI-2 limits: about $50 per submission, no internet, L4×4 machines with 96GB of GPU memory. The Grand Prize requires 85% on the private set within these limits. — [bracai.eu](https://www.bracai.eu/post/arc-agi-2-benchmark); [ARC Prize 2026 ARC-AGI-2 competition](https://arcprize.org/competitions/2026/arc-agi-2) (via snippets)
- Verified frontier results, late 2025:
  - Poetiq's refinement harness on Gemini 3 Pro: **54% at $30.57/task**, up from a Gemini 3 Pro baseline of 31% at $0.81/task.
  - Claude Opus 4.5 (Thinking, 64k): **37.6% at $2.20/task**.
  - Gemini 3 Pro used 96 reasoning tokens on task #4cd1b7b2, while Gemini 3 Deep Think used 138,000. — [bdtechtalks](https://bdtechtalks.substack.com/p/poetiq-crushed-arc-agi-2-at-half); [Poetiq](https://poetiq.ai/posts/arcagi_verified/) (via snippets)
- Gemini 3 Deep Think is reported at **84.6% on ARC-AGI-2 at $13.62/task** — [officechai](https://officechai.com/ai/gemini-3-deep-think-benchmarks-arc-agi/), via snippet. This appears to be a later (2026) Deep Think version. Poetiq's "half the cost of Gemini 3 Deep Think" comparison referred to an earlier Deep Think run. One aggregator attributes "84.6% at $0.25/task" to "Gemini 3.7 Flash" ([localaimaster](https://localaimaster.com/blog/arc-agi-benchmark-explained)), which looks like an aggregator error or conflation. Treat both as unconfirmed.
- Aggregators, June 2026: GPT-5.5 leads the ARC-AGI-2 public leaderboard at about 85%, ahead of GPT-5.4 Pro (~83%) and Gemini 3.1 Pro (~77%). DeepSeek V4 Flash 0731 scores 61.4% at $0.04/task. — [bracai.eu](https://www.bracai.eu/post/arc-agi-2-benchmark); [localaimaster](https://localaimaster.com/blog/arc-agi-benchmark-explained)
- The Jeremy Berman, Eric Pang and TRM results are covered under Key Questions 2 and 3.

**ARC-AGI-3 (interactive games, launched Mar 25, 2026)**
- Format: the first fully interactive ARC benchmark, with "hundreds of handcrafted games with thousands of levels." Humans solved every environment. At launch every frontier model scored under 1%: GPT-5.4, Claude Opus 4.6 and Grok 4.2 scored between 0% and 0.37%. — [ARC-AGI-3 launch](https://arcprize.org/blog/arc-agi-3-launch); [DataCamp](https://www.datacamp.com/blog/arc-agi-3); [ARC-AGI-3 paper, arXiv 2603.24621](https://arxiv.org/pdf/2603.24621)
- Scoring uses **RHAE (Relative Human Action Efficiency)**:
  - Per level: min(human_actions / agent_actions, 1), squared.
  - Per game: the average of level scores, weighted by level index.
  - The human baseline is the upper-median (fewest actions) first-time human player.
  - There are 25 public games. A private set of 110 games is split in half between the public and private Kaggle leaderboards. — [ARC-AGI-3 methodology docs](https://docs.arcprize.org/methodology); [Kaggle data page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data)
- Prizes: $850K in total, including a **$700K Grand Prize for the first agent to reach 100% on the fully private set**. — [ARC Prize 2026 ARC-AGI-3](https://arcprize.org/competitions/2026/arc-agi-3)
- Preview competition (Jul 18 – Aug 19, 2025): **StochasticGoose (Tufa Labs) won with 12.58%** (18 levels). It used a small CNN with simple RL to predict which actions change the 64×64 frame, then biased exploration toward those actions. — [ARC-AGI-3 Preview 30-day learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- Spring 2026, verified: **GPT-5.5 0.43%, Opus 4.7 0.18%**. Failure modes found: "true local effect, false world model", "wrong level of abstraction from training data", and "solved the level, didn't reinforce the reward". — [ARC Prize on X](https://x.com/arcprize/status/2050261221165989969)
- Jul 24, 2026: **Claude Opus 5 scored 30.2%** on ARC-AGI-3, reported as ARC Prize-verified, up from a prior best of about 7.8–8%. API price was $5/$25 per M tokens. — [aitoolsrecap](https://aitoolsrecap.com/Blog/claude-opus-5-launch-benchmarks-pricing-2026); [ARC Prize Opus 5 results](https://arcprize.org/results/anthropic-claude-opus-5)
- Sept 2026: **GPT-6 Astra (released Sept 3, 2026) scored 62.7% on ARC-AGI-3 Semi-Private with the Standard harness, at max reasoning, for a total of $26,098.**
  - With OpenAI's "Provider Adapter" harness it scored 99.9% at high reasoning for $18,817.
  - It used fewer actions than the median human on 96% of levels.
  - ARC Prize said it "builds the most precise symbolic model of novel environments we've seen." — [ARC Prize on X](https://x.com/arcprize/status/2095597602545025138); [ARC Prize Astra blog](https://arcprize.org/blog/astra); [superpowerdaily](https://superpowerdaily.com/posts/gpt-6-astra-hits-99-9-on-arc-agi-3-but-scores-62-7-in-a-shared-test); [daily.dev](https://daily.dev/posts/gpt-6-astra-benchmarks-99-9-or-62-7-the-full-arc-agi-3-story-1dabqghdx)
- Results are very sensitive to the harness, and these are public-set only:
  - An AWS Strands agent took Claude Opus 5 "from 30% to 99.95%" — [dev.to/aws](https://dev.to/aws/how-a-strands-agent-took-claude-opus-5-from-30-to-9995-on-arc-agi-3-4kel).
  - NVIDIA AVO scored 100 RHAE on all 25 **public** games (183 levels). Commentary notes that 100% on the public set "is not the same as scoring 100% on the ARC-AGI-3 benchmark" — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/), via snippet.
  - OpenAI's post is titled "How enabling two settings tripled our scores on the ARC-AGI-3 benchmark" — [OpenAI](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/).
- Chollet on NVIDIA's ARC-AGI-3 work: "Like all high-performing approaches on ARC-AGI-3, it uses deep learning-guided on-the-fly synthesis of symbolic world models, i.e. navigating the world by generating programs to represent what you know." — [François Chollet on X](https://x.com/fchollet/status/2090838046937645398)
- Compute-capped Kaggle ARC-AGI-3 track, Milestone #1 (closed Jun 30, 2026, $37.5K): 1st Tufa Labs ("Duck Harness"), 2nd Reki, 3rd Md Boktiar Mahbub Murad. Tufa's "mean score across all public games was 1.6002 ± 0.4475." The scale is unclear but appears to be low single-digit RHAE. Tufa noted that "hand-crafted tools actually hurt the model; letting it improvise worked better." Milestone #2 closes Sep 30, 2026. — [ARC Prize Milestone #1](https://arcprize.org/blog/arc-prize-2026-milestone-1); [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002); [AlphaSignal](https://alphasignal.ai/news/tufa-labs-wins-25k-beating-frontier-ai-on-the-world-s-hardest-benchmark)

**Summary table: accuracy vs cost per task (V = ARC Prize-verified on hidden set; S = self-reported public eval)**

| System | Params | ARC-AGI-1 | ARC-AGI-2 | $/task | Date | Status |
|---|---|---|---|---|---|---|
| o3-preview (low) | undisclosed | 75.7% | – | ~$20 or ~$200 (conflicting) | Dec 2024 | V |
| Greenblatt (GPT-4o, ~8k programs/task) | – | 50% public / ~42% semi-private | – | – | Jun 2024 | S/V |
| ARChitects 2024 (TTT) | small LLM | 53.5% private | – | Kaggle-capped | Nov 2024 | V (Kaggle) |
| HRM | 27M | 32% (claimed ~40–41%) | 2% | – | Aug 2025 | V |
| TRM | 7M | 40% | 6.2% | $1.76 / $2.10 | Oct 2025 | V |
| Berman (Grok 4, evolutionary NL programs) | – | 79.6% | 29.4% | $8.42 / $30.40 | Sep 2025 | reported on ARC Prize leaderboard |
| Pang (DreamCoder-style library + LLM) | – | 77.1% | 26.0% | $2.56 (v1) | Sep 2025 | reported on ARC Prize leaderboard |
| NVARC (Qwen3-4B + synthetic + TTT) | 4B | – | 24.03% private | ~$0.20 | Nov 2025 | V (Kaggle) |
| Poetiq (Gemini 3 Pro harness) | – | – | 54% | $30.57 | late 2025 | V |
| Opus 4.5 Thinking 64k | – | – | 37.6% | $2.20 | late 2025 | V |
| CompressARC (no pretraining) | 76K | 20% | – | – | 2025 | S |
| URM (universal transformer) | small | 53.8% pass@1 | 16.0% pass@1 | – | Dec 2025 | S |
| VARC (ViT + TTT) | 18M | 54.5% (60.4% ens.) | 8.3% | – | Nov 2025 | S |
| LoopViT | 18M / 3.8M | 65.8% / 60.1% | reported, no number found | – | Feb 2026 | S |
| DeepSeek V3.2 agent harness | open-weight | 67.25% pass@2 | – | $0.62 | Jul 2026 | S |
| BDH-CQ (Pathway) | 150M | 29.5% pass@2 | – | $0.0007 | Aug 2026 | S (externally reproduced) |
| Claude Opus 5.5 | undisclosed | 98.5% | 93.3% | $0.16 / $0.41 | 2026 | V |

### Inferences
- The frontier inference cost curve on ARC-AGI-1 fell by roughly 2–4 orders of magnitude in about 20 months, from $20–200+/task for 75.7% in Dec 2024 to $0.16/task for 98.5% in 2026. On per-task **inference** cost alone, tiny specialized models no longer beat the frontier: TRM costs $1.76/task for 40%. Their remaining efficiency claim is on **training** compute and parameters (7M vs undisclosed frontier sizes likely in the 10^11–10^12+ range, which is not sourced here). An "extreme low compute across the whole pipeline" argument must therefore count frontier pretraining FLOPs. That is the one term where small systems win overwhelmingly.
- On Kaggle's compute-capped track, which is the closest proxy for "low total compute", the best open result is still far from human level: 24% on ARC-AGI-2 in 2025, and low single digits on ARC-AGI-3 in mid-2026.
- ARC-AGI-3's jump from 0.4% to 62.7% in about 5 months (Apr to Sep 2026) came from frontier LLMs plus agent harnesses that build executable or symbolic world models. That mix is neurosymbolic in structure but depends on frontier-scale priors. Chollet's description ("DL-guided on-the-fly synthesis of symbolic world models") supports program synthesis as the successful **inference-time structure**, even while the "intuition" component is a large model.
- The harness effect is huge: 30% to 99.95% for the same model, and 62.7% vs 99.9% for Astra. This suggests that much of the "reasoning" gain on novel tasks comes from structure (search, verification, memory, world-model code) wrapped around a model, not from the parameters. That is the core evidence for "structure substitutes for scale", with the caveat that the base model is still frontier-sized.

### Gaps
- Could not access arcprize.org/leaderboard directly, so I have no full verified table with dates. The exact release date of Opus 5.5 and the date of its ARC Prize verification are not confirmed. I found no verified ARC-AGI-3 score for Opus 5.5, because testing was blocked by API reverse-engineering classifiers per a search snippet.
- The ARC Prize 2026 Kaggle ARC-AGI-2 track's mid-2026 top private score was not found.
- The scale of Tufa Labs' Milestone #1 Kaggle ARC-AGI-3 score ("1.6002") and its per-task cost are not confirmed.
- The o3-preview $/task figures conflict ($20 vs $200 vs $4,560 for the high-compute 87.5% run) and remain unresolved.
- ARC-AGI-4 (2027) is mentioned only in an R40 headline and is not verified.

## Key Question 2: Tiny latent/recursive reasoning models (HRM, TRM, 2026 successors): results, critiques, and what they imply about depth-via-recursion vs parameters

### Takeaway
HRM (27M parameters) and TRM (7M) showed that very small networks trained from scratch on about 1,000 examples, with iterated latent refinement, can reach 32–40% on ARC-AGI-1 (verified) and 55–87% on Sudoku-Extreme. These numbers need to be read against the critiques:

- ARC Prize and follow-up studies show the gains come mostly from the **outer refinement loop, heavy augmentation and voting, and per-task identity embeddings trained on the evaluation tasks' demonstrations**, not from the brain-inspired hierarchy.
- Effective recursion is shallow.
- Results collapse to 0% without the puzzle ID.

The 2026 successors (URM, LoopViT, BDH-CQ) push to 54–66% on ARC-AGI-1 with 4–150M parameters. Together these support **"depth via weight-tied recurrence plus iterative refinement" as a strong parameter-efficiency lever (roughly 2–5× in parameters)**. They do not yet show general, transferable reasoning.

### Cited Findings
- **HRM** (Sapient, June 2025, arXiv 2506.21734):
  - Setup: 27M params, trained from scratch on about 1,000 examples, 30×30 grid context (900 tokens), no pretraining or CoT data.
  - Self-reported: **ARC-AGI-1 40.3%**, **Sudoku-Extreme 55%**, **30×30 Maze 74.5%**. — [HRM paper](https://arxiv.org/html/2506.21734v3), via snippet
  - Training compute: Sudoku took about 10 h on one RTX 4070 laptop GPU, or about 10 min on 8 GPUs. Maze took about 1 h on 8 GPUs. ARC-1/ARC-2 took about 24 h on 8 GPUs with about 960–1,120 examples. — [HRM GitHub](https://github.com/sapientinc/HRM)
- **ARC Prize analysis of HRM** (Aug 2025), verified on the semi-private set:
  - Scores: **ARC-AGI-1 32%, ARC-AGI-2 2%**, down from roughly 41% claimed on the public set.
  - The "hierarchical" architecture had minimal impact. A plain transformer with the same parameter count and training setup came within about 5 pp.
  - The "under-documented outer loop refinement process drove substantial performance, especially at training time."
  - The model relied on puzzle-ID conditioning, i.e. fitting the evaluation tasks via puzzle IDs.
  - Augmentation plus majority vote mattered, with about 300 augmentations performing about the same as 1K. — [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis); [ARC Prize on X](https://x.com/arcprize/status/1956431617951740044); [summary by Dan Mac on X](https://x.com/daniel_mac8/status/1956513474357707036)
- **TRM** (Alexia Jolicoeur-Martineau, Samsung SAIL Montréal, Oct 2025, arXiv 2510.04871):
  - "7M parameters neural network". Self-reported **ARC-AGI-1 45%, ARC-AGI-2 8%, Sudoku-Extreme ~87%**.
  - Training compute: Sudoku on 1 L40S in under 20 h; Maze-Hard on 4 L40S in under 24 h; **ARC on 4 H100 for about 3 days**. The repo was archived Apr 1, 2026. — [TRM GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
  - TRM won 1st place in the ARC Prize 2025 Paper Award. — [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- **ARC Prize verification of TRM** (Oct 2025): **ARC-AGI-1 40% at $1.76/task; ARC-AGI-2 6.2% at $2.10/task**.
  - Greg Kamradt: "This model is tiny! 7M params, but … it is relatively expensive to run because pre-training and inference are coupled."
  - Reproducing ARC-AGI-1 semi-private took 11 h 23 min on 2×8 H100 at $8/h, a total of **$176.38**. — [ARC Prize on X](https://x.com/arcprize/status/1978872651180577060); [Greg Kamradt on X](https://x.com/GregKamradt/status/1978875294934274364)
- **TRM dissection** (Dec 2025, arXiv 2512.11847):
  - Test-time augmentation plus a 1000-sample majority vote adds about **+11 pp Pass@1** over single-pass inference.
  - Replacing the puzzle ID with a blank or random token gives **zero accuracy**. The authors suggest the ID embedding "acts as a key for retrieving a task-specific 'program' stored in the embedding space, while the small recursive trunk serves as a shared interpreter."
  - "Most of the final accuracy is achieved at the first recursion step … performance saturates after few latent updates, indicating shallow effective recursion." — [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)
- **TRM under Kaggle limits** (Ronan McGovern, Trelis, Nov 2025):
  - Pre-training on 1,280 public tasks took 700k+ steps over 48 h on 4×H100 and reached 10% on the public eval.
  - Post-training during the competition took 12,500 steps and reached **6.67% semi-private** within the compute limits. — [arXiv 2511.02886](https://arxiv.org/html/2511.02886v1)
- **URM, Universal Reasoning Model** (Dec 2025, arXiv 2512.14693): a Universal Transformer with short convolution and truncated backprop through loops. Self-reported **53.8% pass@1 on ARC-AGI-1 and 16.0% pass@1 on ARC-AGI-2**, vs 40% (TRM) and 34.4% (HRM) on ARC-AGI-1. The authors attribute the gains to "the recurrent inductive bias and strong nonlinear components of Transformer, rather than elaborate architectural designs." — [URM arXiv](https://arxiv.org/abs/2512.14693)
- **VARC, "ARC Is a Vision Problem!"** (MIT, Kaiming He group, Nov 2025): an 18M ViT trained from scratch, with canvas representation, augmentation and TTT.
  - Single-model **ARC-1 54.5±0.7%, ARC-2 8.3±0.4%**.
  - An ensemble of the ViT and a 55M U-Net reaches **60.4%** on ARC-1. — [arXiv 2511.14761](https://arxiv.org/abs/2511.14761)
- **LoopViT** (Feb 2026, arXiv 2602.02156): a weight-tied looped hybrid conv+attention block with an entropy-based dynamic exit that needs no extra parameters.
  - The **18M model reaches 65.8% on ARC-AGI-1**; the **3.8M model reaches 60.1%**.
  - A 5.9M LoopViT beats the 18M VARC baseline, and an 11.2M LoopViT beats the 73M VARC ensemble.
  - The authors say "adaptive iterative computation offers a far more efficient scaling axis … than simply increasing network width." — [LoopViT arXiv](https://arxiv.org/abs/2602.02156); [LoopViT GitHub](https://github.com/WenjieShu/LoopViT)
- **CompressARC** (Isaac Liao and Albert Gu, CMU; blog 2025, arXiv Dec 2025): **76K parameters with no pretraining**. It trains only on the single target puzzle at inference by minimizing description length (MDL) and solves **20% of the ARC-AGI-1 eval set**. — [CompressARC blog](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html); [arXiv 2512.06104](https://arxiv.org/abs/2512.06104)
- **BDH-CQ** (Pathway, Aug 2026, arXiv 2608.09888): demonstrations update a recurrent memory, and the model then reasons in continuous latent space without verbalizing.
  - The **150M model reaches 29.5% pass@2 on the ARC-AGI-1 public eval at $0.0007/task**, about $0.28 for all 400 tasks. The authors claim this breaks the ARC-AGI-1 cost-accuracy Pareto frontier.
  - Remigiusz Kinas (Bielik AI) and Richard Zhong (NYU) reproduced the result with a black-box evaluation. — [arXiv 2608.09888](https://arxiv.org/abs/2608.09888); [alphaXiv on X](https://x.com/askalphaxiv/status/2088038388343468370); [XenoSpectrum](https://xenospectrum.com/en/bdhcq-latent-reasoning-arc-cost/)
- **Tiny Autoregressive Recursive Models** (Mar 2026, ICLR 2026 workshop): found "no reliable performance gains from the full Autoregressive TRM architecture". The authors caution against AR-TRM as a research direction, though two-step refinement shows some promise. — [arXiv 2603.08082](https://arxiv.org/html/2603.08082); [OpenReview](https://openreview.net/pdf?id=aY5kmaNrwB)
- Other 2026 follow-ups exist, but I only have their titles: "What Survives When You Compress a Recursive Reasoner for the Edge?" (arXiv 2606.26488), "Steering Recurrent Reasoners at Inference Time with Readout Feedback" (arXiv 2608.24136), "Boosting Inference with Guided Reasoning: Stochastic Exploration for Recursive Models" (arXiv 2605.25230), and "Dissecting Hierarchical Reasoning Models: A Mechanistic Study" (arXiv 2609.22197). — titles from search results only

### Inferences
- Training-compute scale (my estimate, not from a source): TRM's ARC run of about 4 H100 × 3 days is about 288 H100-hours. At about 1 PFLOP/s bf16 peak and 30–40% utilization, that is roughly **3–4×10^20 FLOPs**, which costs on the order of $600–2,300 at $2–8 per GPU-hour. HRM's ARC run of about 8 GPUs × 24 h is a similar order of magnitude. Both are about 5–6 orders of magnitude below published frontier pretraining budgets. The frontier figures should come from other researchers' notes.
- Recurrence buys parameter efficiency. LoopViT at 3.8M beats an 18M feed-forward ViT, and URM beats TRM and HRM through a better recurrent block. But the dissections show much of the "reasoning" is task-specific fitting: per-task ID embeddings are trained on the evaluation tasks' demonstration pairs, followed by heavy voting. **These are transductive per-benchmark learners, not general reasoners.** They are closer to "fast program induction by gradient descent" than to a reusable general model.
- The consistent lesson across HRM, TRM, Poetiq and the ARChitects' 2025 "recursive self-refinement" is that **iterative refinement loops** at training time and at inference are the key ingredient. The specific architecture matters much less.
- CompressARC (76K parameters, zero pretraining, 20%) is the purest evidence that an inductive bias plus MDL search at test time can extract a nontrivial amount of abstraction with near-zero prior knowledge. It is still far below human level.

### Gaps
- There is no ARC Prize-verified hidden-set score for URM, VARC, LoopViT or BDH-CQ. Their numbers are public-eval only, and the public eval is known to be easier and was used during development.
- LoopViT's ARC-AGI-2 number was not retrieved.
- I did not find reliable training-FLOP or GPU-hour figures for URM, LoopViT, VARC or BDH-CQ.
- I found no evidence that any HRM/TRM-family model has been applied to ARC-AGI-3 or to language tasks with success. The only related result is the negative AR-TRM finding.

## Key Question 3: Program synthesis and library learning (DreamCoder, LILO, Stitch, Chollet's position, DSL search, LLM-guided program search, neurosymbolic hybrids)

### Takeaway
LLM-guided program search with symbolic verification is the most consistently successful ARC paradigm. Its efficiency improved by orders of magnitude between 2024 and 2026:

| System | Date | LLM calls per task | ARC-AGI-1 | $/task |
|---|---|---|---|---|
| Greenblatt (GPT-4o) | 2024 | ~8,000 | 50% public | – |
| Berman | 2025 | ~500 | 79.6% | $8.42 |
| Pang (DreamCoder-style library reuse) | 2025 | ~10 | 77.1% | $2.56 |
| Open-weight DeepSeek V3.2 harness | 2026 | – | 67% | $0.62 |

Library learning (reusing abstractions) was the single biggest efficiency jump. The same structure, a learned proposal model plus executable hypotheses checked against evidence, dominates ARC-AGI-3 as "programmatic world models". Chollet's thesis is that intelligence means skill-acquisition efficiency and needs DL-guided discrete program search. The ARC-AGI-3 results fit that thesis, and his new lab Ndea is betting on it.

### Cited Findings
- Chollet and ARC Prize's framing: "The conclusion that deep learning can scale to AGI assumes a certain definition of AGI – one that is oriented around AI's skill as opposed to AI's ability to learn skill." ARC was published in 2019 "to test whether AI systems could efficiently learn skill." On search: "The best solution to fight combinatorial explosion … is to leverage intuition over the structure of program space, provided by a deep learning model … Deep learning models are inexact and need to be complemented with discrete search and symbolic checking." — [ARC Prize: How to beat ARC-AGI by combining DL and program synthesis](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis)
- Chollet in June 2024: "If you are generating lots of programs, checking each one with a symbolic checker … and selecting those that work, you are doing program synthesis." — [François Chollet on X](https://x.com/fchollet/status/1802801425514410275)
- **Ndea** (YC W2026, $43M), founded by Chollet and Mike Knoop, is "betting that program synthesis fused with deep learning — not larger transformers — is the path to general intelligence." — [StartupHub.ai](https://www.startuphub.ai/ai-news/claudes-corner/2026/claudes-corner-ndea-yc-w2026) (secondary source)
- **Greenblatt** (Redwood, June 2024): GPT-4o generated about 8,000 Python programs per problem, selected by correctness on the demonstrations. It scored **50% on the public test**. On ARC Prize's then-new semi-private leaderboard it scored about 42% and was at the top. — [Redwood blog](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt); [Buck Shlegeris on X](https://x.com/bshlgrs/status/1806397587085468116)
- **Jeremy Berman** (Sept 2025): Grok 4 with "multi-agent collaboration with evolutionary test-time compute" evolving natural-language instructions instead of Python. Scores: **ARC-AGI-1 79.6% at $8.42/task; ARC-AGI-2 29.4% at $30.40/task**, the top ARC-AGI-2 leaderboard score at the time. — [Berman on X](https://x.com/jerber888/status/1968001933211471891); [StartupHub.ai](https://www.startuphub.ai/ai-news/ai-video/2025/jeremy-bermans-evolutionary-leap-natural-language-for-arc-agi-2)
- **Eric Pang** (Sept 2025): "DreamCoder-inspired, LLM-assisted program synthesis" with an expanding library of learned concepts.
  - Scores: **ARC-AGI-1 77.1% at about $2.56/task; ARC-AGI-2 26.0%**.
  - Uses about 10 LLM calls per task, vs about 500 for Berman and about 8,000 for Greenblatt. Pang claims it broke the cost-performance Pareto frontier. — [Pang write-up (gist)](https://gist.github.com/inspiredlabs/a3fc232eba4b9754ad4a8234b85b8d34); [Pang GitHub](https://github.com/epang080516/arc_agi); [Pang on X](https://x.com/_eric_pang_/status/1968009342508208589)
- **SOAR** (Pourcel, Colas, Oudeyer; ICML 2025): alternates LLM-driven evolutionary search with "hindsight" fine-tuning of the LLM on its own search traces. It takes open LLMs "from just a few percent" to **52% on the ARC-AGI-1 public test**, with no human DSL. The authors released 5M generated programs. — [arXiv 2507.14172](https://arxiv.org/abs/2507.14172); [PMLR](https://proceedings.mlr.press/v267/pourcel25a.html)
- **Cost-effective agent harnesses** (Jul 2026, arXiv 2607.06764): open-weight DeepSeek V3.2 in non-thinking mode, with no ARC fine-tuning.
  - An Explorer-Definer pipeline separates pattern discovery from program synthesis. It reaches **57.50% pass@2 at $0.25/task**.
  - A Reflective Orchestrator reaches **67.25% at $0.62/task**, up from a 15.50% base (+51.75 pp from architecture alone). The authors say this is more than 10× cheaper than comparable custom frontier-model systems (public eval). — [arXiv 2607.06764](https://arxiv.org/abs/2607.06764)
- TTT ensembled with program synthesis (BARC) reached 61.9% on the ARC-AGI-1 public validation set, "matching average human performance" (Nov 2024). — [Akyürek et al., arXiv 2411.07279](https://arxiv.org/abs/2411.07279)
- **DreamCoder** (Ellis et al.): wake-sleep library learning. The wake phase searches for programs with the current library. The sleep phase extracts common sub-expressions as new primitives when they reduce total description length, and trains a neural search policy on replays and "dreams". — [DreamCoder, arXiv 2006.08381](https://arxiv.org/pdf/2006.08381); [PLDI 2021](https://dl.acm.org/doi/10.1145/3453483.3454080)
- **Stitch**: top-down library learning that is "1000–10000x faster and 100x more memory efficient than DreamCoder's compression" and runs in seconds on one CPU. — [Stitch, arXiv 2211.16605](https://arxiv.org/pdf/2211.16605)
- **LILO**: LLM synthesis plus Stitch compression plus auto-documentation. It outperforms DreamCoder on REGEX (+33.14), LOGO (+20.42) and CLEVR (+2.26). — [LILO, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/819cebb05f993840e8a52d7564c5c282-Paper-Conference.pdf)
- **ARC-AGI-3 programmatic world models**:
  - Tycho (Jul 30, 2026) models games as "parameterized rendered deterministic Moore machines". A coding agent builds, tests, repairs or bypasses executable hypotheses. With Claude Opus 4.8, delegating to a model builder gave the best mean RHAE: **88.49 on the 25 public games**. — [Tycho, arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
  - Related papers: "Executable World Models for ARC-AGI-3 in the Era of Coding Agents" ([arXiv 2605.05138](https://arxiv.org/html/2605.05138v2)) and "OPINE-World: Programmatic World Modeling…" ([arXiv 2607.01531](https://arxiv.org/pdf/2607.01531)). Only their titles were retrieved.
  - Chollet: all high-performing ARC-AGI-3 approaches use "deep learning-guided on-the-fly synthesis of symbolic world models". — [Chollet on X](https://x.com/fchollet/status/2090838046937645398)

### Inferences
- The strongest compute-efficiency evidence in the whole ARC record is **library reuse plus a small number of LLM proposals plus execution checks**. Pang cut LLM calls by 50× relative to Berman at similar accuracy on ARC-AGI-1. Stitch made symbolic compression 1000×+ cheaper. Search amortized through learned abstractions is where the efficiency comes from.
- The 2026 DeepSeek V3.2 harness result (+51.75 pp from structure alone at $0.25–0.62/task) is direct evidence that **explicit hypothesis-verify loops can substitute for a large share of model scale and reasoning-token spend**. It still needs a capable mid/large open model as the proposer.
- ARC-AGI-3 makes the case for program synthesis stronger: agents that write and test code models of an environment do best. The best "intuition engines" in 2026 are still frontier LLMs, though. Nothing yet shows a small proposer paired with symbolic search reaching competitive ARC-AGI-3 scores under Kaggle limits.

### Gaps
- There are no ARC Prize-verified semi-private numbers for SOAR or for the 2026 DeepSeek harness.
- Pang's ARC-AGI-2 $/task was not retrieved.
- I found no quantified training compute for the LLM proposers used in these systems. That compute dominates true "whole-pipeline" cost and is hidden inside API prices.
- Chollet's 2019 "On the Measure of Intelligence" was not fetched directly. Its definition (intelligence = skill-acquisition efficiency over a scope of tasks, relative to priors and experience) is paraphrased here from ARC Prize's blog framing only.

## Key Question 4: Test-time training / test-time adaptation (TTT/TTA) and how much it helps small models

### Takeaway
Test-time training was the defining technique of the Kaggle-constrained ARC competitions from 2023 to 2025. In each case it roughly doubled small-model performance:

- MindsAI went from 33% to 55.5%.
- Akyürek et al.'s 8B model reached 53%, and 61.9% ensembled with program synthesis.
- The 2024 winner (ARChitects, 53.5%) and the 2025 winner (NVARC, 24% on ARC-AGI-2 at $0.20/task) both used TTT.

At its extreme, test-time adaptation alone with a 76K-parameter model and no pretraining (CompressARC) gets 20%. Small models gain the most from TTT, but the gains come from fitting each task's few demonstrations. General capability beyond that is not demonstrated.

### Cited Findings
- **Akyürek et al.** (Nov 2024; ICML 2025, retitled "…for Few-Shot Learning"): TTT on an 8B LM gives **53% on the ARC public validation set**, "improving the state-of-the-art by nearly 25% for public and purely neural approaches". Ensembled with program synthesis it reaches **61.9%**, "matching average human performance".
  - The three crucial components are: (1) initial fine-tuning on similar tasks, (2) auxiliary task format and augmentations, and (3) per-instance training. — [arXiv 2411.07279](https://arxiv.org/abs/2411.07279); [PMLR v267](https://proceedings.mlr.press/v267/akyurek25a.html)
  - The released code uses Llama-3 8B and 8.1B BARC checkpoints. — [marc GitHub](https://github.com/ekinakyurek/marc)
- MindsAI pioneered TTT for ARC in 2023 and raised the private-set SOTA from 33% to 55.5% in 2024. The ARChitects' 53.5% (1st place, 2024) also used TTT. — [ARC Prize 2024 Technical Report](https://arxiv.org/html/2412.04604v1)
- 2025: NVARC (1st) built on the ARChitects' TTT pipeline with a Qwen3-4B model and "on-the-fly gradient steps using each test puzzle's small example set". It scored **24.03% private at about $0.20/task**. MindsAI (3rd, 12.64%) used a "heavily-engineered test-time-training pipeline". — [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis); [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/) (via snippet)
- VARC's 18M ViT uses TTT as a core pipeline component and reaches 54.5% on ARC-1. — [arXiv 2511.14761](https://arxiv.org/abs/2511.14761)
- TRM test-time adaptation under Kaggle limits: 12,500 post-training steps give 6.67% semi-private on ARC-AGI-2. — [arXiv 2511.02886](https://arxiv.org/html/2511.02886v1)
- CompressARC: "the only deep learning method for ARC-AGI where training happens only on a single sample: the target inference puzzle itself". It uses 76K parameters and scores 20% on ARC-AGI-1. — [CompressARC](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)
- Test-time RL at scale: AlphaProof's solve rates on hard problems come "after an initial TTRL compute investment of 50 TPU days or 1 day on 50 TPUs per problem". — [Julian Schrittwieser blog on AlphaProof paper](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/)
- Inference-time refinement without weight updates, for comparison: Poetiq's harness took Gemini 3 Pro from 31% ($0.81/task) to 54% ($30.57/task) on ARC-AGI-2. That is +23 pp for about 38× the cost. — [bdtechtalks](https://bdtechtalks.substack.com/p/poetiq-crushed-arc-agi-2-at-half)

### Inferences
- TTT is the most compute-efficient known way to turn a small model's prior into task-specific skill under hard budgets. It is effectively "learning at inference", which fits Chollet's skill-acquisition framing well.
- The cost moves into inference. Each task pays for gradient steps, which is why TRM's per-task cost ($1.76) is higher than frontier LLMs' ($0.16) despite 7M parameters.
- For an ASI-under-low-compute argument, the key open question is whether TTT's gains **compound** across tasks (continual or library-style learning) or are thrown away after each task. In every ARC system above they are discarded, which wastes compute.

### Gaps
- I found no systematic measurement of TTT's per-task FLOPs for the 2024–2025 Kaggle winners.
- There is no published scaling law (TTT gain vs base-model size) in the sources I retrieved.
- I found no evidence of TTT being used successfully in the ARC-AGI-3 interactive setting under Kaggle limits.

## Key Question 5: AlphaGeometry / AlphaProof and other symbolic+neural systems: compute vs capability, and whether structure (verifiers, search) substitutes for scale

### Takeaway
Formal-verifier systems reached olympiad-level mathematics with far smaller models than frontier LLMs, but only in narrow, fully formalized domains:

- AlphaGeometry2 solves 84% of IMO geometry problems from 2000–2024.
- AlphaProof reached IMO 2024 silver together with AG2.

They pay for it with very heavy search or test-time RL: about 50 TPU-days per hard problem for AlphaProof. By 2025, general LLMs with verification-and-refinement pipelines reached IMO gold in natural language. The verifier and search structure transfers. The narrow symbolic engines do not.

### Cited Findings
- **AlphaGeometry2** (Feb 2025; JMLR 2025): **84% solve rate on all IMO geometry problems 2000–2024**, up from 54% for AlphaGeometry 1.
  - Coverage of the domain language rose from 66% to 88%.
  - The symbolic engine was rewritten in C++ and is about two orders of magnitude faster.
  - The language model uses the Gemini architecture, trained on an order of magnitude more synthetic data.
  - It uses a new search, SKEST (Shared Knowledge Ensemble of Search Trees), and solves 20 of 30 of the hardest IMO-shortlist geometry problems. — [AlphaGeometry2, arXiv 2502.03544](https://arxiv.org/html/2502.03544v1); [JMLR](http://jmlr.org/papers/volume26/25-1654/25-1654.pdf)
- **AlphaProof** (IMO 2024; Nature paper Nov 2025): a pre-trained LM coupled with AlphaZero RL in Lean. Together with AG2 it reached IMO 2024 silver-medal standard, the first medal-level AI result. — [DeepMind blog](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/); [Nature/PubMed](https://pubmed.ncbi.nlm.nih.gov/41225005/)
- AlphaProof's compute: about **50 TPU-days per problem** of test-time RL. — [Julian Schrittwieser blog](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/)
- IMO 2025 gold was reached with a "Model-Agnostic Verification-and-Refinement Pipeline" on general LLMs. — [arXiv 2507.15855](https://arxiv.org/pdf/2507.15855) (title only)

### Inferences
- These systems show that **a sound verifier plus search lets a mid-sized model reach expert-level results in a closed formal domain**, which is evidence that structure substitutes for scale there. The substitution is domain-bound, because it needs a formal language, a symbolic engine and a verifier.
- The compute is not small: 50 TPU-days per problem is huge next to $0.16–$30 per ARC task. Search replaces parameters but not FLOPs. The efficiency gain is in data and parameters, not in total compute.

### Gaps
- AlphaGeometry1's original numbers (25/30 IMO problems, about 100M synthetic training examples) were not confirmed by a retrieved source in this session.
- AlphaProof's training compute and model size were not retrieved.
- Whether AlphaProof-style systems competed at IMO 2025 is unclear from my sources.

## Key Question 6: Latent reasoning in continuous space (Coconut, looped/universal transformers, depth-recurrent models such as Geiping et al. 2025)

### Takeaway
Weight-tied recurrence and continuous latent reasoning give real but moderate efficiency gains:

- Ouro's looped LMs match standard models 2–3× their size, at 7.7T training tokens.
- Huginn (3.5B) scales test-time compute by looping, up to the compute of a model about 50B in size.
- Coconut solves planning-style tasks (ProsQA) with far fewer "thought" tokens but underperforms text CoT on GSM8K.

In Sept 2026 GPT-6 Astra, the top ARC-AGI-3 system, was **reported (not confirmed by OpenAI)** to use a constrained "recurrent depth" architecture. If true, the looped/latent-depth idea has moved from tiny-model research into the frontier. That would be evidence that depth via recurrence scales, but inside large models rather than instead of them.

### Cited Findings
- **Coconut** (Meta, Dec 2024): feeds the last hidden state back as a "continuous thought".
  - **ProsQA**: 97.0% with about 14.2 latent vectors, vs 77.5% with 49.4 text tokens for CoT (GPT-2 scale).
  - **GSM8K**: 34.1% vs 42.9% for the CoT baseline, using 8.2 vectors vs 25 tokens. — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/meta-introduces-chain-of-continuous-thought-coconut-to-improve-next-token-prediction); [arXiv 2412.06769](https://arxiv.org/html/2412.06769v1)
  - A later analysis finds causal leverage across latent steps is "highly heterogeneous", with a few steps having outsized influence. — [arXiv 2602.08783](https://arxiv.org/pdf/2602.08783)
  - A related 2026 paper is titled "The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models". — [arXiv 2604.06374](https://arxiv.org/pdf/2604.06374) (title only)
- **Huginn-3.5B** (Geiping et al., Feb 2025): a depth-recurrent LM with 3.5B params, trained on 800B tokens on AMD MI250X hardware. Unrolling up to about 50 loops at inference improves reasoning benchmarks, reaching "an effective compute budget equivalent to a 50B-parameter model". It needs no special CoT data. — [Geiping et al., arXiv 2502.05171](https://arxiv.org/pdf/2502.05171); [EmergentMind summary](https://www.emergentmind.com/topics/huginn-3-5b)
- **Ouro** (ByteDance Seed, Oct 2025): looped LMs of **1.4B and 2.6B trained on 7.7T tokens match 4B and 8B standard transformers**, a 2–3× parameter-efficiency gain. It uses an entropy-regularized objective for learned depth allocation. — [arXiv 2510.25741](https://arxiv.org/abs/2510.25741); [Ouro project page](https://ouro-llm.github.io/)
- **URM** (Universal Transformer lineage) and **LoopViT** on ARC: see Key Question 2. URM reaches 53.8% on ARC-AGI-1 and 16% on ARC-AGI-2 (self-reported). LoopViT's 3.8M model reaches 60.1% on ARC-AGI-1.
- **GPT-6 Astra architecture** (Sept 2026, reported, not confirmed):
  - The Information reported on Sept 1 that Astra uses "a constrained form of 'recurrent depth'", reusing the same Transformer layers more than once before each token.
  - Neither OpenAI's model card nor its launch material names recurrent depth or looped Transformers.
  - Chief Scientist Jakub Pachocki reportedly said Astra's computation depth "stays close to GPT-4 levels" and that loops are capped to preserve chain-of-thought monitorability. — [kingy.ai](https://kingy.ai/blog/recurrent-depth-openai-astra/); [ABAB News](https://www.ababnews.com/news/c9402727-7ef1-4842-a88a-a103b83b48ec); [Sebastian Raschka magazine](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and)
  - Claims that Astra "leverages ByteDance's architecture" (Ouro) come from a crypto-news feed and are low reliability. — [Lookonchain](https://www.lookonchain.com/feeds/71057)
- Other 2026 looped-model follow-ups, titles only: LoopCoder-v2 "Only Loop Once for Efficient Test-Time Computation Scaling" ([arXiv 2606.18023](https://arxiv.org/pdf/2606.18023)), "Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth" ([arXiv 2608.18222](https://arxiv.org/pdf/2608.18222)), and "Hyperloop Transformers" ([arXiv 2604.21254](https://arxiv.org/pdf/2604.21254)).

### Inferences
- Depth via recurrence consistently buys about 2–5× parameter efficiency, at LM scale (Ouro 2–3×, Huginn ~14× in effective compute) and at tiny scale (LoopViT ~5×). It **does not reduce training tokens**: Ouro used 7.7T. It also shifts cost to inference FLOPs, since each loop is a full pass.
- If the Astra reports are accurate, the frontier has adopted this idea, which argues against a clean "tiny models vs scale" split. The likely direction is that looped depth, refinement and program-synthesis harnesses get absorbed into large pretrained models.
- Latent (non-verbal) reasoning cuts inference tokens, as in Coconut and BDH-CQ ($0.0007/task). It weakens interpretability and monitorability, a trade-off OpenAI reportedly managed by capping loops.

### Gaps
- Astra's architecture, parameter count and training compute are not officially disclosed.
- I did not retrieve Huginn's per-benchmark numbers (GSM8K, ARC-Challenge etc.) from the model card, because huggingface.co was blocked.
- I found no ARC-AGI-2 or ARC-AGI-3 result for Huginn, Ouro or Coconut.

## Key Question 7: Main limitations: do these approaches generalize beyond narrow puzzle domains to language, world knowledge and open-ended tasks?

### Takeaway
The low-compute specialists mostly do not generalize:

- HRM, TRM, URM, VARC and LoopViT are per-benchmark, transductive learners. Their ARC performance depends on training on the evaluation tasks' demonstrations with per-task ID embeddings, and it collapses to 0% without them.
- They have no language or world knowledge, and the autoregressive/language adaptation of TRM gave no reliable gains.
- Every paradigm loses 2–3× from ARC-AGI-1 to ARC-AGI-2.

The methods that did generalize to ARC-AGI-3 and to olympiad mathematics combine a **large pretrained model (knowledge and intuition) with structure (program synthesis, verification, refinement, test-time adaptation, looped depth)**. The best-supported low-compute direction is therefore "small or mid-sized learned prior plus explicit search/verification plus test-time learning plus reusable libraries". The evidence does not support "tiny model alone". The pretrained prior remains the dominant and unsolved compute cost.

### Cited Findings
- The HRM/TRM puzzle-ID dependence is covered under Key Question 2: a blank or random ID gives 0% ([arXiv 2512.11847](https://arxiv.org/abs/2512.11847)), and HRM relied on puzzle IDs fit to the evaluation tasks ([ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)).
- TRM's effective recursion is shallow: most accuracy arrives at the first recursion step. — [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)
- AR-TRM showed no reliable gains, and the authors caution against the direction. — [arXiv 2603.08082](https://arxiv.org/html/2603.08082)
- Program synthesis, neuro-symbolic and neural methods all drop 2–3× from ARC-AGI-1 to ARC-AGI-2. — [Living survey](https://arxiv.org/html/2603.13372v1)
- NVARC's small 4B model depended on synthetic data seeded partly by GPT-4o summaries, i.e. distillation from a large model. — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/) (via snippet)
- Frontier LLM failure modes on ARC-AGI-3 (Apr/May 2026) were "true local effect, false world model", "wrong level of abstraction from training data", and "solved the level, didn't reinforce the reward". — [ARC Prize on X](https://x.com/arcprize/status/2050261221165989969)
- On ARC-AGI-3, the successful systems are frontier LLMs in world-model-building harnesses: Opus 5 at 30.2%, Astra at 62.7%, and Tycho with Opus 4.8 at 88.49 RHAE on the public set. Compute-capped Kaggle agents remain far lower. — [ARC Prize Astra](https://arcprize.org/blog/astra); [Tycho](https://arxiv.org/html/2607.28287v1); [ARC Prize Milestone #1](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- Formal symbolic systems (AG2, AlphaProof) are restricted to formalized domains and still need huge search compute (about 50 TPU-days per problem). — [Julian Schrittwieser blog](https://www.julian.ac/blog/2025/11/13/alphaproof-paper/)
- Coconut underperforms text CoT on GSM8K (34.1% vs 42.9%). — [The Batch](https://www.deeplearning.ai/the-batch/meta-introduces-chain-of-continuous-thought-coconut-to-improve-next-token-prediction)

### Inferences
These are synthesis points for the Chinese ASI report.

1. The empirical record separates **priors** (world knowledge and intuition, currently obtained only by large-scale pretraining) from **skill acquisition** (search, verification, refinement, TTT, library growth). The low-compute methods have become very efficient at the second part: 7M–150M-parameter models, sub-$1/task harnesses, and 10 LLM calls per task with library reuse. None has yet shown how to get the first part cheaply.
2. The most compute-efficient demonstrated recipe for novel-task reasoning, as of Sept 2026, has five parts:
   - (a) a moderately sized pretrained proposer, possibly with looped depth;
   - (b) program or world-model synthesis with execution-based verification;
   - (c) iterative refinement loops;
   - (d) test-time adaptation;
   - (e) an accumulating library of abstractions (DreamCoder/Stitch/Pang style) so that search cost is amortized across tasks.

   Every ARC milestone from 2024 to 2026 uses at least three of these five.
3. The "tiny recursive model" results are best read as evidence that **depth/iteration is a cheaper scaling axis than width/parameters** (2–5× parameter savings). They do not show that general reasoning can emerge in a 7M network. Without transfer across tasks and a knowledge prior, they are sophisticated per-task program induction.
4. The frontier's ARC-AGI-1/2 inference cost collapse (98.5% / 93.3% at $0.16 / $0.41 per task) means any claim that small systems are "more efficient" must be made on **total pipeline cost including pretraining**. Those numbers need frontier-training-FLOP estimates, which should come from other researchers' notes. It also means the frontier is itself absorbing the low-compute techniques: refinement harnesses, and reportedly looped depth in Astra.
5. ARC-AGI-3 (interactive, action-efficiency scored) is currently the best benchmark for skill-acquisition efficiency. Its leading paradigm is programmatic world-model synthesis, which is Chollet's thesis in practice.

### Gaps
- I found no study showing HRM/TRM/URM-style tiny recursive models succeeding on natural-language, world-knowledge or open-ended tasks.
- I found no whole-pipeline FLOP accounting (pretraining + fine-tuning + inference) that compares a frontier LLM on ARC with a tiny specialist on the same basis.
- I found no evidence on whether library-learning systems keep compounding over very long task sequences (thousands of diverse tasks) without library bloat.
- Ndea's technical results, if any were published by Sept 2026, were not found.
