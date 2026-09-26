# Brain-inspired, developmental and non-mainstream paradigms as routes to low-compute general intelligence (status as of Sept 2026)

Method note: arXiv, ACL Anthology, OpenReview, PubMed, Meta AI, VERSES, TechCrunch and several other domains were blocked by this session's egress proxy. Most numbers therefore come from search-engine extracts of primary sources, plus one GitHub mirror of the V-JEPA 2 paper that I fetched in full. Primary-source URLs are cited where the extract clearly came from them. Where only secondary or low-quality sources were available, the notes say so. Each item is dated. Claims are tagged as **[peer-reviewed]**, **[preprint]**, **[company claim]** or **[independent check]** where that distinction matters.

---

## 1. World models and self-supervised predictive architectures (JEPA line, AMI Labs, Dreamer, Genie)

### Takeaway
Of all the "alternative" paradigms, the JEPA and latent-world-model line has the strongest and most reproducible evidence of efficiency gains. The results are: I-JEPA pretraining is more than 10x cheaper than MAE; V-JEPA 2-AC plans 16x faster than a generative video world model and reaches 65–80% zero-shot pick-and-place from 62 hours of robot video; Dreamer 4 matches VPT-style behaviour with about 100x less data. However, these are still backprop-trained deep nets on GPUs, and V-JEPA 2's pretraining used over 1M hours of video. The gains are therefore "smarter objective / plan in latent space", not "tiny compute". No JEPA system has yet shown general-purpose (language/reasoning) capability per FLOP above LLMs.

### Cited Findings
**LeCun's thesis (2022).**
- In "A Path Towards Autonomous Machine Intelligence" (June 2022, position paper, not an empirical result), LeCun proposes a modular architecture: perception, a world model, cost/intrinsic motivation, an actor, short-term memory and a configurator. At its centre is a hierarchical JEPA trained self-supervised to predict in *representation space* rather than pixel/token space. He argues that this, plus energy-based (non-contrastive, non-generative) training and model-predictive planning, is the route to human-like sample efficiency — [LeCun 2022, OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)

**I-JEPA (CVPR 2023) [peer-reviewed].**
- A ViT-Huge/14 was pretrained on ImageNet in under 72 hours on 16 A100s (under 1,200 GPU-hours). This is "over 2.5× faster than a ViT-S/16 pretrained with iBOT and over 10× more efficient than a ViT-H/14 pretrained with MAE" — [Assran et al., arXiv 2301.08243](https://arxiv.org/pdf/2301.08243); [Meta AI blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)

**V-JEPA 2 (June 2025) [preprint, Meta].**
- Architecture and data:
  - Encoder: ViT-g, about 1B parameters.
  - Pretraining data: VideoMix22M, 22M samples and over 1M hours of video plus about 1M images (SSv2, Kinetics, HowTo100M, curated YT-1B, ImageNet).
  - Pretraining GPU-hours: not stated in the text I could access.
  — [V-JEPA 2 paper mirror](https://github.com/Ar9av/daily-research-papers/blob/main/2026-06-09-2506.09985/paper.md); [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- Understanding benchmarks:
  - SSv2: 77.3% top-1.
  - Epic-Kitchens-100 action anticipation: recall@5 of 39.7, a 44% relative gain over the prior best.
  - After alignment with an LLM for video QA: PerceptionTest 84.0 and TempCompass 76.9.
  — [paper mirror](https://github.com/Ar9av/daily-research-papers/blob/main/2026-06-09-2506.09985/paper.md); [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- V-JEPA 2-AC (robot planning):
  - Setup: a 300M-parameter action-conditioned predictor trained on only 62 hours of unlabeled DROID robot video.
  - Zero-shot results on Franka arms in unseen labs: reach 100%, grasp 72.5%, pick-and-place 75%. The Octo baseline, trained on 1M+ trajectories, scored 100% / 15% / 15%.
  — [paper mirror](https://github.com/Ar9av/daily-research-papers/blob/main/2026-06-09-2506.09985/paper.md)
- Planning speed versus Cosmos:
  - Time per action: 16 s for V-JEPA 2-AC versus about 4 min for NVIDIA Cosmos, a pixel-generative world model.
  - Success: 60–80% for V-JEPA 2-AC versus 0–20% for Cosmos.
  - Sampling budget: 800 samples for V-JEPA 2-AC versus 80 for Cosmos.
  — [EmergentMind summary of 2506.09985](https://www.emergentmind.com/topics/v-jepa-2-ac); [Gonzo ML review](https://gonzoml.substack.com/p/v-jepa-2-scaling-v-jepa)

**LeJEPA and later JEPA work.**
- LeJEPA (Balestriero & LeCun, Nov 14 2025, arXiv 2511.08544) [preprint]:
  - Proves that an isotropic Gaussian embedding distribution minimizes worst-case downstream risk.
  - Introduces SIGReg, a linear-complexity regularizer that replaces the stop-grad, EMA-teacher and other anti-collapse heuristics.
  - Reports that in-domain pretraining beats larger generic pretrained models.
  — [alphaXiv 2511.08544](https://www.alphaxiv.org/abs/2511.08544); [code](https://github.com/rbalestr-lab/lejepa)
- Follow-ups visible in search results:
  - "When Does LeJEPA Learn a World Model?" (arXiv 2605.26379, May 2026).
  - LeVJEPA, on video pretraining without heuristics (arXiv 2608.27395, Aug 2026).
  - I could not read their contents — [2605.26379](https://arxiv.org/pdf/2605.26379); [2608.27395](https://arxiv.org/pdf/2608.27395)
- Reports that "V-JEPA 2.1" was released in March 2026 and that a "stable-worldmodel" benchmark followed on May 20 2026 come only from low-quality aggregator blogs. They are unverified — [StartupHub](https://www.startuphub.ai/ai-news/ai-figures/2026/figure-yann-lecun-jepa-release-cadence-2026-08-20)

**AMI Labs (2026).**
- LeCun's post-Meta company, Advanced Machine Intelligence (AMI) Labs, is Paris-based. It announced a $1.03B seed at a $3.5B pre-money valuation (about $4.5B post-money) on Mar 9 2026, described as Europe's largest seed round.
  - Stated plan: build JEPA-based world models for industrial, robotics and healthcare uses.
  - Team: Saining Xie (Chief Science Officer), Pascale Fung, Michael Rabbat (VP world models), Laurent Solly (COO).
  — [TechCrunch, 2026-03-09](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/); [Latent Space](https://www.latent.space/p/ainews-yann-lecuns-ami-labs-launches)
- Conflict: one site's headline says "$30M". This contradicts all other sources and looks like an error — [tech-insider](https://tech-insider.org/yann-lecun-ami-labs-1-billion-world-models-2026/)

**DreamerV3 (Nature, April 2025) [peer-reviewed].**
- One fixed set of hyperparameters across 150+ tasks. It was the first algorithm to collect Minecraft diamonds from scratch without human data or curricula. All Dreamer agents were trained on one GPU each — [Nature s41586-025-08744-2](https://www.nature.com/articles/s41586-025-08744-2); [project page](https://danijar.com/project/dreamerv3/)
- Budget: 30M environment steps, described as "17 days" of gameplay.
- Conflict on hardware: some secondary sources say "a single V100", while the paper states agents were trained on a single A100.
- Comparison: OpenAI's VPT used over 70,000 hours of human gameplay video and 720 V100 GPUs for 9 days — [The Decoder](https://the-decoder.com/deepminds-dreamerv3-collects-minecraft-diamonds-without-human-data/)

**Dreamer 4 (Sept 2025) [preprint, Google DeepMind].**
- First agent to obtain Minecraft diamonds purely from offline data.
- Runs real-time interactive inference at 21 FPS on a single H100.
- Beats the offline VPT agent with about 100x less data: 2,500 hours of video, of which only 100 hours are action-labeled, reaching about 85% performance.
— [arXiv 2509.24527](https://arxiv.org/abs/2509.24527); [InfoQ](https://www.infoq.com/news/2025/10/dreamer-4-minecraft-agent/)

**Genie 3 (Aug 2025, Google DeepMind) [company demo; no paper or compute disclosed].**
- What it does: text-prompted interactive worlds at 720p and 24 fps in real time, consistent for "several minutes", with visual memory reaching back about 1 minute — [DeepMind blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- Limitations:
  - Autoregressive and needs "substantial dedicated compute", so longer sessions are too expensive to scale.
  - Action space is mostly navigation.
  - Multi-agent interaction is weak.
  - Cannot reproduce real-world geography.
  — [TechTalks](https://bdtechtalks.com/2025/08/07/deepmind-genie-3/)

### Inferences
- JEPA's efficiency argument has partial empirical support at three levels:
  - **Pretraining** (I-JEPA versus MAE, about 10x).
  - **Planning/inference** (latent-space planning versus pixel generation: 16x faster and far more successful).
  - **Downstream data** (62 hours of robot data versus Octo's 1M+ trajectories).
- The *total* pipeline is not low-compute: V-JEPA 2 consumed over 1M hours of video. The data-efficiency gain moves to the fine-tuning and control stage.
- Genie 3 is the counter-example. Pixel-generative world models are compute-hungry at inference, which supports LeCun's specific claim that "predict in latent space, not pixels" is the efficient variant.
- Dreamer 3 and 4 show that learned world models plus imagination training cut *environment data* by about 100x at single-GPU scale on a hard open-ended task. This is among the most credible "low compute, general-ish agent" results, though still one domain.
- No JEPA or world-model system has demonstrated language, mathematics or abstract reasoning competitive with LLMs. AMI Labs had not (by Sept 2026) published a system that tests the "beyond LLM" claim.

### Gaps
- V-JEPA 2 total pretraining FLOPs/GPU-hours: not found in the accessible text.
- Genie 3 parameter count and serving compute: not disclosed.
- AMI Labs: no technical results found as of Sept 2026, beyond unverified aggregator claims about V-JEPA 2.1.
- The contents of "When Does LeJEPA Learn a World Model?" (May 2026) could not be read.

---

## 2. Active inference, the free energy principle and predictive coding (VERSES claims versus verified results)

### Takeaway
VERSES' AXIOM shows real, interesting sample and parameter efficiency on simple 2D arcade games: 0.95M versus 420M parameters, and competence within about 10k steps. However, the benchmark, the paper and the headline multipliers come from VERSES itself, with only company-commissioned "third-party" validation. The Mastermind comparisons pit a purpose-built Bayesian solver against general LLMs and say little about general intelligence. Predictive-coding networks now match backprop on small vision benchmarks, but they have not shown a compute advantage on GPUs.

### Cited Findings
**AXIOM (VERSES, arXiv 2505.24784, June 2025) [preprint].**
- Method:
  - Gradient-free and object-centric.
  - Grows and prunes four mixture models via variational updates.
  - Uses Bayesian model reduction and expected-free-energy planning.
  — [arXiv 2505.24784](https://arxiv.org/pdf/2505.24784)
- Headline numbers (company claim, Gameworld 10k, 10 games × 10 seeds):
  - "Up to 60% better" play.
  - 7.6x more sample-efficient (3,175 versus 24,207 steps).
  - 39x faster GPU runtime and 12x cheaper.
  - 400x smaller: 0.95M parameters versus 420M for DreamerV3.
  — [VERSES press release via MarketScreener](https://www.marketscreener.com/quote/stock/VERSES-AI-INC-140070794/news/Verses-AI-Inc-Announces-New-Axiom-Model-Is-Up-to-60-Better-97-More-Efficient-and-Learning-39-Tim-50139233/); [GlobeNewswire, 2025-06-02](https://www.globenewswire.com/news-release/2025/06/02/3091981/0/en/VERSES-Digital-Brain-Beats-Google-s-Top-AI-At-Gameworld-10k-Atari-Challenge.html)
- Qualitative claims: AXIOM often reaches most of its final reward within 5k steps, while BBF and DreamerV3 "failed to achieve competence altogether in 3 of the games" — [VERSES research blog](https://www.verses.ai/research-blog/axiom-mastering-arcade-games-in-minutes-with-active-inference-and-structure-learning)
- Status: an OpenReview PDF exists ("AXIOM: Learning to Play Games in Minutes"), so it was submitted to a conference. I could not confirm acceptance — [OpenReview](https://openreview.net/pdf/607c311748066d425e7d29a692d45b0e6686cdd2.pdf)
- "Third-party" validation came from Soothsayer Analytics, a firm VERSES engaged. This is not an academic replication — [Soothsayer](https://soothsayeranalytics.com/media/soothsayer-validates-a-new-path-in-ai)

**Genius on Mastermind (Feb 2025) [company claim].**
- Versus DeepSeek R1:
  - Solve rate: 100% for Genius versus 45% for R1.
  - Time per game: 1.1–4.5 s versus about 934 s on average.
  - Total for 100 games: about 5 min of compute versus 26 h; $0.05 versus $38.94.
  - Headline: "245× faster, 779× cheaper".
  — [GlobeNewswire 2025-02-04](https://www.globenewswire.com/news-release/2025/02/04/3020351/0/en/VERSES-Genius-Outperforms-DeepSeek-R1-Model-in-Code-Breaking-Mastermind-Challenge.html); [SEC exhibit 99-1](https://www.sec.gov/Archives/edgar/data/1879001/000149315225007173/ex99-1.htm)
- An earlier comparison claimed Genius was 140x faster than OpenAI o1-preview — [Denise Holt (VERSES-aligned commentator)](https://deniseholt.substack.com/p/verses-ai-crushes-openai-o1-in-head)

**Origin of the benchmark programme.**
- In 2024, VERSES said it would "stop debating scientific principles with peers" and instead run benchmarks. It acknowledged that the Bayesian inference behind active inference "has historically been difficult to scale" — [VERSES blog](https://www.verses.ai/blog/blogs/on-upcoming-2024-benchmark-work-from-verses)
- Wider claim: comparable or better performance than state-of-the-art RL with "90% less data" and "a fraction of the compute" on Atari [company claim] — [VERSES Atari release](https://www.verses.ai/news/verses-to-release-atari-benchmark-results-at-world-economic-forum-in-davos)

**Predictive coding networks (PCNs).**
- PCX library (ICLR 2025): PC training scaled to about AlexNet-scale models, with competitive results on CIFAR-100 and Tiny ImageNet. On VGG-5-type models it reaches about 89% on CIFAR-10, comparable to backprop. The paper notes that hyperparameter search dropped from "days to hours", which shows how costly PC training had been — [Pinchetti et al., arXiv 2407.01163](https://arxiv.org/html/2407.01163v1)
- 2025 advances: ResNet-18 on CIFAR-10 at about 92%, "virtually identical to BP", whereas baseline PC "struggle[s] to surpass 20%" on the same architecture — [arXiv 2506.23800](https://arxiv.org/html/2506.23800v1)

### Inferences
- **Why Gameworld 10k is a weak test.**
  - It appears to be a new benchmark introduced alongside AXIOM. I found no evidence that other groups use it.
  - The games are simple object-and-physics 2D arcade tasks that suit AXIOM's object-centric priors.
  - So "60% better than DreamerV3" is a home-field result. It shows that strong structural priors buy large sample efficiency in *matched* domains, not general efficiency.
- **Why the Mastermind comparison says little.** Mastermind has well-known exact solvers, and Knuth's 1977 algorithm always wins within 5 guesses. A Bayesian solver beating a general-purpose LLM is expected and says nothing about general intelligence.
- **What active inference still lacks.** It remains theoretically elegant, meaning it unifies perception, action and curiosity under one objective. But no independent group has shown it scaling to high-dimensional, open-ended domains such as language, 3D vision or robotics at deep-learning scale.
- **Predictive coding helps hardware, not GPUs.** PC's selling point is local, parallel learning suited to neuromorphic or analog substrates. On GPUs it needs iterative inference (many relaxation steps per sample), so it is *more* compute than backprop per example. Any energy win depends on custom hardware that does not yet exist at scale.

### Gaps
- No independent academic replication of AXIOM or Genius results was found.
- AXIOM's peer-review status is unconfirmed.
- No PCN result at transformer/LLM scale (over 1B parameters) was found.

---

## 3. Numenta Thousand Brains Project / Monty

### Takeaway
Monty (open-sourced by the Thousand Brains Project; paper in Neural Computation, May/June 2026) reports enormous training-FLOP savings, from about 1e4–1e5x up to about 5e8x fewer than ViTs. However, this is on a narrow 3D object-recognition and pose task (77 YCB objects in simulation). The code is explicitly "early beta", and there is no evidence yet on language, reasoning or complex manipulation.

### Cited Findings
- Publication: Leadholm, Clay, Knudstrup, Lee & Hawkins, "Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference", Neural Computation 38(6), 2026 [peer-reviewed; preprint arXiv 2507.04494, July 2025] — [DOI 10.1162/neco.a.1508](https://doi.org/10.1162/neco.a.1508); [arXiv 2507.04494](https://arxiv.org/pdf/2507.04494)
- Method: Monty replicates semi-independent "cortical column" learning modules. They use associative, Hebbian-like binding within object reference frames, which enables rapid and continual learning — [arXiv 2507.04494](https://arxiv.org/pdf/2507.04494)
- Setup and accuracy:
  - Setup: 77 YCB objects × 14 rotations.
  - After only 1 view per object, Monty reaches about 62% accuracy versus about 40% for a from-scratch ViT (chance is 1.3%).
  - Monty is also reported to reach higher accuracy per inference FLOP — [EmergentMind summary](https://www.emergentmind.com/topics/thousand-brains-systems)
- Training FLOPs (the sources conflict):
  - One summary says about 340,000x fewer than a from-scratch ViT and about 528,000,000x fewer than a pretrained plus fine-tuned ViT — [EmergentMind](https://www.emergentmind.com/topics/thousand-brains-systems)
  - A search extract of the paper says "thirty-three thousand times less" than a ViT and "five hundred and twenty seven million times less" versus pretraining plus fine-tuning — [arXiv 2507.04494 via search](https://arxiv.org/pdf/2507.04494)
  - Treat the figures as order-of-magnitude only.
- Code status: "early beta … not production-ready code … Expect frequent changes" — [GitHub tbp.monty](https://github.com/thousandbrainsproject/tbp.monty)
- Project manifesto: "The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence" (Dec 2024) — [arXiv 2412.18354](https://arxiv.org/abs/2412.18354)

### Inferences
- The FLOP ratios are real but flattering. They compare a structured, pose-aware, few-object system with rich 3D sensorimotor inputs against general image classifiers, on a task designed around Monty's inductive biases (reference frames, movement).
- Monty is best read as evidence that **explicit structure (reference frames) plus local associative learning can make *learning* nearly free for the domains it fits**. That is the right *kind* of result for the "extremely low compute across training and inference" question. But capability breadth is far below general intelligence.
- There is no demonstration of compositional or abstract (non-spatial) knowledge. Hawkins' theory claims this is possible, but it is untested.

### Gaps
- No independent replication.
- No benchmark outside YCB object and pose recognition.
- Roadmap details are held on docs.thousandbrains.org (blocked).
- Real-robot results through 2026 were not found.

---

## 4. Developmental / child-like learning (BabyLM; human data budgets)

### Takeaway
Children reach adult-like linguistic competence on about 3e7–1.1e8 words, 4–5 orders of magnitude less than LLMs. Three rounds of BabyLM show that architecture and objective tweaks at about 1e8 words can match or beat billion/trillion-token models on *grammatical* benchmarks. They do not close the gap in knowledge or reasoning. Curriculum learning mostly failed. Performance still tracks training FLOPs, so data efficiency is largely bought with multi-epoch compute. Multimodal and interactive "child-like" training has so far not helped.

### Cited Findings
**Human data budget.**
- Children are exposed to about 3–11M words per year (Hart & Risley range), so about 30–110M words by age 10, when they have adult-like competence. LLMs receive about 4–5 orders of magnitude more — [Frank 2023, Trends in Cognitive Sciences](https://www.sciencedirect.com/science/article/abs/pii/S1364661323002036)
- A Behavioral and Brain Sciences commentary argues the "data gap" framing misses that children create languages, violate their input statistics and have critical periods, so human learning is not just efficient statistics — [BBS commentary](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/beyond-the-data-gap-children-create-languages-violate-their-input-statistics-and-exhibit-critical-periods/2BF9E7E520D6A309C3C18E8835315AD9)

**BabyLM 2023 (CoNLL 2023) [peer-reviewed findings].**
- Tracks: Strict (100M words) and Strict-small (10M).
- The winner, ELC-BERT (built on LTG-BERT), plus LTG-BERT itself outperformed every other submission *and* the Llama 2 and RoBERTa-base "skylines" (trained on far more data) on overall score and on all test suites (BLiMP, (Super)GLUE, MSGS).
- LTG-BERT's gains come from standard transformer engineering: extra layer norms, GEGLU, DeBERTa-style disentangled attention and scaled initialization.
- Curriculum learning, a popular approach, was largely unsuccessful.
— [Findings 2023, ACL](https://aclanthology.org/2023.conll-babylm.1.pdf); [arXiv 2504.08165](https://arxiv.org/html/2504.08165v1)
- Distillation: ensemble-distilled small students (BabyLlama-2) can outperform their teachers at limited data — [arXiv 2409.17312](https://arxiv.org/html/2409.17312v1)

**BabyLM 2024 [peer-reviewed findings].**
- 31 papers and 64 models.
- GPT-BERT, a hybrid causal/masked objective, won both 100M- and 10M-word text tracks.
- Multimodal track: only 3 teams and 8 models, and none beat the baselines, so no winner. Adding image-text pairs within the 100M budget did not help.
- There was "a strong relationship between training FLOPs and average performance".
— [Findings 2024, arXiv 2412.05149](https://arxiv.org/abs/2412.05149); [GPT-BERT, arXiv 2410.24159](https://arxiv.org/pdf/2410.24159)

**BabyLM 2025 (first BabyLM Workshop, EMNLP 2025) [peer-reviewed findings].**
- Tracks: Strict (100M), Strict-small (10M), and a new Interaction track in which students (still ≤100M words) may learn from pretrained teacher models via RLHF/PPO or SimPO-style preference baselines.
- Baselines included GPT-BERT and GPT-2 Small.
- Findings: submissions beat baselines in the Strict-small and Interaction tracks; new objectives and architectures work best; interaction with teacher models can yield high-quality LMs.
- Over 30 submissions.
— [Findings of the Third BabyLM Challenge](https://aclanthology.org/2025.babylm-main.28/); [2025 call for papers, arXiv 2502.10645](https://arxiv.org/pdf/2502.10645)
- A 2025 paper found that "Dialogue Is Not Enough to Make a Communicative BabyLM (But Neither Is Developmentally Inspired Reinforcement Learning)" — [arXiv 2510.20358](https://arxiv.org/pdf/2510.20358)
- BabyLM 2026 goes multilingual ("BabyLM Turns 4") — [arXiv 2602.20092](https://arxiv.org/pdf/2602.20092)

### Inferences
- **Grammar is cheap; world knowledge and reasoning are not.** About 1e8 words suffices for near-adult *syntactic* competence in models (BLiMP-type), matching the human budget. But BabyLM models do not show broad knowledge or multi-step reasoning. The human advantage probably comes from embodied multimodal grounding, social interaction and innate priors that nobody has yet reproduced (multimodal and RL-interaction tracks have failed so far).
- **Data efficiency is not compute efficiency.** The 2024 FLOPs–performance correlation means BabyLM winners trade data for many epochs. For a "low total compute" goal, data efficiency alone is insufficient.
- **"Learning like a child" helps in two ways, one proven and one not.** Proven so far: inductive bias and objective design (e.g., GPT-BERT, LTG-BERT). Not proven: curricula or developmental staging.

### Gaps
- The exact 2025 winners and scores could not be read (ACL Anthology blocked). I could not confirm whether 2025 rules capped training epochs.
- No solid numbers on how far 100M-word models fall behind trillion-token models on reasoning or knowledge benchmarks such as MMLU.
- Evidence on embodied, child-like robot learning projects at scale was not gathered in this pass.

---

## 5. Neuromorphic and spiking systems

### Takeaway
Neuromorphic hardware delivers real but usually 2x–70x energy gains at *inference* for suitably sparse, quantized or recurrent models: Loihi 2 LLM about 2x less energy than an edge GPU; NorthPole about 73x tokens/s/W versus an H100-class GPU at low latency (IBM claim). None of these systems reduces the cost of *learning*: the models are trained or converted on GPUs. "Spiking LLMs" such as SpikingBrain are mostly efficient-attention transformers upcycled from Qwen and run on GPUs.

### Cited Findings
**Intel Hala Point (April 2024) [company claim].**
- Scale: 1,152 Loihi 2 chips (Intel 4 process), 1.15B neurons, 128B synapses, 140,544 cores, 2,600 W maximum.
- Throughput and efficiency: 20 POPS; 15 TOPS/W at INT8 on 10:1 sparse networks.
- Deployed at Sandia.
— [Intel newsroom](https://www.intel.com/content/www/us/en/newsroom/news/intel-builds-worlds-largest-neuromorphic-system.html)

**Loihi 2 LLM (ICLR 2025 workshop, Mar 2025) [workshop paper].**
- A 370M-parameter MatMul-free LLM was quantized with no accuracy loss.
- It achieves up to 3x higher throughput with 2x less energy than transformer LLMs on an edge GPU — [arXiv 2503.18002](https://arxiv.org/abs/2503.18002)

**"Loihi 3" (Jan 2026) [unverified].**
- Claims: 4 nm, 8M neurons and 64B synapses per chip, graded spikes, "100x" efficiency.
- These appear only in syndicated "TokenRing" articles. I found no Intel primary source — [FinancialContent](https://www.financialcontent.com/article/tokenring-2026-1-19-the-brain-like-revolution-intels-loihi-3-and-the-dawn-of-real-time-neuromorphic-edge-ai)

**IBM NorthPole.**
- Science 2023 [peer-reviewed]: an inference-only, near-memory digital architecture — [Science](https://www.science.org/doi/10.1126/science.adh1174)
- HPEC, Sept 2024 [IBM claim] — 3B-parameter LLM (derived from Granite-8B-Code):
  - Throughput: 16 NorthPole cards deliver 28,356 tokens/s system throughput.
  - Latency: under 1 ms/token.
  - Power: 672 W in 2U.
  - Energy efficiency: 72.7x better (tokens/s/W) than a 4 nm GPU at low latency.
  - Latency at matched efficiency: 46.9x better than a 5 nm GPU.
  — [IBM Research blog](https://research.ibm.com/blog/northpole-llm-inference-results); [modha.org](https://modha.org/2024/09/breakthrough-low-latency-high-energy-efficiency-llm-inference-performance-using-northpole/)
- A scalable, vertically integrated NorthPole LLM system followed in Nov 2025 — [arXiv 2511.15950](https://arxiv.org/pdf/2511.15950)

**SpiNNaker2 / SpiNNcloud (2025).**
- Deployments:
  - Sandia, June 2025: 150–180M neurons.
  - UT San Antonio (NSF THOR "Neuromorphic Commons"), Nov 2025: over 393M neurons.
  — [Next Platform](https://www.nextplatform.com/2025/06/16/sandia-deploys-spinnaker2-neuromorphic-system/); [HPCwire](https://www.hpcwire.com/2025/11/12/spinnclouds-spinnaker2-deployed-at-ut-san-antonio-for-nsf-funded-thor-project/)
- Company claim: "18× more efficient than GPUs" — [HPCwire (press release)](https://www.hpcwire.com/off-the-wire/sandia-deploys-spinnaker2-neuromorphic-system-from-spinncloud/)

**Field review.**
- "Neuromorphic computing at scale" (Kudithipudi et al., Nature 637, Jan 2025) [peer-reviewed review]: the field is at a "critical juncture". It needs scalable architectures, software ecosystems and killer applications. Expect a range of hardware for different niches (edge, size/weight/power-constrained) rather than one general solution — [Nature](https://www.nature.com/articles/s41586-024-08253-8)

**SpikingBrain 1.0 (BICLab/CASIA, Sept 2025; TMLR 2026) [peer-reviewed per GitHub].**
- Models: 7B (linear) and 76B (hybrid-linear MoE), trained on Chinese MetaX GPUs.
- Training: conversion-based continual pretraining with "<2% of the data", about 150B tokens starting from Qwen2.5-7B.
- Quality: about 90% of base-model performance.
- Speed and sparsity: over 100x time-to-first-token speedup at 4M tokens; about 69% spike sparsity.
- Estimated energy savings (theoretical): 43.5x versus FP16 MACs and 6.8x versus INT8 MACs.
— [arXiv 2509.05276](https://arxiv.org/abs/2509.05276); [GitHub](https://github.com/BICLab/SpikingBrain-7B)
- Critique: it is "an upcycled Qwen2.5-7B checkpoint … with pseudo-spiking activation". The gains come from attention re-engineering plus sparsity on conventional GPUs, with "no event-driven GPU kernel; spikes are encoded tensors" — [akadata critique](https://articles.akadata.co.uk/past-the-hype-spikingbrain-7b-qwen2-5-7b-in-new-clothes/)

**SpikingBrain 2.0 (Apr 2026) [preprint].**
- Models: 5B and VL-5B, recovering most of Qwen3-4B's capability with under 7k A100 GPU-hours.
- Long context: 10.13x TTFT speedup at 4M context; over 10M tokens on 8 A100s.
- Neuromorphic path (simulated or ASIC): 64% sparsity; 70.6% area and 46.5% power reduction.
— [arXiv 2604.22575](https://arxiv.org/abs/2604.22575)

### Inferences
- **Hala Point versus GPUs.** 15 TOPS/W (INT8, 10:1 sparse) is about 3–5x a modern GPU's dense INT8 efficiency (H100 is about 2.8 TOPS/W dense INT8; my estimate from public specs), and only under favourable sparsity. That is not the "orders of magnitude" often implied.
- **NorthPole's per-token energy.** 672 W / 28,356 tok/s ≈ 0.024 J/token for a 3B model. This is excellent, but it depends on all weights fitting on-chip (model-size limited) and helps inference only.
- **The central limitation is learning.** Every large neuromorphic or spiking LLM reviewed here is trained with backprop on GPUs, then converted. On-chip learning remains toy-scale. So neuromorphic computing is an *inference-energy* lever of about 1–2 orders of magnitude, not a route to cheap training of general intelligence.
- **Brain gap for context.** The brain runs at about 20 W. GPT-4 training was estimated at 5–50 GWh, but that estimate is from low-quality sources — [Ideasthesia](https://www.ideasthesia.org/brain-efficiency-20-watts/). The remaining brain-versus-silicon gap is dominated by algorithm and learning efficiency, not just switching energy.

### Gaps
- There are no audited, apples-to-apples per-joule benchmarks (such as MLPerf) of neuromorphic versus GPU on general LLM workloads.
- Loihi 3 specifications are unconfirmed.
- Measured (not estimated) energy for SpikingBrain on real neuromorphic silicon was not found.

---

## 6. Analog, in-memory, optical, reversible and thermodynamic computing

### Takeaway
Physics-based substrates show credible 10x-level (sometimes about 100x+) energy-per-operation gains for matrix-vector or attention kernels in lab chips or simulations. The 10³–10⁴x headlines (Extropic 10,000x, gain-cell attention 70,000x, Normal 1,000x) are simulation or kernel-level figures on narrow tasks. None has trained a frontier-scale model. Maturity ranges from test chip (Vaire, Normal CN101, Extropic X0/Z1) to working inference prototypes (IBM analog PCM, Lightmatter photonic).

### Cited Findings
**Analog in-memory (IBM).**
- Nature, Aug 2023 [peer-reviewed]:
  - Hardware: 35M phase-change memory devices across 34 tiles (14 nm).
  - Efficiency: up to 12.4 TOPS/W sustained.
  - Speech result: word error rate 9.258%.
  — [Nature s41586-023-06337-5](https://www.nature.com/articles/s41586-023-06337-5)
- HERMES chip: 64-core analog in-memory computing, 63.1 TOPS at 9.76 TOPS/W for 8-bit matrix-vector multiplies — [IBM Research](https://research.ibm.com/blog/analog-ai-chip-inference)

**Gain-cell analog attention (Leroux et al., Nature Computational Science, Sept 2025) [peer-reviewed; largely circuit and simulation].**
- Results: up to 70,000x lower attention energy and 100x speed-up versus GPUs, i.e., 2 and 4 orders of magnitude lower latency and energy for the *attention* part.
- Needs a special initialization algorithm to reach GPT-2-level text performance, because non-idealities prevent mapping pretrained models directly.
— [Nature Comp. Sci.](https://www.nature.com/articles/s43588-025-00854-1)

**Photonic (Lightmatter, Nature, Apr 2025) [peer-reviewed].**
- Hardware: 4 × 128×128 photonic tensor cores (GF 90 nm photonics).
- Throughput and power: 65.5 trillion 16-bit-equivalent ops/s at about 78 W electrical plus 1.6 W optical.
- Precision: 7–10-bit effective.
- Accuracy: ResNet on ImageNet 79.3–79.7%, CIFAR-10 86.4%; BERT-tiny on IMDB 83.2%; also runs Atari deep RL.
— [Nature s41586-025-08854-x](https://www.nature.com/articles/s41586-025-08854-x); [Physics World](https://physicsworld.com/a/photonic-computer-chips-perform-as-well-as-purely-electronic-counterparts-say-researchers/)

**Thermodynamic (Extropic).**
- X0 chip (2025): demonstrated "pbits" that turn transistor thermal noise into programmable randomness — [Extropic TSU 101](https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware)
- Denoising Thermodynamic Models on modelled TSUs matched GPU diffusion models on Fashion-MNIST at about 1/10,000 the energy — *in simulation*, with a 70×70 grid of locally connected sampling cells — [arXiv 2510.23972](https://arxiv.org/html/2510.23972v1)
- Z1 chip: reported for early 2026 with 250k pbits [company claim] — [Extropic](https://extropic.ai/writing/from-one-to-one-billion/)
- Independent skeptic: TSUs are "a highly specialized, physics-native sampler rather than a GPU replacement". 10–100x looks realistic; 100–10,000x only for small, well-matched tasks — [ThothHermes Substack](https://thothhermes.substack.com/p/will-extropics-tsus-beat-gpus-on)

**Thermodynamic (Normal Computing).**
- CN101 tape-out on Aug 12 2025, called the "first thermodynamic computing chip". Targets linear algebra, sampling and diffusion inference, claiming "up to 1000×" efficiency on targeted workloads [company claim; characterization pending]. CN201/CN301 are planned — [Normal Computing](https://www.normalcomputing.com/blog/normal-computing-announces-tape-out-of-worlds-first-thermodynamic-computing-chip)

**Reversible (Vaire).**
- Ice River test chip (22 nm, 2025): net energy recovery factor of 1.77 for a resonator-driven capacitor array and 1.41 for a shift register (above 1 counts as proof of concept).
- Target: 99.97% recovery with a future MEMS resonator.
— [EE Times](https://www.eetimes.com/vaire-demos-energy-recovery-with-reversible-computing-test-chip/)

### Inferences
- **Lightmatter check.** 65.5 TOPS / about 80 W ≈ 0.8 TOPS/W. That is *not* better than a modern GPU at low precision (an H100 is about 1.4 TFLOPS/W BF16 dense and about 2.8 FP8, from public specs). So secondary claims that it is "10x more efficient than GPUs" are not supported by the Nature numbers; the Nature result is about accuracy parity and generality, not efficiency. Photonics' credible near-term win is interconnect (e.g., Lightmatter Passage), not compute.
- **Analog and thermodynamic gains are kernel-level.** Amdahl's law limits the end-to-end system gain: digital periphery, data movement and ADC/DAC often dominate. A 70,000x attention gain may become about 10x system-level.
- **Training remains the weak spot.** All these substrates target inference or sampling. Analog training suffers from device non-idealities; thermodynamic sampling suits energy-based and diffusion models, not transformer training.
- **Realistic ceiling.** Hardware alone plausibly gives 1–2 orders of magnitude of energy per useful operation over the next several years, versus the roughly 10⁵–10⁶x brain-versus-GPU gap often cited. Most of the remaining gap has to come from algorithms and representations.

### Gaps
- No measured Z1 or CN101 benchmark results were found as of Sept 2026.
- No analog or optical system has been shown training a model over 1B parameters.
- No independent head-to-head energy audits were found.

---

## 7. Other paradigms: hyperdimensional computing, liquid networks, local learning rules, CLS/continual learning, small recursive reasoners

### Takeaway
None of these has shown general capability beyond deep learning.
- Liquid AI's LFM2 is an efficient *deployment* architecture trained on 10–12T tokens, so it is not data- or training-efficient.
- Forward-forward and other local rules still underperform backprop.
- HDC is efficient but narrow.
- Continual-learning architectures (Google's Nested Learning / Hope) are promising but early.
- The most striking "tiny compute" results are recursive reasoners (HRM 27M, TRM 7M parameters) on ARC-AGI. Independent analysis shows their gains come from iterative refinement and augmentation rather than brain-like hierarchy, and they are task-specific.

### Cited Findings
**Liquid AI LFM2 (tech report Nov 2025).**
- Models: 350M, 700M, 1.2B and 2.6B dense, plus an 8.3B MoE with 1.5B active parameters. All use 32K context.
- Architecture: gated short convolutions plus sparse attention.
- Training: pretrained on **10–12T tokens**, with distillation.
- Results: LFM2-2.6B scores IFEval 79.56% and GSM8K 82.41%.
— [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- Speed:
  - 2x faster decode and prefill on CPU than Qwen3.
  - LFM2 decodes at 213 tok/s on a Galaxy S25 Ultra and 42 tok/s on a Raspberry Pi 5.
  - LFM2.5-8B-A1B decodes at 253 tok/s on an M5 Max.
  — [Liquid AI blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models); [LFM2.5 blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b)

**Forward-forward (Hinton 2022) and follow-ups.**
- FF replaces backprop with two forward passes and a layer-local "goodness" objective — [Hinton FF paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
- 2025 CNN work trains deeper FF CNNs on CIFAR-10, but FF remains "slower than backpropagation and lacks generalization" — [PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12586560/); [arXiv 2504.11229](https://arxiv.org/pdf/2504.11229)

**Hyperdimensional computing / vector-symbolic architectures.**
- Efficient, noise-tolerant and hardware-friendly. For example, the QuantHD FPGA design reports 42.3x energy efficiency and 4.7x speedup versus prior HDC.
- Applications are mostly classification and biosignals, with accuracy trade-offs — [PMC review 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192801/); [ACM Computing Surveys](https://dl.acm.org/doi/10.1145/3538531)

**Complementary learning systems / continual learning.**
- Google's Nested Learning (NeurIPS 2025) treats architecture and optimizer as nested learning levels updating at different rates.
- Its "Hope" architecture (a self-modifying Titans variant plus a Continuum Memory System) targets catastrophic forgetting. It beat Titans, Samba and a Transformer on language modeling and common-sense reasoning in the paper's small-scale tests — [Google Research blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- Monty's associative binding is another CLS-like continual learner (see section 3).

**HRM (Sapient, June 2025) [preprint].**
- Paper claims: 27M parameters, about 1,000 training tasks, 41% on ARC-AGI-1 (public) — [arXiv 2506.21734](https://arxiv.org/abs/2506.21734)
- ARC Prize independent verification:
  - Semi-private scores: 32% on ARC-AGI-1 and 2% on ARC-AGI-2.
  - Ablation: the outer refinement loop is the key driver, taking accuracy from 18.6% to 35.5% with one loop. A plain transformer with the same pipeline gets similar gains, so the "brain-inspired" hierarchy contributes little.
  — [ARC Prize analysis](https://arcprize.org/blog/hrm-analysis)

**TRM (Samsung SAIL Montreal, Oct 2025) [preprint].**
- 7M parameters and 2 layers, recursing up to 16 improvement steps.
- Scores: 45% on ARC-AGI-1 and 8% on ARC-AGI-2. Reported comparisons: Gemini 2.5 Pro 37.0%, o3-mini-high 34.5%, DeepSeek R1 15.8% on ARC-AGI-1.
— [arXiv 2510.04871](https://arxiv.org/pdf/2510.04871); [GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels); [ARC Prize 2025 technical report](https://arxiv.org/pdf/2601.10904)

### Inferences
- **Recursion and test-time refinement give large per-parameter gains.** HRM and TRM show that recursive latent refinement (iterative "thinking" in a small network) plus heavy augmentation gives large *per-parameter* gains on abstract puzzles. This echoes cortical recurrence, but the gain is algorithmic, not neuro-mimetic.
- **They are narrow.** They are trained per benchmark distribution, have no language or world knowledge, and are not general.
- **Efficient deployment is the only commercially mature line.** Liquid and LFM-style hybrids trade architecture for inference speed but still depend on trillion-token pretraining.
- **Backprop-free learning rules have not closed the accuracy gap at scale.** Their efficiency case depends on hypothetical local-learning hardware.

### Gaps
- Hope/Nested Learning model sizes and absolute scores were not retrieved.
- No large-scale, ImageNet-level or LLM-level forward-forward or Hebbian results were found.
- TRM's semi-private ARC verification numbers were not retrieved.

---

## 8. Comparative assessment: which paradigm has the strongest evidence of beating mainstream deep learning on capability per FLOP or per joule for *general* tasks?

### Takeaway
None of the alternative paradigms has demonstrated better capability per FLOP or per joule than mainstream deep learning on *general* tasks as of Sept 2026. The strongest, independently checkable efficiency evidence comes from paradigms that stay *inside* deep learning:
- JEPA and latent world models, with 10–100x data and compute savings on video, robotics and Minecraft;
- recursive small reasoners on ARC;
- efficient inference hardware (NorthPole, analog, neuromorphic), with 2–70x energy gains, inference only.

The radical brain-like paradigms (active inference/AXIOM, Thousand Brains/Monty) show the largest multipliers (10²–10⁸x). But they are company-run, on narrow, prior-matched tasks, and not replicated.

### Cited Findings
- **Largest verified algorithmic efficiency, general-ish agents.**
  - DreamerV3: diamonds from scratch on 1 GPU, versus VPT's 720 V100 × 9 days plus 70k hours of video — [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2); [The Decoder](https://the-decoder.com/deepminds-dreamerv3-collects-minecraft-diamonds-without-human-data/)
  - Dreamer 4: about 100x less data than VPT, offline — [arXiv 2509.24527](https://arxiv.org/abs/2509.24527)
- **Largest verified self-supervised pretraining efficiency.** I-JEPA is over 10x more efficient than MAE at ViT-H — [arXiv 2301.08243](https://arxiv.org/pdf/2301.08243)
- **Latent versus pixel world models for planning.** 16 s versus 4 min per action, with 60–80% versus 0–20% success — [EmergentMind on V-JEPA 2-AC](https://www.emergentmind.com/topics/v-jepa-2-ac)
- **Largest claimed training-FLOP reduction.** Monty: about 1e4–1e5x versus a ViT from scratch and about 5e8x versus a pretrained ViT, on 77 YCB objects — [EmergentMind](https://www.emergentmind.com/topics/thousand-brains-systems); [Neural Computation 2026](https://doi.org/10.1162/neco.a.1508)
- **Largest claimed RL efficiency.** AXIOM: 400x fewer parameters and 39x less GPU runtime than DreamerV3 on VERSES' own Gameworld 10k — [MarketScreener/VERSES](https://www.marketscreener.com/quote/stock/VERSES-AI-INC-140070794/news/Verses-AI-Inc-Announces-New-Axiom-Model-Is-Up-to-60-Better-97-More-Efficient-and-Learning-39-Tim-50139233/)
- **Largest inference energy gains on real LLM workloads.**
  - NorthPole: 72.7x tokens/s/W versus a GPU at low latency (3B LLM; IBM) — [IBM](https://research.ibm.com/blog/northpole-llm-inference-results)
  - Loihi 2: 2x less energy than an edge GPU (370M model) — [arXiv 2503.18002](https://arxiv.org/abs/2503.18002)
- **Largest per-parameter reasoning efficiency.**
  - TRM: 7M parameters at 45% on ARC-AGI-1 — [arXiv 2510.04871](https://arxiv.org/pdf/2510.04871)
  - HRM's gains were independently attributed to refinement loops rather than brain-like hierarchy — [ARC Prize](https://arcprize.org/blog/hrm-analysis)
- **Data efficiency of child-scale learning.** Achieved for grammar (BabyLM winners beat Llama 2 skylines on BLiMP/GLUE-type suites), but the gains correlate with more training FLOPs and do not extend to multimodal or interactive learning — [BabyLM 2023](https://aclanthology.org/2023.conll-babylm.1.pdf); [BabyLM 2024](https://arxiv.org/abs/2412.05149)

### Inferences
Evidence scorecard (author's synthesis; "Gen." is generality of the demonstrated capability):

| Paradigm | Claimed efficiency lever | Best demonstrated gain | Independent verification | Gen. | Helps training? | Helps inference? |
|---|---|---|---|---|---|---|
| JEPA / latent world models (V-JEPA 2, LeJEPA; AMI Labs) | Predict in representation space; plan in latent space | ~10x pretraining (I-JEPA); 16x planning; 62 h robot data | Peer-reviewed (I-JEPA), open weights (V-JEPA 2) | Medium (vision, robotics) | Yes (moderate) | Yes |
| Dreamer-style model-based RL | Learn by imagination | 1 GPU vs 720 GPUs (VPT); ~100x less data | Nature 2025 | Medium (games, control) | Yes | Moderate |
| Genie-style pixel world models | Simulation for agents | — | — | Medium | No (costly) | No (costly) |
| Active inference (AXIOM / Genius) | Bayesian structure learning, gradient-free | 400x parameters, 39x runtime on toy games | Company-only | Low | Yes (in-domain) | Yes (in-domain) |
| Thousand Brains / Monty | Cortical columns, reference frames, Hebbian binding | 1e4–1e8x fewer training FLOPs on YCB | Peer-reviewed, self-run | Low | Yes (in-domain) | Yes |
| Developmental (BabyLM) | Human-scale data, better objectives | Grammar at 1e8 words ≈ LLM | Community challenge | Low–Med (syntax) | Data yes; compute no | — |
| Neuromorphic / SNN | Sparse, event-driven, in-memory | 2–73x energy/token | Mostly vendor | N/A (substrate) | No | Yes |
| Analog / optical / thermodynamic / reversible | Physics-native ops | 10x (lab) to 1e4x (simulation, kernel) | Mixed | N/A (substrate) | Mostly no | Yes (kernels) |
| Recursive tiny reasoners (HRM/TRM) | Iterative latent refinement | 7M parameters ≈ frontier LLMs on ARC-1 | ARC Prize partially | Low (puzzles) | Yes | Yes |
| Local learning (PC, FF, HDC) | No backprop | Parity at CIFAR scale (PC) | Academic | Low | Not on GPUs | HDC yes (narrow) |

- **The most likely low-compute route is a hybrid, not one paradigm.** It would combine four things:
  - (a) self-supervised latent world models (JEPA/Dreamer) that replace much text-token pretraining with predictive learning from sensorimotor streams and plan in latent space;
  - (b) structured inductive biases (object- and reference-frame-centric, as in AXIOM and Monty) and recursive test-time refinement (as in TRM) that give large sample- and parameter-efficiency in matched domains;
  - (c) continual, local or fast-weight memory (CLS/Nested Learning) so that learning is incremental rather than retrain-from-scratch;
  - (d) sparse, low-precision, near-memory inference hardware (NorthPole/Loihi/analog) giving about 1–2 orders of magnitude of energy.

  The radical paradigms supply (b) and (c) as ideas, but have not yet shown they scale to general domains.
- **The gap the report should stress.**
  - Every paradigm that beats deep learning by huge factors does so on narrow, prior-matched tasks.
  - Every paradigm that works on general tasks is still gradient-trained deep learning, with efficiency gains of about 10–100x rather than the roughly 10⁵–10⁶x brain gap.
  - The unsolved problem is getting brain-like *learning* efficiency (about 1e8 words, about 20 W) for *general* capability. No paradigm has solved it as of Sept 2026.
- **Where the hardware ceiling sits.** Hardware alternatives mainly cut *inference* energy. Because training is a one-off cost while inference is amortized over many uses, a low-total-compute superintelligence would still need an algorithmic breakthrough in learning efficiency. Better chips alone will not get there.

### Gaps
- No standardized, independent capability-per-joule benchmark spans these paradigms.
- AMI Labs and other JEPA-scaling efforts have not published results that test whether JEPA surpasses LLMs in language or reasoning.
- No independent replication of AXIOM or Monty efficiency multipliers.
- Measured silicon results for thermodynamic chips (Z1, CN101) and "Loihi 3" were not found as of Sept 2026.
