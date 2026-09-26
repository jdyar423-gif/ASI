# How small can the innate prior of a general learner be? Genome bottleneck, meta-learned/evolved learning rules, reward/loss curricula, developmental data efficiency (status as of 26 Sept 2026)

**Method note for the report writer.** This session's egress proxy blocked arxiv.org, pnas.org, pmc.ncbi.nlm.nih.gov, biorxiv.org, nature.com, science.org, aclanthology.org, openreview.net, huggingface.co, gwern.net, cshl.edu and most blogs. Most numbers below therefore come from search-engine extracts of the primary source and are marked **(snippet)**. Items read directly from GitHub-hosted repositories or project pages are marked **(repo)**. One number I measured myself: the parameter count of the released Disco103 meta-network, from its weight file, marked **(measured)**. Arithmetic marked **(my calc)** is mine. Status tags: **[PR]** = peer-reviewed; **[PP]** = preprint or self-reported. Background already in `../极低算力超级智能路径/theory_limits.md` (lifetime anchor ~1e24 FLOP, evolution anchor ~1e41 FLOP, LeCun's visual-bandwidth argument, Byrnes' ~1e11-bit learned-information figure), in `brain_inspired_alternatives.md` §4 (BabyLM 2023–2025 overview, FLOPs–performance correlation, AXIOM) and in `skeptics_and_forecasts.md` (Marblestone's "~50 MB genome / Python code generating a curriculum" quotes) is **not** repeated. VeLO's cost and critique were covered in `self_improvement.md`; below I only convert it to FLOP.

---

## 1. Genome information budget and the genomic bottleneck: how many bits can the innate prior contain?

### Takeaway
The genome caps the human innate prior at about 6e9 bits raw. Only about 8% of it is under selection, so the realistic cap is ≲5e8 bits (≈64 MB). The brain-specific *algorithmic* part (architecture, learning rules, reward/steering circuitry) is plausibly 1e6–1e8 bits. Specifying the brain's ~1e14 connections explicitly would take ~4e15 bits, so the genome falls at least 6 orders of magnitude short. It must therefore encode rules and reward circuitry, not weights. The best ML demonstration of "compressing weights through a genome" (Shuvaev et al., PNAS 2024) reaches 1e2–1e3× compression. That gives a head start at "birth" but **no faster learning**. So the human prior is small in bits, but the ML stand-in for it has not been shown to produce a better *learner*.

### Cited Findings
**The bottleneck arithmetic (Zador 2019, Nature Communications, Aug 2019) [PR]**
- The human genome has about 3×10⁹ nucleotides, so it can encode no more than about 1 GB. The brain has about 10¹¹ neurons and >10³ synapses per neuron. Specifying one connection target takes about log₂10¹¹ ≈ 37 bits, so all 10¹⁴ connections would take about 3.7×10¹⁵ bits. "Even if every nucleotide… were devoted to efficiently specifying brain connections, the information capacity would still be at least six orders of magnitude too small." The genome must therefore specify "rules for wiring up the brain" (snippet) — [Zador 2019, bioRxiv full text](https://www.biorxiv.org/content/10.1101/582643v1.full); [CSHL repository](https://repository.cshl.edu/id/eprint/38307/)
- Zador's central claim: "much of an animal's behavioral repertoire is not the result of clever learning algorithms… but arises instead from behavior programs already present at birth". Animals are "born with highly structured brain connectivity, which enables them to learn very rapidly" (snippet) — [Zador 2019](https://www.researchgate.net/publication/335310161_A_critique_of_pure_learning_and_what_artificial_neural_networks_can_learn_from_animal_brains)

**How much of the genome is functional? (Rands, Meader, Ponting & Lunter, PLOS Genetics, July 2014) [PR]**
- 8.2% (7.1–9.2%) of the human genome is under negative selection and therefore likely functional. Only 2.2% has kept its constraint in both human and mouse since they diverged. Protein-coding sequence has a constraint half-life of more than a billion years; lncRNA loci turn over fastest (snippet) — [PLOS Genetics](https://journals.plos.org/plosgenetics/article?id=10.1371%2Fjournal.pgen.1004525)

**Learned vs innate capacity**
- Estimates of the brain's information storage cluster at about 10–100 TB, with a full range of about 1 TB to 2.5 PB (snippet) — [AI Impacts: Information storage in the brain](https://aiimpacts.org/information-storage-in-the-brain/)

**ML test of the bottleneck: Shuvaev, Lachi, Koulakov & Zador, "Encoding innate ability through a genomic bottleneck" (PNAS 121(38), Sept 2024; bioRxiv since Mar 2021) [PR]**
- Setup: a small "g-network" (genome) generates the weight matrix of a larger "p-network" (phenotype), framed as lossy compression of the weights. The p-network has 5 layers, each compressed by its own g-network (snippet) — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2409160121); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11420173/)
- MNIST (2-layer fully connected): the "GN30" setting is **322-fold** compression, and "performance is excellent even with **1,038-fold** compression" (snippet) — [bioRxiv v2](https://www.biorxiv.org/content/10.1101/2021.03.16.435261v2.full)
- CIFAR-10: at **92-fold** compression, initial ("at birth") performance is 76% against 10% for a naive network. The snippet does not say whether 76% is absolute accuracy or a fraction of the trained network's accuracy (snippet) — [bioRxiv v2 PDF](https://www.biorxiv.org/content/10.1101/2021.03.16.435261v2.full.pdf)
- RL (MuJoCo HalfCheetah): after only **4.9-fold** compression, episode-0 performance approaches asymptotic performance, with a modest decline at higher compression (snippet) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11420173/)
- **Key negative:** "genomic compression did not affect the learning trajectory; the only speedup was due to the higher initial performance" (snippet) — [bioRxiv v2 PDF](https://www.biorxiv.org/content/10.1101/2021.03.16.435261v2.full.pdf)
- Transfer: for complex (not simple) problems the bottleneck captures essential circuit features and improves transfer. When CIFAR-10 is transferred to SVHN by initializing only the first two layers from the g-network, 50% performance is reached in about **3.2 epochs vs 4.8** for direct weight transfer (≈1.5× faster) (snippet) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11420173/); [CSHL press release via ScienceDaily, Nov 2024](https://www.sciencedaily.com/releases/2024/11/241125125041.htm)

**Architecture alone as a prior: Weight Agnostic Neural Networks (Gaier & Ha, 2019) [PR, NeurIPS 2019]**
- Evolved topologies with a **single shared weight** reach 82.0% ± 18.7% on MNIST with a random weight value. An ensemble over weight values reaches 91.6%, the best single tuned weight 91.9%, and fully trained weights 94.2%. For comparison, a linear classifier gets 91.6% and a 2-layer CNN 99.3%. A conventional random-init net gets ~10% (repo/project page) — [weightagnostic.github.io](https://weightagnostic.github.io/)

**Where the innate complexity sits in the brain (Byrnes' framing)**
- Byrnes' model: a "Learning Subsystem" (cortex, striatum, cerebellum, amygdala…) learns from scratch, and a mostly hard-wired "Steering Subsystem" (hypothalamus, brainstem, parts of pallidum) holds innate drives, i.e. primary rewards (snippet) — [Byrnes, Intro to brain-like-AGI safety §3](https://www.greaterwrong.com/posts/hE56gYi5d68uux9oM/intro-to-brain-like-agi-safety-3-two-subsystems-learning-and)
- The Asterisk essay cites a 2023 Nature paper by Fei Chen and colleagues: a *disproportionate* number of mouse-brain cell types sit in the hypothalamus, midbrain and brainstem. It reads this as evolution putting its innovation into the steering subsystem (snippet; secondary) — [Asterisk, "The Sweet Lesson of Neuroscience"](https://asteriskmag.com/issues/13/the-sweet-lesson-of-neuroscience)

### Inferences
- **Bit budget (my calc).** Raw genome: 3.1e9 bp × 2 bits ≈ **6.2e9 bits ≈ 775 MB**. Constrained 8.2% (7.1–9.2%): ≈2.5e8 bp → **≤5.1e8 bits ≈ 64 MB (55–71 MB)**. Deeply conserved 2.2%: ≤1.4e8 bits ≈ 17 MB. These are upper bounds, since constrained sites carry <2 bits each. Most of the functional genome builds cells, bodies and metabolism. So the brain's *algorithmic* specification (region layout, cell types, plasticity rules, neuromodulatory and reward circuitry, developmental schedule) is plausibly **1e6–1e8 bits (0.1–12 MB)**. That is consistent with Marblestone's "~50 MB" upper figure in the earlier notes.
- **Compression gap.** Biology needs ≥6e5× compression (3.7e15 / 6.2e9) against the raw genome, or ~7e6× against the functional genome. Shuvaev's weight-compression reaches 1e2–1e3× while keeping performance. The biological regime is therefore **3–4+ orders of magnitude beyond what "compressed weights" can do**. The genome must encode a *learner plus teacher* (rules + rewards + curriculum), not a network.
- **The ML evidence cuts both ways.** It shows innate *competence* can be compressed (WANN, Shuvaev). It has not shown that a compressed prior yields a *more sample-efficient learner*, which is the thing ASI-at-1e23 needs. Shuvaev's own result is that the learning curve is only shifted, not steepened.
- **Innate vs learned information: ≥1e4–1e6×.** Innate ≤64 MB against learned 1–100 TB means >99.99% of adult brain information is learned. A compact prior therefore does not remove the need for a large volume of experience. It only determines how efficiently that experience is converted.

### Gaps
- I could not read Shuvaev et al.'s full tables: exact per-task accuracies, the RL compression factors beyond HalfCheetah, or the parameter counts of the g-networks. The 76% CIFAR figure is ambiguous.
- I found no peer-reviewed estimate of the brain-specific bit content of the genome (e.g., brain-expressed genes, cis-regulatory elements active in neurons, human-accelerated regions). The 1e6–1e8-bit range is my inference.
- Titles seen but not read: "Stochastic Wiring of Cell Types Enhances Fitness…" (bioRxiv, Aug 2024, likely Koulakov lab) and "Distilling a Modular Reservoir Through a Genomic Bottleneck" (arXiv 2606.28380, June 2026).

---

## 2. Meta-learned / evolved learning rules and algorithms: how much compute does discovery take, and how compact and general are the results?

### Takeaway
Machine-discovered learning components are **compact**. The Disco103 RL update rule is a 0.75M-parameter network (≈3 MB fp32, ≈2.4e7 bits), smaller than the functional genome. It is **genuinely general within its domain**: SOTA on Atari, and it transfers to unseen Crafter, NetHack and Sokoban. Discovery cost roughly **1e22 FLOP** (DiscoRL) to **~1e24 FLOP** (VeLO). But every success so far discovers *one component* (an RL loss, an optimizer, a preference loss, a plasticity rule) inside a human-designed scaffold. Nobody has meta-learned or evolved a complete general learner (architecture + rules + rewards) that learns like a human. Attempts to impose a genome-style bottleneck on evolved rules (Palm et al.) made optimization *harder*.

### Cited Findings
**DiscoRL / Disco57 / Disco103 (Oh, Farquhar, Kemaev, Calian, Hessel, Zintgraf, Singh, van Hasselt, Silver; Nature 648:312–319, Oct 2025) [PR]**
- Method: a neural "meta-network" defines the loss for the agent's policy *and* for extra predictions with no predefined meaning. A population of agents in many environments generates meta-gradients to improve it (repo/project page) — [DiscoRL project page (repo docs)](https://google-deepmind.github.io/disco_rl/); [Nature](https://www.nature.com/articles/s41586-025-09761-x)
- Compute: **Disco57 was discovered on 1,024 TPUv3 cores for 64 h; Disco103 on 2,048 TPUv3 cores for 60 h.** The best rule was found within about 600M steps (snippet; unclear whether per environment or in total) — [PubMed abstract / Nature](https://pubmed.ncbi.nlm.nih.gov/41125136/)
- Results: Disco57 reaches Atari IQM 13.86, beating all existing rules, with "substantially higher wall-clock efficiency" than MuZero (snippet). Disco103 (trained on Atari + ProcGen + DMLab-30) reaches human-level on Crafter and approaches MuZero SOTA on Sokoban; neither was seen during discovery (snippet) — [Nature](https://www.nature.com/articles/s41586-025-09761-x); [36kr summary](https://eu.36kr.com/en/p/3527315416767366)
- Scaling: performance rises with the number, diversity and complexity of discovery environments. The rule "also generalises when used to train agents with much more parameters and data than those used for discovery" (repo/project page) — [google-deepmind/disco_rl docs](https://github.com/google-deepmind/disco_rl)
- **Size of the discovered rule:** the released `disco_103.npz` holds **754,778 float32 parameters** (LSTM-based meta-network; 2.8 MB file) **(measured)** — [google-deepmind/disco_rl weights](https://github.com/google-deepmind/disco_rl)

**Learned optimizers (VeLO, Metz et al., Nov 2022) [PP] + critique [PR, PMLR v239]**
- 4,000 TPU-months of meta-training. It struggles to optimize models "much wider and deeper" than those seen in meta-training, and generalizes poorly to longer training runs (snippet) — [Rezk et al., "Is Scaling Learned Optimizers Worth It?"](https://arxiv.org/abs/2310.18191); [μLO, arXiv 2406.00153](https://arxiv.org/html/2406.00153v4); [Celo, arXiv 2501.12670](https://arxiv.org/html/2501.12670v2)

**Evolved / meta-learned plasticity**
- Najarro & Risi (NeurIPS 2020) [PR]: evolution searches for synapse-specific Hebbian rules on random-weight networks. Agents self-organize weights within a lifetime. A quadruped learns to walk and adapts to unseen morphological damage in <100 timesteps. The approach uses **>450K trainable plasticity parameters** (snippet) — [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/file/ee23e7ad9b473ad072d57aaa9b2a5222-Paper.pdf); [repo](https://github.com/enajx/HebbianMetaLearning)
- Palm, Najarro & Risi, "Testing the Genomic Bottleneck Hypothesis in Hebbian Meta-Learning" (PMLR v148, NeurIPS 2020 pre-registration workshop) [PR-workshop]: they cut the number of distinct Hebbian rules below the number of synapses, hoping this would regularize. "Simultaneously learning the Hebbian learning rules and their assignment to synapses is a difficult optimization problem, leading to poor performance" (snippet) — [PMLR](https://proceedings.mlr.press/v148/palm21a.html)
- Confavreux et al. (NeurIPS 2023) [PR]: simulation-based inference recovers *families* of co-active plasticity rules in recurrent spiking networks. Some rules refine and some reject mean-field predictions. This is a scientific tool, not a capability result (snippet) — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract-Conference.html). Follow-up: "Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks" (PLOS Comput Biol; title only) — [PLOS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012910)

**Discovering algorithms from scratch (AutoML-Zero, Real, Liang, So & Le, 2020) [PR, arXiv 2003.03384]**
- Evolution over programs built from basic math operations rediscovered linear regression with gradient descent and 2-layer networks trained with backprop, and found algorithms that beat hand-designed baselines of similar complexity. Even in the reduced demo space "only ~1 in 10⁸ algorithms in the space can solve" linear regression. A single-CPU baseline reproduction takes 12–18 h (repo) — [google-research/automl_zero](https://github.com/google-research/google-research/tree/master/automl_zero)

**LLM-driven objective discovery (DiscoPOP, Lu, Holt, Fanconi, Chan, Foerster, van der Schaar, Lange; Sakana, June 2024) [PP]**
- An LLM proposes preference-optimization losses, each is scored by fine-tuning, and the loop repeats. About **100 candidate objectives** were evaluated. The winner (LRML, "DiscoPOP") beats DPO on held-out evaluations. The cost is "dozens of full fine-tuning runs" (snippet) — [arXiv 2406.08414](https://arxiv.org/abs/2406.08414); [Sakana blog](https://sakana.ai/llm-squared/); [repo](https://github.com/SakanaAI/DiscoPOP)

**Meta-learned in-context learners**
- Kirsch, Harrison, Sohl-Dickstein, van Steenkiste & Schmidhuber, "General-Purpose In-Context Learning by Meta-Learning Transformers" (arXiv Dec 2022; NeurIPS 2022 workshop) [PP/workshop]: training moves through phase transitions from memorization to system identification to general learning as model size and number of tasks grow. The capabilities of meta-trained learners are "bottlenecked by the accessible state size (memory)", not parameter count. Generalization appears "only with large enough batch sizes and sufficient, but not too many, tasks" (snippet) — [arXiv 2212.04458](https://arxiv.org/abs/2212.04458)
- Lake & Baroni, "Human-like systematic generalization through a meta-learning neural network" (Nature 623:115–121, Oct 2023) [PR]: a standard seq2seq transformer meta-trained on **100K episodes** generated from a compositional grammar prior (MLC). On the few-shot instruction task the model scores **82.4%** against **80.7%** for humans, with item-level correlation r = 0.788 (repo) — [brendenlake/MLC](https://github.com/brendenlake/MLC); [PubMed](https://pubmed.ncbi.nlm.nih.gov/37880371/)
- Bauer et al. (DeepMind), "Human-Timescale Adaptation in an Open-Ended Task Space" (AdA; ICML 2023) [PR]: a meta-RL agent adapts to unseen 3D XLand 2.0 tasks "with just a few minutes of experience, a similar timescale to humans". It relied on an auto-curriculum, large attention memory and distillation, scaled to **>500M parameters**, and shows scaling laws in network size, memory length and task-distribution richness (snippet) — [PMLR v202](https://proceedings.mlr.press/v202/bauer23a/bauer23a.pdf); [project site](https://sites.google.com/view/adaptive-agent/)

### Inferences
- **Discovery compute (my calc).** TPUv3 peak is about 52–62 TFLOP/s per core (bf16; my assumption from Google specs, not fetched). Disco57 ≈ 2.36e8 core-s → **1.2–1.5e22 FLOP peak**; Disco103 ≈ 4.4e8 core-s → **2.3–2.7e22 FLOP peak**. At 30–50% utilization that is ~4e21–1.4e22 FLOP. So a SOTA, cross-domain RL learning rule cost about **1e22 FLOP**, roughly GPT-3-class pretraining (~3e23) divided by 30. VeLO's 4,000 TPU-months ≈ 1.05e10 chip-s → **~5e23–3e24 FLOP peak**, depending on TPU generation. That is Chinchilla-class pretraining (~6e23) up to ~5× it, and the result did not generalize to larger or longer problems.
- **Compactness.** Disco103 is ≈2.4e7 bits at fp32, less at lower precision: ≈5% of the functional-genome bound and ≈0.4% of the raw genome. So a *state-of-the-art learning rule for one broad domain (RL)* fits comfortably inside a "genome". AutoML-Zero's rediscovered backprop is a few lines of code, a few hundred bits. MDL is not the binding constraint; discovery cost and scaffolding are.
- **What the scaffolding hides.** DiscoRL fixed the agent architecture, observation encoders, environments and meta-gradient machinery by hand. It discovered only the loss and target semantics. The general learner's other parts (architecture, memory, reward and steering, curricula) were not discovered. Kirsch et al. show that meta-learning a *general* learner needs a very broad task distribution and a large state. AdA needed >500M parameters and an auto-curriculum just for one procedurally generated 3D domain.
- **Outer-loop cost model.** The cost of discovering a prior ≈ (number of inner-loop evaluations or meta-gradient steps) × (cost of one inner-loop lifetime). DiscoRL kept lifetimes cheap (small agents, Atari) and relied on transfer to bigger agents, which worked. If a proxy lifetime for a *general* learner costs 1e18–1e20 FLOP and the search needs 1e3–1e5 lifetimes, discovery costs **1e21–1e25 FLOP**. If proxies do not transfer and each lifetime must be near-human (1e22–1e24 FLOP), even 1e3 evaluations push it to **1e25–1e27 FLOP**. The evolution anchor (1e41) is the no-scaffolding worst case.
- **Evidence against a simple genome-like bottleneck as a free lunch.** Palm et al. found that sharing rules hurts optimization. Najarro & Risi's working version has *more* rule parameters (450K+) than a compact genome would suggest. Compact priors are found by strong outer-loop optimization (meta-gradients, LLM search), not by imposing compactness.

### Gaps
- I could not confirm the Disco meta-network's architecture details, the number of meta-updates, or the exact FLOP count in the paper's Methods. Nor could I confirm whether "600M steps" is per environment.
- DiscoPOP's total GPU-hours were not found. My guess of ~1e20–1e21 FLOP (≈100 DPO fine-tunes of a 7B model) is unverified.
- AdA's total training compute was not found.
- No result found (to Sept 2026) that meta-learns or evolves a *complete* general-purpose learner (architecture + rules + rewards) that is sample-efficient on open-ended, human-like data.

---

## 3. Reward/loss curricula as the prior: can a handful of loss functions + curricula replace massive data?

### Takeaway
The "steering subsystem" hypothesis (Marblestone–Wayne–Kording 2016; Byrnes) says the genome mostly specifies diverse, stage-dependent cost functions and innate rewards, not knowledge. It is well argued but has **no direct capability demonstration**. The closest ML evidence is *synthetic pre-pretraining curricula*, where a tiny data-generating program stands in for the prior. These give **1.5× (formal languages, peer-reviewed) to ~10× token-equivalence (neural cellular automata, preprint)** at small scale, with modest absolute gains. They are not the 1e4–1e5× gap between child and LLM data. Classic developmental curricula largely failed in BabyLM (earlier notes).

### Cited Findings
**Theory**
- Marblestone, Wayne & Kording, "Toward an integration of deep learning and neuroscience" (Frontiers in Computational Neuroscience, 14 Sept 2016) [PR]. Three hypotheses: (1) the brain optimizes cost functions; (2) "cost functions are diverse across areas and change over development", e.g. simple visual contrasts early and faces later, "bootstrap[ping] more complex knowledge based on simpler knowledge"; (3) optimization runs inside a pre-structured architecture matched to behavioral problems (snippet) — [Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00094/full)
- Byrnes: the Steering Subsystem sends "supervision signals" analogous to AI reward/cost functions "but much more diverse, elaborate, and species specific". These are hard-coded in the genome with "some space for low-dimensional, within-lifetime calibration" (snippet) — [Asterisk](https://asteriskmag.com/issues/13/the-sweet-lesson-of-neuroscience); [Byrnes §3](https://www.greaterwrong.com/posts/hE56gYi5d68uux9oM/intro-to-brain-like-agi-safety-3-two-subsystems-learning-and)
- Intrinsic reward alone: Burda, Edwards, Pathak, Storkey, Darrell & Efros ran "the first large-scale study of purely curiosity-driven learning, i.e. without any extrinsic rewards, across **54** standard benchmark environments", with prediction error as the reward (repo) — [openai/large-scale-curiosity](https://github.com/openai/large-scale-curiosity)

**Synthetic "prior-injection" curricula (tiny generator → transferable inductive bias)**
- Hu, Petty, Shi, Merrill & Linzen, "Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases" (ACL 2025) [PR]. For a 1B-parameter LM trained on ~1.6B natural-language tokens, pre-pretraining on hierarchical formal languages (e.g., shuffle-Dyck) reaches the same loss with a **33% smaller token budget** and better linguistic generalization. Attention heads formed during pre-pretraining remain crucial for syntax (snippet + repo) — [ACL Anthology](https://aclanthology.org/2025.acl-long.478/); [repo](https://github.com/michahu/pre-pretraining)
- Han, Lee, Kumar & Agrawal, "Training Language Models via Neural Cellular Automata" (arXiv 2603.10055, Mar 2026) [PP]. Pre-pre-training on **164M tokens** of NCA trajectories (random neural transition rules the model must infer in context), then 4–13B tokens of natural pretraining:
  - Final perplexity improves by 5.7% (OpenWebText), 5.2% (OpenWebMath) and 4.2% (CodeParrot) against training from scratch, and beats matched-size C4 and Dyck pre-pre-training.
  - GSM8K rises 3.82% → 4.36%, HumanEval 6.75% → 7.49%, and BigBench-Lite 20.91% → 26.51%.
  - Against **1.6B tokens of C4** (~10× more), NCA still "converges 1.4× faster and achieves 5% better final perplexity".
  - Attention layers carry the transferable structure, and the best NCA complexity depends on the domain (repo/blog) — [project blog](https://hanseungwook.github.io/blog/nca-pre-pre-training/); [HF papers page](https://huggingface.co/papers/2603.10055)
- A counter-signal (title only, not read): "Instability of LLM Pre-Pretraining: It Doesn't Always Help. An Investigation on Multiple Languages" (arXiv 2608.08800, 2026) [PP] — [Pith](https://pith.science/paper/2608.08800)

**A compact prior compiled into an amortized learner (strongest demonstration, but narrow)**
- TabPFN (Hollmann et al., Nature, Jan 2025) [PR] is trained *only* on synthetic tables sampled from a parametric structural-causal-model prior. On datasets up to 10,000 samples, "in 2.8 seconds, TabPFN outperforms an ensemble of the strongest baselines tuned for 4 hours" (snippet) — [Nature](https://www.nature.com/articles/s41586-024-08328-6)
- MLC (§2) is the same pattern for compositional language: a grammar prior sampled into 100K episodes yields human-level systematic generalization.

### Inferences
- **Size of the win from "a small generator as prior" (my calc).** Formal-language pre-pretraining gives 1.5× token efficiency. NCA gives ~10× token-equivalence against natural text *at the pre-pre-training stage*, which translates to 1.4–1.6× faster convergence and ~5% better perplexity overall. TabPFN (≈5,000× wall-clock against 4-hour tuning) and MLC show the win can be enormous when the prior matches a narrow domain's generative process. **No experiment shows a compact reward/loss curriculum producing broad world knowledge or reasoning from child-scale data.**
- **Why curricula alone probably cannot close the knowledge gap.** A reward or curriculum specifies *what to learn and in what order*. It does not supply the information content of the world. With innate ≤64 MB and learned 1–100 TB (§1), the steering-subsystem story implies the learner still needs large, *rich* experience (for humans ~1e15 bytes of vision by age 4; see theory_limits.md). The prior can lower the cost of each bit extracted, but it does not create the bits.
- **Most promising "prior" form for low-compute ASI.** Given this evidence, it is not hand-written developmental curricula (BabyLM negative) but **generated-data priors that teach in-context rule inference** (NCA, formal languages, MLC, TabPFN, PFN-style), combined with **discovered update rules** (DiscoRL). Both are compact (kB–MB). Both have shown only 1.3–10× general-domain gains so far.

### Gaps
- No quantitative results found for Burda et al. (the README is only descriptive), nor for any modern "intrinsic-reward-only" agent learning broad skills at low compute.
- No experimental test found of Marblestone/Byrnes-style *multi-loss, stage-dependent* training against single-objective training at matched compute.
- The NCA paper is a 2026 preprint at small scale (4–13B pretraining tokens). Whether its gains persist at 1e11–1e13 tokens is unknown.

---

## 4. Developmental data efficiency: what has actually been learned from child-scale data (BabyLM 2023–2026, SAYCam, chicks)?

### Takeaway
At child scale, *perception* and *word–referent mapping* are learnable with generic architectures:
- 61 h of headcam gives 61.6% vs CLIP's 66.7% on 22 concepts.
- 200 h of headcam video gives ~70% of an ImageNet-trained model.
- ViTs match newborn chicks on view-invariant recognition from the same impoverished visual input.

*Grammar* is learnable from ≤1e8 words: BLiMP ~75–80% at 100M words, and GPT-BERT's GLUE 81.5 vs the 68.4 baseline. But **world knowledge is not**: EWoK sits near chance (50%), with a best of 58.4% in 2024 and 2025 baselines at 49.5–52.4%. The 2025 Interaction track (LLM teachers) did not beat text-only models and was merged away for 2026. As of Sept 2026 I found **no** result showing broad knowledge or reasoning from ≤1e8–1e9 words.

### Cited Findings
**BabyLM 2024 (Second Challenge, CoNLL 2024 findings; arXiv 2412.05149) [PR]**
- GLUE macro-average, Strict (100M words): **GPT-BERT 81.5**, BabbleGPT 71.7, AntLM 66.3; best baseline LTG-BERT 68.4. Strict-Small (10M): GPT-BERT 76.5, DeBaby 73.7, MLSM 73.3; best baseline BabyLlama 63.3. GPT-BERT's 10M-word model beats the 100M-word runners-up (snippet) — [Findings 2024](https://arxiv.org/pdf/2412.05149)
- EWoK was a hidden evaluation: "current systems do not learn world knowledge within 100M words, with most submissions performing near chance at 50%, and the maximum score being **58.4%**." The organisers suggest the BabyLM corpus may simply not contain the knowledge EWoK tests (snippet) — [Findings 2024](https://arxiv.org/pdf/2412.05149)

**BabyLM 2025 (Third Challenge / First BabyLM Workshop, EMNLP 2025) [PR]**
- Findings: new objectives and architectures work best, and GPT-BERT, DeBERTa and LTG-BERT backbones lead. There is "not a complete correlation between training FLOPs and performance" in 2025, compared with the "strong relationship" reported in 2024 (snippet) — [Findings 2025](https://aclanthology.org/2025.babylm-main.28/)
- Exposure rule: the 2025 evaluation code "assumes that you trained on the entire budget (**100M words for strict-small and 1B words for strict**)". That is, 10 epochs over the 10M/100M datasets (repo) — [evaluation-pipeline-2025 README](https://github.com/babylm/evaluation-pipeline-2025)
- Baseline scores in the same README (in a commented-out block, so possibly preliminary; repo):
  - Strict (100M) GPT-BERT causal-focus: BLiMP 79.29, BLiMP-supplement 70.42, **EWoK 52.32**.
  - GPT-2 Small: BLiMP 75.07, **EWoK 51.22**.
  - Strict-small (10M) EWoK: 49.47–50.23.
  - Interaction (SimPO) baseline: EWoK 52.44.
  - Multimodal baselines: Flamingo and GIT reach Winoground 51.6/55.5 and VQA 52.3/54.1, against 50.0 and 45.0/48.4 without vision.
  — [evaluation-pipeline-2025](https://github.com/babylm/evaluation-pipeline-2025)
- Interaction track: students ≤100M words, with Llama-3.1-8B-Instruct as teacher in the baseline. "None of the submissions in the INTERACTION track outperformed models submitted to the STRICT track, which is why the 2026 challenge merged these tracks" (snippet) — [BabyLM 2026 CfP, arXiv 2602.20092](https://arxiv.org/pdf/2602.20092); [2025 CfP](https://arxiv.org/html/2502.10645v1)

**BabyLM 2026 (in progress; multilingual) [PP]**
- Looped GPT-BERT (arXiv 2609.09691, Sept 2026): 12.18M parameters (4 physical layers × 12 recurrent passes). On the BabyLM 2026 leaderboard it reaches BLiMP 71.19 vs GPT-BERT causal-focus 71.66 and GPT-2 65.23, and GLUE 62.55 vs 65.13/63.80. Overall average 35.42 (snippet) — [arXiv 2609.09691](https://arxiv.org/abs/2609.09691)

**Child headcam (SAYCam) grounding: Vong, Wang, Orhan & Lake (Science, 1 Feb 2024) [PR]**
- Data: one child aged 6–25 months, **61 h** of video, about 600K frames paired with **37.5K** transcribed utterances, and about 250K word instances. The contrastive CVCL model reaches **61.6%** on Labeled-S (22 concepts). CLIP out of the box gets **66.7%** despite far more training data. The model generalizes zero-shot to new visual referents (snippet) — [Science](https://www.science.org/doi/10.1126/science.adi1374); [PsyPost summary](https://www.psypost.org/ai-autonomously-learns-language-through-the-experience-of-a-single-child-in-groundbreaking-study/)
- 2026 follow-up, "Objects Before Words" (BabyMind, arXiv 2606.12985, June 2026) [PP]: an object-first bias (mask-based object candidates, short-window "object files" by tracking, multiple-instance contrastive loss) gains **+2.6 points** over CVCL on Labeled-S forced choice on SAYCam-S, with consistent in-vocabulary OOD gains (snippet) — [arXiv 2606.12985](https://arxiv.org/abs/2606.12985)

**Child-view vision without strong inductive biases: Orhan & Lake (Nature Machine Intelligence 6:271–283, Mar 2024) [PR]**
- Generic SSL models (DINO/Mugs/MAE; ResNeXt, ViT) trained on **~200 h** of one child's headcam video over two years. The best embeddings reach on average **~70% of a high-performance ImageNet-trained model**. They learn broad semantic categories and object localization without supervision, but are "less object-centric" than ImageNet models (snippet) — [NMI](https://www.nature.com/articles/s42256-024-00802-0); models [silicon-menagerie](https://github.com/eminorhan/silicon-menagerie) (repo)

**Newborn chicks vs transformers: Pandey, Wood & Wood (NeurIPS 2023) [PR]**
- Chicks were reared with a single object. The researchers simulated their first-person views in a game engine and trained self-supervised ViTs with a time-contrastive loss (views within 300 ms treated as similar). The ViTs "solved the same view invariant object recognition tasks as the chicks": they were "not more data hungry than newborn visual systems" (snippet + repo) — [arXiv 2312.02843](https://arxiv.org/abs/2312.02843); [ViT-CoT repo](https://github.com/buildingamind/ViT-CoT); related "Parallel development of object recognition in newborn chicks and deep neural networks" ([PLOS Comput Biol; title only](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012600))

### Inferences
- **Two regimes.** (a) Perceptual and grammatical competence: child-scale data plus generic architectures get within 1–1.5× of large-data models (61.6 vs 66.7; ~70% of ImageNet; BLiMP high-70s). Here the innate prior need not carry much, and a temporal/contrastive objective suffices (chicks, SAYCam). (b) Knowledge and reasoning: at ≤1e8 words (≤1e9 exposures) EWoK is at chance. No 2025–2026 entry, interaction track or LLM-teacher setup fixed this.
- **Interpretation.** The EWoK failure is at least partly a *data-content* problem (text children hear lacks the facts; children get them from multimodal, embodied experience), not purely a prior problem. The multimodal tracks also failed, but they used ≤100M words plus images, not child-scale embodied video (~1e15 bytes). The experiment that would test "prior + child-scale multimodal experience ⇒ knowledge" **has not been run at scale.**
- **Chance levels (unverified).** Labeled-S in Vong et al. is, as I recall, 4-way forced choice (chance 25%). EWoK and BLiMP pairs are 2-way (chance 50%). So 61.6% is far above chance, while EWoK at 50–58% is not.
- **For the compute question.** Even where child-scale data suffices, the learner is not cheap. BabyLM winners spend many epochs (2025 capped at 10), and SAYCam models are standard SSL runs. So data efficiency here is not compute efficiency (see brain_inspired_alternatives.md §4).

### Gaps
- The 2025 BabyLM winners' names and final scores could not be verified (ACL Anthology blocked). One search extract attributed the 2025 Strict and Strict-small wins to Charpentier & Samuel's layer-selection approach, but this looks conflated with 2023's ELC-BERT, so it is unconfirmed.
- The BabyLM 2026 leaderboard (Hugging Face) was not readable.
- Not read (titles only, 2025–2026): "On the robustness of modeling grounded word learning through a child's egocentric input" (arXiv 2507.14749), "Continual Visual and Verbal Learning Through a Child's Egocentric Input" (arXiv 2606.05115), "Temporal Slowness in Central Vision Drives Semantic Object Learning" (arXiv 2602.04462), "Squeezing More from Limited Data with Recursive Transformers" (arXiv 2608.26973), "Language Acquisition Device in Large Language Models" (arXiv 2605.16758).
- No 100M–1B-word model evaluated on MMLU/ARC-style knowledge or reasoning was found. Search attempts returned only generic small-LM content.

---

## 5. Core knowledge and innate object/physics priors: innate or learned, and do built-in priors cut data/compute in ML?

### Takeaway
Spelke's core systems (objects, agents/actions, number, space/geometry, plus possibly social partners) are well supported in infants. The 2022–2025 ML evidence, however, shows much of the *object/physics* core can be **learned** from modest video by generic predictive objectives: V-JEPA gets above-chance intuitive physics from one week of video, and ViTs match chicks. Building in object-centric structure helps (PLATO: robust physics effects from 28 h, "depends critically" on object representations; BabyMind +2.6 points). But the demonstrated multipliers are **small-to-moderate and domain-narrow**. None cuts the data or compute for *general* competence by orders of magnitude.

### Cited Findings
- Spelke & Kinzler, "Core knowledge" (Developmental Science 10(1), 2007) [PR]: "Human cognition is founded, in part, on four systems for representing objects, actions, number, and space. It may be based, as well, on a fifth system for representing social partners." The evidence converges from infants, non-human primates, and children and adults across cultures. The systems are described as ancient, early emerging, invariant, automatic and encapsulated (snippet) — [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-7687.2007.00569.x); [PDF](https://www.harvardlds.org/wp-content/uploads/2017/01/SpelkeKinzler07-1.pdf)
- PLATO (Piloto, Weinstein, Battaglia & Botvinick; Nature Human Behaviour, July 2022) [PR]: an object-centric model (auto-encoding plus object tracking, inspired by developmental psychology) learns solidity, persistence and related concepts. It shows "robust violation-of-expectation effects… with as little as **50,000 examples (28 hours of visual experience)**". Learning "depends critically on object-level representations" (snippet) — [Nature Human Behaviour](https://www.nature.com/articles/s41562-022-01394-8)
- Garrido et al. (Meta FAIR), "Intuitive physics understanding emerges from self-supervised pretraining on natural videos" (arXiv 2502.11831, Feb 2025) [PP]: V-JEPA, which predicts in a learned representation space, shows object permanence and shape consistency on IntPhys and InfLevel. Pixel-space video predictors and multimodal LLMs are near chance. "Models trained on **one week of unique video** achieve above chance performance" (snippet) — [arXiv 2502.11831](https://arxiv.org/abs/2502.11831)
- Pandey, Wood & Wood (NeurIPS 2023) [PR]: see §4. Generic ViTs plus a temporal-contrast objective match newborn chicks' view-invariant recognition from the same single-object rearing input — [arXiv 2312.02843](https://arxiv.org/abs/2312.02843)
- Orhan & Lake (NMI 2024) [PR]: child-view models without strong inductive biases are less object-centric than ImageNet models. That gap is a candidate for what an innate object prior adds — [NMI](https://www.nature.com/articles/s42256-024-00802-0)
- BabyMind (arXiv 2606.12985, June 2026) [PP]: an object-first bias gives +2.6 points on child-view word grounding — [arXiv](https://arxiv.org/abs/2606.12985)
- The earlier notes cover object-centric RL: VERSES' AXIOM claims large parameter and runtime savings on its own game suite, not independently replicated (brain_inspired_alternatives.md §2).

### Inferences
- **Innate vs learnable.** The ML record suggests that object permanence, shape constancy and view invariance can be acquired from days to weeks of video with a generic *predictive* objective. The strongest innate candidates are therefore the *objective* ("predict in latent space", "things seen close in time are the same thing") and *attentional/reward biases* (faces, agents, contingency), not hard-coded physics. That fits §3's steering-subsystem view and keeps the prior's bit budget small.
- **Size of the multipliers.** Documented gains from built-in object/physics structure are ~1.0–1.5× on child-view benchmarks (BabyMind +2.6 points) and "works at all vs doesn't" on narrow VoE tests (PLATO). No study reports an order-of-magnitude cut in data or compute for broad competence. The claim that core-knowledge priors buy 10–100× overall remains **undemonstrated**.

### Gaps
- No controlled ML study was found that measures data or compute saved by building in number, geometry or agent/social core knowledge.
- V-JEPA's full data-scaling curve (hours of video vs accuracy) and IntPhys-2 results were not read.
- No numbers were read from 2026 probing work ("How Do Video Foundation Models Encode Intuitive Physics?", arXiv 2606.09646).

---

## 6. Bottom line: minimum description length and compute of a sufficient prior, and compute to discover it

### Takeaway
- **MDL of the prior.** Best estimate is **~1e7–1e8 bits (≈1–12 MB)**, with a range of 1e6–5e8 bits (upper bound = functional genome ≈64 MB). Biology and ML agree that learning rules, losses and rewards fit in this budget: Disco103 is 2.4e7 bits, and AutoML-Zero's backprop is ~1e2–1e3 bits.
- **Compute for a learner that uses it** to reach human-level general competence: **~1e23–1e25 FLOP** (lifetime anchor), *plus* rich multimodal experience. The prior does not remove the experience requirement.
- **Compute to discover the prior.** Component-level discovery inside human scaffolds costs 1e21–1e24 FLOP (demonstrated). A *complete* general-learner prior costs, on my estimate, **~1e24–1e27 FLOP (central ~1e25–1e26)** if cheap proxy lifetimes transfer, as they did for DiscoRL. It is far more if they do not (evolution upper bound 1e41).
- **Answer to the first round's uncertainty.** A genome-sized prior **plausibly exists**. It has **not been demonstrated**, and the evidence to date (1.3–10× general-domain gains, knowledge still at chance with ≤1e9 words) favours the "buy a prior with mid-scale pretraining, then continual learning" route in the near term.

### Cited Findings (quantitative anchors used in the estimate)
| Quantity | Value | Source / status |
|---|---|---|
| Raw genome information | ≤~1 GB (3×10⁹ nt); 6.2e9 bits at 2 bits/bp | [Zador 2019](https://www.biorxiv.org/content/10.1101/582643v1.full) [PR] (snippet); my calc |
| Functional (constrained) fraction | 8.2% (7.1–9.2%) → ≤5.1e8 bits ≈ 64 MB | [Rands et al. 2014](https://journals.plos.org/plosgenetics/article?id=10.1371%2Fjournal.pgen.1004525) [PR] (snippet); my calc |
| Bits to specify connectome explicitly | ~3.7e15 bits (≥6 OOM above genome) | [Zador 2019](https://www.biorxiv.org/content/10.1101/582643v1.full) [PR] (snippet) |
| Learned storage in adult brain | ~10–100 TB (range 1 TB–2.5 PB) | [AI Impacts](https://aiimpacts.org/information-storage-in-the-brain/) (snippet) |
| Weight compression retaining innate performance | 92× (CIFAR), 322–1,038× (MNIST), 4.9× (HalfCheetah); no learning speedup | [Shuvaev et al. 2024](https://www.pnas.org/doi/10.1073/pnas.2409160121) [PR] (snippet) |
| Size of a discovered SOTA RL learning rule | 754,778 params ≈ 3 MB fp32 ≈ 2.4e7 bits | [disco_rl weights](https://github.com/google-deepmind/disco_rl) (measured) |
| Compute to discover it | 1,024 TPUv3 cores × 64 h (Disco57); 2,048 × 60 h (Disco103) ≈ 1–3e22 FLOP peak | [Oh et al. 2025](https://www.nature.com/articles/s41586-025-09761-x) [PR] (snippet); my calc |
| Learned optimizer meta-training | 4,000 TPU-months ≈ 5e23–3e24 FLOP peak; poor generalization to larger/longer runs | [Rezk et al.](https://arxiv.org/abs/2310.18191) (snippet); my calc |
| Synthetic-prior token savings (general LM) | 1.5× (formal languages); 164M NCA tokens > 1.6B C4 tokens (~10×) at pre-pre-training stage | [Hu et al. ACL 2025](https://aclanthology.org/2025.acl-long.478/) [PR]; [Han et al. 2026](https://hanseungwook.github.io/blog/nca-pre-pre-training/) [PP] |
| Narrow-domain compiled priors | TabPFN 2.8 s beats 4 h-tuned ensemble; MLC 82.4% vs human 80.7% | [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) [PR]; [MLC repo](https://github.com/brendenlake/MLC) [PR] |
| World knowledge from ≤1e8 words | EWoK ≈ chance (best 58.4% in 2024; 2025 baselines 49.5–52.4%) | [BabyLM 2024 findings](https://arxiv.org/pdf/2412.05149) [PR]; [eval-2025 README](https://github.com/babylm/evaluation-pipeline-2025) (repo) |
| Child-scale perception | 61 h → 61.6% vs CLIP 66.7%; 200 h → ~70% of ImageNet model | [Vong et al. 2024](https://www.science.org/doi/10.1126/science.adi1374) [PR]; [Orhan & Lake 2024](https://www.nature.com/articles/s42256-024-00802-0) [PR] |

### Inferences
1. **MDL is not the bottleneck.** Every strand of evidence puts the needed innate specification at ≤1e8 bits: the genome bound, Byrnes' steering-subsystem view, cell-type concentration in hypothalamus and brainstem, and the size of discovered rules. That is 3–5 orders of magnitude below the ~1e11–1e13 bits in a frontier LLM's weights. The first-round question "can the prior be genome-sized?" is therefore **probably yes in principle**. The binding questions are (a) whether we can *find* it, and (b) whether a learner using it still needs lifetime-scale compute and experience.
2. **A compact prior does not by itself imply low total FLOP (my calc).** The human lifetime anchor is ~1e24 FLOP for ~1e8 words, i.e. **~1e16 FLOP per word-equivalent** if all compute is charged to language. DeepSeek-V3 spends 3.3e24 / 1.48e13 ≈ **2e11 FLOP per token**. The brain is ~1e4–1e5× more *sample*-efficient and ~1e4–1e5× less *compute-per-sample*-efficient, and **total compute is the same order (~1e24)**. A brain-like learner with a genome-sized prior therefore lands near 1e23–1e25 FLOP for *human-level* competence, not far below. The 1e23–1e24 ASI scenario additionally needs (i) per-sample compute well below the brain's, and (ii) superhuman breadth without proportionally more experience. Neither is evidenced.
3. **Measured multipliers from priors are far smaller than the gap.** General-domain prior injections so far give 1.3–10× (formal languages, NCA, Shuvaev transfer at ~1.5× fewer epochs, BabyMind +2.6 points). The child–LLM *data* gap is ~1e5× (theory_limits.md), and the *compute* gap between a 1e24 run and a 1e25–1e26 pretraining run is 10–100×. Closing 10–100× in compute through priors alone would take roughly 2–4 stacked, independent 3× improvements that also hold at scale. That is plausible but undemonstrated. Only narrow domains (TabPFN, MLC, PLATO, WANN) show orders-of-magnitude sample-efficiency from priors.
4. **Discovery compute (my estimate).** Components inside human-designed scaffolds cost 1e21–1e22 FLOP (DiscoRL) to ~1e24 (VeLO). A full general-learner prior needs an outer loop over proxy "lifetimes" of a general learner. At 1e19–1e21 FLOP per proxy lifetime and 1e3–1e5 lifetimes or meta-updates, that gives **~1e22–1e26 FLOP**. If proxies must be near-human scale (1e22–1e24 each), it gives **1e25–1e28**. My central estimate is **~1e25–1e26 FLOP**. That is comparable to one frontier pretraining run, and it is a one-time cost, which is why the first report kept it outside the per-instance ledger. Using neuroscience (connectomics, cell-type atlases, Byrnes/Marblestone reverse-engineering) instead of search could make discovery compute-cheap but science-limited. Its timeline is unknown.
5. **Probability (my judgement, low confidence).** P(a genome-sized prior plus ≤1e24 FLOP of end-to-end learning yields broadly human-level, sample-efficient general competence, demonstrated by ~2032) ≈ **20–30%**. P(the same at ≤1e24 FLOP for *superhuman* general capability) ≈ **5–15%**. The complementary route (a mid-scale "bought" prior of ~1e24–1e25 FLOP of pretraining or distillation, then continual learning) is more likely to reach the goal first. Priors discovered along the way (DiscoRL-type rules, NCA/PFN-type synthetic curricula) will probably be absorbed into it, giving the "1–2 OOM higher" scenario of the first report, not the genome-only one.
6. **What would flip the verdict (falsifiable markers).** (a) A ≤1e9-word or child-scale-multimodal model scoring well above chance on EWoK or other knowledge/reasoning suites *without* an LLM teacher. (b) A meta-learned or evolved *full* learner (not one component) beating a hand-designed one on open-ended tasks with transfer to larger scale, as DiscoRL did for RL. (c) Synthetic-prior pre-pretraining gains that *grow* rather than shrink with pretraining scale.

### Gaps
- No primary source quantifies how much compute a *complete* general-learner prior would take to discover. The ranges above are my extrapolations from DiscoRL, VeLO and the evolution anchor.
- The per-word brain-compute comparison assigns all lifetime compute to language. It is an order-of-magnitude illustration, not a measured quantity.
- Several 2025–2026 preprints relevant to this question were seen only as titles (listed in §§1, 3, 4, 5) and could not be assessed, because arXiv and Hugging Face were blocked and the session's search budget ran out.
