# Computational and Statistical Learning Theory on the Minimum Compute for General Learning and Reasoning (beyond AIXI/Solomonoff/Levin)

Method note for the report writer. arxiv.org, ar5iv, openreview and github.io project pages were blocked for direct fetch in this session, so every finding comes from one of three places:
- **Search-engine extracts** of primary sources, linked to the primary URL. These carry a **(snippet)** tag where the wording is only from the extract.
- **Classic results cited from memory**, tagged **(from memory, verify)** and linked to the canonical arXiv or DOI page. Treat the exact constants in these as unverified.
- **My own arithmetic**, which appears only under "Inferences" and is tagged "my estimate", with its assumptions stated.

This file deliberately does **not** repeat the AIXI, Solomonoff, Levin search, Gödel machine, Legg and No-Free-Lunch material, or the Kaplan/Chinchilla/α≈4/d exponent material. Those are covered in `research_notes/极低算力超级智能路径/theory_limits.md` (Q4–Q5).

Proven theorems are always conditional. Most hardness results assume an unproven conjecture (P≠NP, TC⁰≠NC¹/P, hardness of planted clique or of LPN, or cryptographic assumptions). Upper bounds and "SGD learns X" theorems hold only for stylised models (sparse parity, single- and multi-index Gaussian models, linear representations). Neither kind gives a FLOP floor for "general intelligence", because that task distribution is not formally defined.

## Q1. Statistical–computational gaps: what problems with "sample-cheap but compute-expensive" learning imply about trading data for compute

### Takeaway
For many canonical problems, the fewest samples that suffice information-theoretically can only be exploited by (conjecturally) exponential computation. Polynomial-time or gradient/SQ learners need either polynomially more data, far more compute, a larger model, or luck. The sharpest example is sparse parity: the Edelman et al. Pareto frontier over (data, compute, width, luck).

Three consequences follow:
- Maximal **sample efficiency is generally not compute-efficient**. The brain's position (≈1e5× fewer language samples, ≈1e5× more compute per sample) is what this theory predicts for a learner sitting nearer the information-theoretic sample limit.
- For *passive* learners with gradients or statistical queries, the gap closes in only two ways: **intermediate supervision / curriculum** (chain of thought, decomposed targets), or **active queries or experiments**.
- Worst-case general learning of even small neural networks is cryptographically hard. Any efficient "general" learner therefore relies on the real world having benign, low-"globality", curriculum-rich structure.

### Cited Findings
**Sparse parity: the archetypal gap**
- Barak, Edelman, Goel, Kakade, Malach, Zhang (NeurIPS 2022), "Hidden progress in deep learning: SGD learns parities near the computational limit":
  - They frame learning a k-sparse parity of n bits as "a canonical discrete search problem which is statistically easy but computationally hard".
  - Neural nets learn it with abrupt phase transitions after **≈n^O(k) iterations**, which "nearly matches SQ lower bounds".
  - SGD makes hidden continual progress via a Fourier gap in the population gradient, invisible to loss (snippet).
  - [arXiv 2207.08799](https://arxiv.org/pdf/2207.08799); [NeurIPS](https://proceedings.neurips.cc//paper_files/paper/2022/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html)
- Edelman, Goel, Kakade, Malach, Zhang (NeurIPS 2023), "Pareto frontiers in neural feature learning: data, compute, width, and luck":
  - Offline sparse parity has an SQ lower bound "interpreted as a multi-resource tradeoff frontier". Learning succeeds only if the learner is sufficiently "rich (large model), knowledgeable (large dataset), patient (many training iterations), or lucky (many random guesses)".
  - Sparse initialisation and larger width improve **sample efficiency**, with width acting as **parallel search** that amplifies the chance of finding "lottery ticket" neurons.
  - Wide, sparsely initialised MLPs sometimes beat tuned random forests on tabular benchmarks (snippet).
  - [arXiv 2309.03800](https://arxiv.org/abs/2309.03800); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/960573a3b797441aec39caa9f74bc793-Paper-Conference.pdf); [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pareto-frontiers-in-neural-feature-learning-data-compute-width-and-luck/)
- Information-theoretically, a k-sparse parity over n bits needs only O(k log n) samples, but brute force takes ~n^k time. Kearns's SQ model gives an n^Ω(k) lower bound for any SQ algorithm. **(from memory, verify)** — [Kearns 1998, JACM](https://dl.acm.org/doi/10.1145/293347.293351)
- Learning parity with noise (LPN), full parity over n bits: Blum–Kalai–Wasserman runs in 2^O(n/log n) time **and** samples. Their paper also separates SQ learning from noise-tolerant PAC learning. **(from memory, verify)** — [Blum, Kalai, Wasserman, arXiv cs/0010022](https://arxiv.org/abs/cs/0010022). For k-sparse noisy parity the best known algorithms remain ~n^(ck) with c<1, e.g. G. Valiant 2012 (~n^0.8k). **(from memory, verify)** — [Valiant, ECCC TR12-006](https://eccc.weizmann.ac.il/report/2012/006/)

**Planted clique and sparse PCA**
- Planted clique: a clique of size ~2 log n is detectable information-theoretically, but no polynomial-time algorithm is known below ~√n. Sum-of-squares lower bounds support the conjecture. **(from memory, verify)** — [Barak et al. 2016, SoS lower bound for planted clique, arXiv 1604.03084](https://arxiv.org/abs/1604.03084)
- Sparse PCA (Berthet & Rigollet, COLT 2013):
  - Detecting a k-sparse spike in dimension d needs ~k log d samples information-theoretically.
  - Under planted-clique hardness, polynomial-time detection needs ~k² log d samples.
  - This is a clean "pay with more data, not more compute" gap. **(from memory, verify)**
  - [PMLR v30](https://proceedings.mlr.press/v30/Berthet13.html)

**Gradient/SQ learners in high-dimensional models**
- Single-index models:
  - Online SGD needs n ≈ d^(k*−1) samples, where k* is the "information exponent". **(from memory, verify)** — [Ben Arous, Gheissari, Jagannath, arXiv 2003.10409](https://arxiv.org/abs/2003.10409)
  - Landscape smoothing reduces this to ≈d^(k*/2), matching the correlational-SQ (CSQ) lower bound. **(from memory, verify)** — [Damian, Nichani, Ge, Lee, arXiv 2305.10633](https://arxiv.org/abs/2305.10633)
- Leap complexity: for two-layer nets trained by SGD on sparse Boolean or Gaussian targets, sample and time complexity scale like d^max(Leap−1, 1). "Staircase" targets, where each new feature is reachable from previous ones, are learnable with ~d samples; high-leap targets are not. **(from memory, verify)** — [Abbe, Boix-Adserà, Misiakiewicz, arXiv 2302.11055](https://arxiv.org/abs/2302.11055)
- Which regime practical deep learning sits in:
  - Abbe & Sandon (2020): SGD on poly-size nets is, in principle, P-universal, so it can learn anything poly-time learnable, given contrived architectures and initialisations. With noise or low precision it collapses to SQ limitations. **(from memory, verify)** — [arXiv 2001.02992](https://arxiv.org/abs/2001.02992)
  - Abbe, Kamath, Malach, Sandon, Srebro (NeurIPS 2021): small-batch, high-precision mini-batch SGD can simulate PAC learning; large-batch or low-precision SGD is SQ-limited. **(from memory, verify)** — [arXiv 2108.04190](https://arxiv.org/abs/2108.04190)
- Abbe et al. (NeurIPS 2024), "How far can transformers reason? The globality barrier and inductive scratchpad": transformers cannot efficiently learn targets of high "globality degree" from input–output pairs. Scratchpads that decompose the target into local steps remove the barrier. **(from memory, verify)** — [arXiv 2406.06467](https://arxiv.org/abs/2406.06467)

**Intermediate supervision (chain-of-thought data) collapses the gap**
- Wies, Levine, Shashua (ICLR 2023): parity becomes learnable in polynomial time when sub-task outputs are supervised. **(from memory, verify)** — [arXiv 2204.02892](https://arxiv.org/abs/2204.02892)
- Malach (2023): even linear next-token predictors trained on CoT-style data can learn any efficiently Turing-computable function. The relevant complexity measure becomes "length complexity" (how many intermediate tokens are needed). **(from memory, verify)** — [arXiv 2309.06979](https://arxiv.org/abs/2309.06979)
- Kim & Suzuki (ICLR 2025): transformers provably learn parity efficiently when trained with chain of thought. **(from memory, verify)** — [arXiv 2410.08633](https://arxiv.org/abs/2410.08633)

**Active queries collapse some gaps**
- With membership queries (the learner chooses inputs, i.e. experiments), sparse-Fourier and parity-type concepts become polynomial-time learnable (Goldreich–Levin / Kushilevitz–Mansour). **(from memory, verify)** — [Kushilevitz & Mansour, SIAM J. Comput. 1993](https://doi.org/10.1137/0222080)
- Active learning can reduce label complexity exponentially for some classes, e.g. thresholds: log(1/ε) vs 1/ε. **(from memory, verify)** — [Dasgupta, NeurIPS 2005](https://papers.nips.cc/paper/2943-coarse-sample-complexity-bounds-for-active-learning)

**Worst-case hardness of learning neural nets**
- Under cryptographic assumptions, even intersections of halfspaces and small-depth networks are hard to PAC-learn efficiently. **(from memory, verify)** — [Klivans & Sherstov, FOCS 2006](https://doi.org/10.1109/FOCS.2006.24); [Daniely & Vardi, arXiv 2101.08303](https://arxiv.org/abs/2101.08303); [Chen, Gollakota, Klivans, Meka, arXiv 2202.05258](https://arxiv.org/abs/2202.05258)

### Inferences
- **Statistical–computational gaps as the theoretical form of "sample efficiency ≠ compute efficiency".**
  - For sparse parity, sparse PCA, planted clique and LPN, the sample-optimal learner is (conjecturally) exponentially slower than the best polynomial-time learner. The polynomial-time learner in turn needs polynomially more data (k² vs k for sparse PCA), a wider net, or more iterations (Edelman et al.).
  - The brain's "fewer samples, far more compute per sample" point is therefore the expected shape of the frontier, not an anomaly.
- **Implication for the report's ledger.** The report counts data as free (up to the ~300T-token human-text stock), so the compute-minimising move on this frontier is to *consume more data per FLOP*, not to be maximally sample-efficient. Theory thus somewhat favours a "data-hungry but compute-cheap" learner, such as a compact cognitive kernel trained on abundant human text, over a sample-efficient experiential learner, *wherever abundant data exist*. Sample efficiency pays off in compute terms only where data are scarce or must be generated at a compute cost (self-play, simulation, beyond-human domains).
- **Human text as curriculum.** Human text, especially explanations, worked solutions and code, acts as **intermediate supervision**. Theory (Wies; Malach; Abbe globality; Kim & Suzuki) says this can turn exponential-compute concept learning into polynomial-compute learning.
  - A learner starting "from scratch" on raw sensory streams plus sparse reward is, in SQ terms, facing the high-globality regime. Sparse episode reward is close to a 1-bit statistical query per episode.
  - This is a theory-grounded cost of the report's #1 direction (experiential world-model agent) relative to #2 (text-pretrained kernel), unless the agent gets a curriculum.
- **Interaction's theoretical upside.** Interaction is the other known gap-closer: *active queries/experiments* can make SQ-hard classes easy. This is the strongest learning-theory argument **for** the world-model agent, but only if it actually chooses informative experiments. Passive video prediction is still SQ-like.
- **No general learner is efficient on everything.** Cryptographic hardness implies that an ASI's efficiency must come from matching the structure of the real world's "concept distribution" (low globality, staircase-like hierarchies, curriculum available). This echoes the NFL/Legg point in theory_limits.md, but here in polynomial-vs-exponential terms.

### Gaps
- No result quantifies the statistical–computational gap for *natural* language or world-modelling tasks. All gap results are for stylised distributions (parity, Gaussian single/multi-index, planted structures).
- I could not fetch the Edelman et al. paper to extract the exact functional form of the frontier, e.g. whether (samples × width × iterations) must exceed ~n^k.
- Hardness results rely on unproven conjectures: planted-clique hardness, LPN hardness, and cryptographic PRGs.

## Q2. Sample–compute trade-offs in both directions, quantified

### Takeaway
- **More data → less compute** is a theorem-level phenomenon. Runtime can fall *exponentially* with only *polynomially* more examples (Shalev-Shwartz–Shamir–Tromer 2012), and convex-relaxation hierarchies let one use cheaper relaxations as n grows (Chandrasekaran–Jordan 2013).
- **More compute → less data** is also real, but the empirical exchange rates for LM pretraining are bounded to roughly 1–2.5 orders of magnitude so far:
  - multi-epoch repetition gives a data-reuse "half-life" R_D* ≈ 15 epochs for autoregressive LMs vs ≈500 for masked diffusion;
  - regularisation plus ensembling gives 5.17× data efficiency;
  - data reuse in SGD provably reaches information-theoretic sample complexity in single-index models.
- Train-time and test-time compute trade at roughly 1 OOM for 1–1.5 OOM, and only across a few OOM in total.

### Cited Findings
**More data → less compute (theory)**
- Shalev-Shwartz, Shamir, Tromer (AISTATS 2012), "Using more data to speed-up training time": the paper studies learning runtime as a function of the number of examples and gives "the first formal positive result showing that even in the unrealizable case, the runtime can decrease exponentially while only requiring a polynomial growth of the number of examples" (snippet). — [PMLR v22](http://proceedings.mlr.press/v22/shalev-shwartz12.html); [PDF](https://www.cs.huji.ac.il/~shais/papers/ShalevShamirTromer12.pdf)
- Daniely, Linial, Shalev-Shwartz (NeurIPS 2013): for halfspaces over sparse vectors, more data provably speeds up learning under hardness assumptions. **(from memory, verify)** — [arXiv 1311.2271](https://arxiv.org/abs/1311.2271)
- Chandrasekaran & Jordan (PNAS 2013), "Computational and statistical tradeoffs via convex relaxation": for denoising, as n grows one can swap a tight but expensive convex relaxation for looser, cheaper ones and keep the same risk. This gives an explicit time–data trade-off curve. **(from memory, verify)** — [PNAS 110(13)](https://www.pnas.org/doi/10.1073/pnas.1302293110)
- Sparse PCA: ~k² log d samples permit polynomial time, against ~k log d information-theoretically with exponential time (Berthet & Rigollet, see Q1). **(from memory, verify)**

**More compute → less data (theory)**
- Lee, Oko, Suzuki, Wu (NeurIPS 2024), "Neural network learns low-dimensional polynomials with SGD near the information-theoretic limit":
  - With **reused data**, a two-layer net trained by an SGD-based algorithm learns single-index functions with sample and runtime complexity **n ≍ T ≍ d·polylog(d)**, "matching the information theoretic limit up to polylogarithmic factors". Complexity is no longer governed by the information exponent.
  - Reusing samples creates a non-correlational term, so SGD implements a full SQ algorithm rather than only correlational SQ (snippet).
  - [arXiv 2406.01581](https://arxiv.org/pdf/2406.01581); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/6bd5fca2074dcd9ede9de50f71f7ec28-Paper-Conference.pdf)
- Related 2024–2026 results show the following (snippet; the extract did not say which of these two papers states the n≃d vs Ω(d²) comparison):
  - multi-index functions become efficiently learnable by two-layer nets "when data reusing is permitted", whereas single-pass SGD learns only staircase ones;
  - "SGD with reused data reaches low test error using roughly n ≃ d samples, whereas online SGD fails to achieve even weak recovery with much larger sample size n = Ω(d²)".
  - [Data repetition allows SGD to learn high-dimensional multi-index functions, arXiv 2405.15459](https://arxiv.org/pdf/2405.15459); [Full-batch GD outperforms one-pass SGD, arXiv 2602.02431](https://arxiv.org/pdf/2602.02431)

**More compute → less data (empirical exchange rates for LMs)**
- Muennighoff et al. (NeurIPS 2023), "Scaling data-constrained language models":
  - Up to ~4 epochs of repeated data is almost as good as unique data.
  - The value of repeated tokens decays with a fitted half-life **R_D* ≈ 15** epochs, and returns approach zero with much more repetition. **(from memory, verify; R_D*≈15 confirmed by the snippet below)**
  - [arXiv 2305.16264](https://arxiv.org/abs/2305.16264)
- Prabhudesai et al. (NeurIPS 2025), "Diffusion beats autoregressive in data-constrained settings":
  - Muennighoff found R_D* ~ 15 for AR; masked diffusion models reach **R_D* ~ 500**. A second extract gives "≈513 epochs vs 32 for AR" in the authors' own fits.
  - Diffusion's best loss comes at 500 epochs, AR's at ~50. AR wins at low compute near Chinchilla-optimal; diffusion wins beyond a **critical compute threshold** for which they give a closed form.
  - Their hypothesised mechanism is random-order factorisation acting as implicit data augmentation (snippet).
  - The exact threshold exponent was not retrieved.
  - [arXiv 2507.15857](https://arxiv.org/abs/2507.15857); [AIhub summary](https://aihub.org/2025/10/03/diffusion-beats-autoregressive-in-data-constrained-settings/)
- Kim, Kotha, Liang, Hashimoto (ICLR 2026), "Pre-training under infinite compute":
  - Plain epoching and parameter scaling overfit. Optimal weight decay is **30× larger** than standard practice.
  - Ensembling independently trained models gives a lower loss asymptote than the best single regularised model.
  - Combining epoching, regularisation, parameter scaling and ensemble scaling reaches the asymptote at 200M tokens with **5.17× less data** than baseline, and data-scaling laws predict the gain persists at larger token budgets.
  - Distilling the ensemble into an **8× smaller** student keeps **83%** of the ensembling benefit (snippet).
  - [arXiv 2509.14786](https://arxiv.org/abs/2509.14786); [ICLR 2026 via ML Anthology](https://mlanthology.org/iclr/2026/kim2026iclr-pretraining/)

**Train-time vs test-time compute**
- Epoch AI, "Trading off compute in training and inference", covering five techniques (model scaling, MCTS, pruning, resampling, chain of thought):
  - Overtraining spends **~2 OOM extra training compute to save ~1 OOM of inference** at Chinchilla-level performance.
  - Resampling in code generation saves **1 OOM of training by spending ~1.5 OOM more inference**.
  - A small model can mimic a model **1 OOM larger** with **~3 OOM more inference**.
  - Rule of thumb: each technique trades ~1 OOM of one for somewhat more than 1 OOM of the other. Combined trade-offs typically span only a few OOM (snippet).
  - [Epoch blog](https://epoch.ai/blog/trading-off-compute-in-training-and-inference); [LessWrong overview](https://www.lesswrong.com/posts/hDuoHuBjxGue7fwhJ/trading-off-compute-in-training-and-inference-overview)
- Jones (2021), "Scaling scaling laws with board games" (AlphaZero on Hex):
  - Train-time and test-time compute can be traded off along a simple relationship, and the compute frontier shifts predictably with board size (snippet).
  - The specific figure I recall, "each additional 10× of train-time compute eliminates about 15× of test-time compute", is **(from memory, verify)**. A search did not confirm it.
  - [arXiv 2104.03113](https://arxiv.org/abs/2104.03113)
- Snell et al. (2024): compute-optimal test-time scaling is >4× more efficient than best-of-N. On easy and medium problems, FLOPs-matched test-time compute beats a ~14× larger model; on the hardest problems, pretraining compute remains more effective. **(from memory, verify)** — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- Sardana & Frankle (2024), "Beyond Chinchilla-optimal": once expected inference demand is counted, the end-to-end optimum is a smaller model trained on more tokens than Chinchilla. **(from memory, verify)** — [arXiv 2401.00448](https://arxiv.org/abs/2401.00448)

### Inferences
- **Direction matters for the report's ledger (data free, compute costed).** The theorem-level direction "more data → exponentially less compute" is the relevant one for minimising compute: where data exist, consume them cheaply. The reverse direction, spending compute to save data, is what the brain does and what data-constrained LM methods do. Measured AR-LM exchange rates are poor:
  - repetition is nearly free up to ~4× data savings;
  - the AR ceiling is roughly 1 + R_D* ≈ 16× effective data (my reading of Muennighoff's functional form);
  - diffusion stretches this to ~500× only at compute far beyond Chinchilla-optimal;
  - regularisation plus ensembles give ~5×.
  - Current methods therefore buy roughly **1–2.5 OOM of data efficiency with ≥1–2 OOM of extra compute**. The brain seems to buy ~5 OOM of language-data efficiency for ~4–5 OOM more compute per sample, which is roughly *iso-compute*.
- **Not always a trade.** Lee et al. 2024 is a theoretical counterexample to "conservation of compute". In single-index models, data reuse makes *both* samples and runtime ≈ d·polylog d, which is near-optimal on both axes at once. Some sample-efficiency gains are genuine algorithmic improvements rather than moves along a frontier.
  - Theory therefore does not forbid a learner that is both brain-level sample-efficient and LLM-level compute-efficient per sample. It only says that for SQ-hard structure (Q1) no such free lunch exists.
- **Replay has a proof-level analogue.** "Data reuse turns correlational queries into full statistical queries" is a formal counterpart of the neuroscience idea that hippocampal replay extracts more from each experience. It supports putting replay or multi-pass consolidation into a low-compute learner *where data are scarce*.
- **Few-OOM limit on test-time substitution.** Train/test trade-offs are ~1:1 to 1:1.5 in OOM and span only a few OOM. Test-time search cannot substitute for a missing prior beyond ~1–3 OOM (Epoch: 3 OOM of inference to mimic 1 OOM more parameters). This fits Levin-style reasoning: search cost grows exponentially in the "missing bits" of the prior.

### Gaps
- No theorem gives an exchange rate between samples and compute for realistic LM-like distributions. All quantitative rates above are empirical, measured at ≤ a few billion parameters.
- The closed-form critical-compute formula of Prabhudesai et al. and Jones's exact train/test ratio could not be retrieved (arxiv and github.io blocked).
- No study measures an iso-loss curve spanning ~5 OOM of data reduction for LMs, which is what the brain comparison would require.

## Q3. Expressivity and depth: TC⁰ limits, CoT and looping, and the inference compute that reasoning requires

### Takeaway
- **Fixed-depth transformers**, and also fixed-depth state-space models, with log precision sit inside **uniform TC⁰**: highly parallel but shallow computation. Unless TC⁰ = NC¹ or TC⁰ = P, which is widely believed false but unproven, they cannot in one forward pass solve inherently serial problems such as state tracking, circuit evaluation or general arithmetic.
- **Chain of thought** raises expressivity with the number of steps: linear steps add all regular languages, polynomial steps give **exactly P**.
- **Looping and depth** do the same without extra parameters: padded transformers with O(logᵈ n) loops = TC^d, polylog loops → NC.
- There are proven **lower bounds on CoT length**: linear for parity, the middle bit of multiplication, median and reachability, in hard-attention models.
- Empirically, a k-layer block looped L times ≈ a kL-layer model on reasoning, and looping improves knowledge *manipulation*, not knowledge *capacity*.
- Theory therefore says **reasoning costs serial depth, not parameters**. Inference compute per task has a floor ≈ (active parameters per step) × (number of serial steps the problem inherently needs).

### Cited Findings
**Fixed-depth limits**
- Merrill & Sabharwal (TACL 2023), "The parallelism tradeoff": log-precision transformers are in (log-space-)uniform TC⁰, so they cannot solve P-complete problems unless TC⁰ = P. **(from memory, verify)** — [arXiv 2207.00729](https://arxiv.org/abs/2207.00729)
- Merrill, Petty, Sabharwal (ICML 2024), "The illusion of state in state-space models": S4/Mamba-style SSMs are also in TC⁰ and cannot express state tracking such as composing permutations (the NC¹-complete S5 word problem). Being recurrent in form does not make them recurrent in power. **(from memory, verify)** — [arXiv 2404.08819](https://arxiv.org/abs/2404.08819)
- Feng et al. (NeurIPS 2023), "Towards revealing the mystery behind chain of thought": bounded-depth transformers cannot directly output answers to basic arithmetic or equation tasks unless model size grows super-polynomially with input length. Constant-size autoregressive transformers with CoT can, and CoT also handles dynamic programming. **(from memory, verify)** — [arXiv 2305.15408](https://arxiv.org/abs/2305.15408)

**Chain of thought**
- Merrill & Sabharwal (ICLR 2024), "The expressive power of transformers with chain of thought": "a linear number of decoding steps adds the ability to recognize all regular languages"; with "polynomial steps with generalized pre-norm", transformers "recognize exactly the class of polynomial-time solvable problems — the first exact characterization of a type of transformers in terms of standard complexity classes" (snippet). — [arXiv 2310.07923](https://arxiv.org/abs/2310.07923); [ICLR 2024](https://mlanthology.org/iclr/2024/merrill2024iclr-expressive/)
- Li, Liu, Zhou, Ma (ICLR 2024), "Chain of thought empowers transformers to solve inherently serial problems": constant-depth transformers with T CoT steps can compute any function computable by Boolean circuits of size T. **(from memory, verify)** — [arXiv 2402.12875](https://arxiv.org/abs/2402.12875)
- Bavandpour (Amiri), Huang, Rofin, Hahn (ICML 2025), "Lower bounds for chain-of-thought reasoning in hard-attention transformers":
  - They give the first systematic lower bounds on the number of CoT steps, tight up to log factors.
  - Central theorem: if a function has a sublinear-length CoT in the unique-hard-attention model, fixing a constant fraction of input positions already fixes the output.
  - Hence **PARITY, the middle bit of multiplication, MEDIAN and DAG reachability force linear (or larger) CoT length**. The prefix-parity CoT uses Θ(N) steps (snippet).
  - [arXiv 2502.02393](https://arxiv.org/abs/2502.02393); [ICML 2025](https://icml.cc/virtual/2025/poster/45425)

**Looping and padding**
- Merrill & Sabharwal (NeurIPS 2025), "Exact expressive power of transformers with padding" (snippet):
  - fixed-depth transformers with polynomial padding recognise exactly FO-uniform **TC⁰**;
  - padded transformers with **O(logᵈ n) looping** recognise exactly FO-uniform **TC^d**;
  - "with polylogarithmic looping, padded transformers converge to the class NC, the best that could be expected without losing parallelism".
  - [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118324); [ML Anthology](https://mlanthology.org/neurips/2025/merrill2025neurips-exact/)
- Merrill & Sabharwal (2025), "A little depth goes a long way": depth growing as Θ(log n) suffices for regular-language recognition and graph connectivity. That is far cheaper than getting the same power by scaling width, which would need super-polynomial growth, or CoT length. **(from memory, verify)** — [arXiv 2503.03961](https://arxiv.org/abs/2503.03961)
- Sanford, Hsu, Telgarsky (2024): logarithmic depth is necessary and sufficient for k-hop induction-type tasks and connects transformers to massively parallel computation. **(from memory, verify)** — [arXiv 2402.09268](https://arxiv.org/abs/2402.09268)
- Saunshi, Dikkala, Li, Kumar, Reddi (ICLR 2025), "Reasoning with latent thoughts: on the power of looped transformers" (snippet):
  - "many reasoning problems require a large depth but not necessarily many parameters";
  - on addition, p-hop induction and math problems, a **k-layer model looped L times nearly matches a kL-layer model** and is much better than the k-layer model;
  - the authors support this with theory that iterative algorithms can be run by looped models at near-optimal depth.
  - [arXiv 2502.17416](https://arxiv.org/abs/2502.17416); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/2676109d49d1eb26d6bc584a8f556305-Paper-Conference.pdf)
  - Caveat: a public GitHub issue argues the paper does not show that looped models *generally* match equal-depth untied models. — [issue #28](https://github.com/microprediction/latentreasoning/issues/28)
- Ouro / LoopLM (Oct 2025), "Scaling latent reasoning via looped language models":
  - pretrained looped LMs, scaled to **7.7T tokens**, with an entropy-regularised objective for learned depth;
  - **Ouro 1.4B (4 recurrent steps) matches or exceeds 4B standard models**, and 2.6B matches 8B dense models on reasoning; MATH500 reaches 82.4 for Ouro-1.4B vs 59.6 for Qwen3-4B;
  - the advantage "stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities". All figures are self-reported (snippet).
  - [arXiv 2510.25741](https://arxiv.org/abs/2510.25741); [project page](https://ouro-llm.github.io/)
- Giannou et al. (2023), "Looped transformers as programmable computers": a constant-size looped transformer can emulate a general-purpose computer, given external memory in the input. **(from memory, verify)** — [arXiv 2301.13196](https://arxiv.org/abs/2301.13196)

**Cost of attention itself**
- Exact softmax attention needs time quadratic in context length in the worst case under SETH, unless entries are bounded (Alman & Song 2023; Keles et al. 2023). **(from memory, verify)** — [arXiv 2302.13214](https://arxiv.org/abs/2302.13214)

### Inferences
- **Floor on inference compute per task** (my framing, grounded in the theorems above): C_task ≳ 2·N_active·S, where S is at least the problem's inherent serial depth.
  - S = Ω(n) for parity-, multiplication- and reachability-type subproblems.
  - Poly(n) CoT steps are needed to reach all of P.
  - One cannot get around this with width: that would need super-polynomial size (Feng et al.) or would break TC⁰ ≠ NC¹ conjectures.
- **Looping as the cheap route to depth.** Recurrence and looping buy depth with O(1) extra parameters. Ouro's self-reported result gives a rough training-compute comparison (my estimate, assuming the loops multiply per-token compute ~4× and Qwen3-4B ≈ 8.6e23 FLOP per the round-2 table):
  - Ouro-1.4B: ≈6 × 1.4e9 × 7.7e12 × 4 ≈ **2.6e23 FLOP**;
  - Qwen3-4B: ≈ **8.6e23 FLOP**;
  - so ~3× less training compute for similar or better reasoning, with ~3× fewer stored parameters.
- **Serial compute is unavoidable.** Theory therefore favours **small, recurrent/looped cores that spend inference-time serial compute** over large feed-forward models *for reasoning*. It gives no advantage for knowledge (see Q6).
- **Which recurrence counts.** SSMs with fixed-depth, diagonal-style recurrence are still TC⁰. The kind that helps is depth-recurrence, or input-dependent non-diagonal state updates. Linear-time "recurrent" LMs do not automatically escape the serial-depth bound.
- **For the report.** TRM (7M, recursive) and Ouro (looped LM) are the empirical side of a real theorem: depth and serial steps, not parameters, bound reasoning power. The price is that inference compute per task rises in proportion to serial steps. Theory offers no way to make hard serial reasoning cheap at inference. It only allows it to be cheap in *parameters*.

### Gaps
- All expressivity results are about *what can be represented*, not what SGD will *learn*. Learnability of looped or CoT solutions is covered only by stylised results (Q1: Kim & Suzuki, Wies et al.).
- No tight lower bound exists on CoT length for "typical" reasoning benchmarks such as competition math or code. Hard-attention lower bounds may not transfer exactly to soft attention.
- TC⁰ ≠ NC¹ and TC⁰ ≠ P are unproven, so every "cannot" above is conditional.

## Q4. In-context learning and meta-learning as amortised inference: when does a big up-front prior beat per-task search?

### Takeaway
- Theory shows transformers can implement learning algorithms in-context (gradient descent, ridge regression, algorithm selection) and can approximate Bayesian posterior predictives (prior-fitted networks). Pretraining is therefore **amortised inference**: a one-time cost that makes each later task's "learning" a forward pass.
- Multi-task and meta-learning theory gives explicit **amortisation arithmetic in samples**. Learning a shared k-dimensional representation in ambient dimension d costs ~dk samples in total across tasks, after which each new task needs ~k samples instead of ~d. Pre-learning pays off after roughly **k tasks**.
- Compute-side trade-offs (Epoch, Jones, Sardana & Frankle) say the end-to-end optimum shifts toward heavier pretraining as the number of deployed tasks grows.
- For a *general* intelligence, where the number of tasks is effectively unbounded and they share structure, theory favours **an amortised learned prior plus modest per-task search**, not a compact learner doing search from scratch on every task.

### Cited Findings
**Transformers implementing learning in-context**
- Garg et al. (2022) showed transformers learn linear functions and other classes in context. **(from memory, verify)** — [arXiv 2208.01066](https://arxiv.org/abs/2208.01066)
- Akyürek et al. (ICLR 2023) and von Oswald et al. (ICML 2023) showed constructions and evidence that transformers implement gradient descent or ridge regression in their forward pass. **(from memory, verify)** — [arXiv 2211.15661](https://arxiv.org/abs/2211.15661); [arXiv 2212.07677](https://arxiv.org/abs/2212.07677)
- Bai et al. (NeurIPS 2023), "Transformers as statisticians": in-context algorithm selection with near-optimal statistical rates. **(from memory, verify)** — [arXiv 2306.04637](https://arxiv.org/abs/2306.04637)

**In-context learning as Bayesian inference**
- Xie et al. (ICLR 2022): in-context learning emerges as implicit Bayesian inference over latent concepts in the pretraining mixture. **(from memory, verify)** — [arXiv 2111.02080](https://arxiv.org/abs/2111.02080)
- Müller et al. (ICLR 2022), "Transformers can do Bayesian inference" (prior-fitted networks): training on samples from a prior makes the network approximate the posterior predictive. **(from memory, verify)** — [arXiv 2112.10510](https://arxiv.org/abs/2112.10510)
- TabPFN, trained only on synthetic tables, beat a 4-hour-tuned strongest-baseline ensemble in 2.8 s. That is amortised Bayesian inference paying off across many tasks. — [Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) (as cited in the round-2 report)

**Amortisation gap**
- Cremer, Li, Duvenaud (2018): an amortised inference network is generally suboptimal relative to per-instance optimisation. The "amortisation gap" is a cost of amortising, which per-instance refinement can close. **(from memory, verify)** — [arXiv 1801.03558](https://arxiv.org/abs/1801.03558)

**Sample-complexity amortisation theorems**
- Baxter (2000) and Maurer, Pontil, Romera-Paredes (JMLR 2016): multi-task representation learning bounds have the form excess risk ≲ (complexity of the shared representation)/√(nT) + (complexity of the task-specific head)/√n. The representation cost is divided across T tasks. **(from memory, verify)** — [Baxter, JAIR 12](https://doi.org/10.1613/jair.731); [Maurer et al., arXiv 1505.06279](https://arxiv.org/abs/1505.06279)
- Du, Hu, Kakade, Lee, Lei (ICLR 2021) and Tripuraneni, Jin, Jordan (ICML 2021): with a shared k-dimensional linear representation in ℝᵈ, target-task risk scales like **~dk/(n₁T) + k/n₂**, where n₁ is samples per source task and n₂ is target samples. Without meta-learning, a new task costs ~d/n₂. **(from memory, verify)** — [arXiv 2002.09434](https://arxiv.org/abs/2002.09434); [arXiv 2002.11684](https://arxiv.org/abs/2002.11684)

**Compute-side amortisation**
- Epoch's train/inference trade-off rule of thumb (~1 OOM for ~1–1.5 OOM) — see Q2. — [Epoch](https://epoch.ai/blog/trading-off-compute-in-training-and-inference)
- Epoch argues that, given such trade-offs, it is roughly optimal to spend comparable amounts of compute on training and inference. **(from memory, verify)** — [Epoch: Optimally allocating compute between inference and training](https://epoch.ai/blog/optimally-allocating-compute-between-inference-and-training)
- Sardana & Frankle (2024): higher inference demand favours smaller, longer-trained models. **(from memory, verify)** — [arXiv 2401.00448](https://arxiv.org/abs/2401.00448)

### Inferences
- **Break-even in samples** (my derivation from the Du/Tripuraneni form). To reach risk ε on each of T tasks:
  - without meta-learning, total samples ≈ T·d/ε;
  - with meta-learning, ≈ dk/ε + T·k/ε;
  - meta-learning wins when T > dk/(d−k) ≈ **k** (for k ≪ d).
  - Pretraining a shared prior pays for itself after about as many tasks as the prior has "degrees of freedom". General intelligence involves vastly more tasks than any plausible k, so amortisation is overwhelmingly favoured.
- **Break-even in compute** (my framing, Levin-style, consistent with theory_limits.md Q4). Suppose each task's solution has description length L bits under an uninformed prior and L′ < L under a learned prior.
  - Per-task search cost falls by ~2^(L−L′).
  - A prior costing C_prior pays off after T > C_prior / (c_search·(2^L − 2^L′)) tasks.
  - Because the savings are *exponential* in the bits the prior supplies, even an expensive prior is recovered after few tasks whenever it removes more than a few tens of bits of per-task uncertainty.
- **"Compact learner + per-task search" vs "big pretrained prior".** Theory supports the compact-search option only when:
  - (i) tasks are few or structurally unrelated (no shared k-dimensional structure to amortise);
  - (ii) the residual per-task uncertainty after a *small* prior is only a few tens of bits (e.g., ARC tasks under a DSL prior); or
  - (iii) verification is cheap and exact, making search cheap per candidate.
  - Open-ended general competence violates (i). Amortisation theory therefore favours a learned prior, and the real question becomes its **minimum size**, which is Q5–Q6.
- **The amortisation gap argues for hybrids.** The best end-to-end system is an amortised prior (forward-pass inference) *plus* per-instance refinement for hard instances, i.e. test-time training or search. This matches ARC 2025 refinement loops and TRM-style recursion.
- **Ledger accounting.** Under the report's rule, the prior's cost must be paid inside the instance ledger unless it is a genome-sized description. Amortisation theory says this cost is *worth paying* when the instance will face many tasks. It does not say it is small.

### Gaps
- No theorem gives the compute (as opposed to sample) break-even for amortised vs per-task learning in realistic settings. The Levin-style argument above is a heuristic.
- The meta-learning theorems assume linear or low-dimensional shared structure. How "k" should be measured for real-world competence is unknown.
- I did not verify the exact form or constants of the Du/Tripuraneni bounds.

## Q5. Compression-based generalisation bounds and simplicity bias: what they say about minimal model size

### Takeaway
- Compression-based PAC-Bayes/MDL bounds are now **non-vacuous for deployed LLMs up to LLaMA2-70B**, using token-level martingale bounds with aggressive compression.
- In the compute-optimal (Chinchilla) regime, the generalisation gap *shrinks* with scale, because loss variance and quantisation error fall while parameters per token stay constant. Larger compute-optimal models integrate new information more slowly than their capacity grows.
- Real-world data are low-Kolmogorov-complexity, and neural nets are simplicity-biased, so one model can generalise across diverse tasks. This weakens NFL as a practical barrier.
- The bounds imply that what a trained model *needs* to store is much less than its raw parameter bits. But they give **no lower bound** on the model size needed for broad competence, only upper bounds on how compressible trained models are.

### Cited Findings
- Lotfi et al. (2023/ICML 2024), "Non-vacuous generalization bounds for large language models": the first non-vacuous compression-based bounds for LLM pretraining, using SubLoRA with quantisation and prediction smoothing. They held only for heavily compressed, lower-quality models. **(from memory, verify)** — [arXiv 2312.17173](https://arxiv.org/abs/2312.17173)
- Lotfi, Kuang, Amos, Goldblum, Finzi, Wilson (NeurIPS 2024 spotlight), "Unlocking tokens as data points for generalization bounds on larger language models":
  - earlier bounds were vacuous at the billion-parameter scale;
  - using martingale properties to count **tokens** (non-IID) rather than documents allows less restrictive compression (Monarch, Kronecker, post-training quantisation);
  - this yields "non-vacuous generalization bounds for LLMs as large as **LLaMA2-70B**", the first for deployed models that generate high-quality text (snippet).
  - [arXiv 2407.18158](https://arxiv.org/abs/2407.18158); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html); [code](https://github.com/YilunKuang/token-bounds-for-llms)
- Finzi, Kapoor, Granziol, Gu, De Sa, Kolter, Wilson (ICLR 2025), "Compute-optimal LLMs provably generalize better with scale" (snippet):
  - the bound decomposes into **parameters per token**, **loss variance** and **quantisation error at a fixed bitrate**;
  - on the compute-optimal frontier, parameters per token stay constant while loss variance and quantisation error fall, so larger models have smaller generalisation gaps;
  - information-theoretically, "the rate at which they can integrate new information grows slower than their capacity on the compute optimal frontier", which explains why larger models quantise better.
  - [arXiv 2504.15208](https://arxiv.org/abs/2504.15208); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4a13396b4c8a90e01dac61d2b0559ef7-Abstract-Conference.html)
- Goldblum, Finzi, Rowan, Wilson (ICML 2024), "The no free lunch theorem, Kolmogorov complexity, and the role of inductive biases":
  - real-world datasets are overwhelmingly low-complexity;
  - neural nets, including *randomly initialised* LMs, prefer low-complexity sequences;
  - a single low-complexity-biased learner can therefore do well across very different domains, and NFL does not bite in practice. **(from memory, verify)**
  - [arXiv 2304.05366](https://arxiv.org/abs/2304.05366)
- Valle-Pérez, Camargo, Louis (ICLR 2019): the parameter–function map of deep nets is exponentially biased toward simple (low-Lempel-Ziv-complexity) functions, which explains generalisation. **(from memory, verify)** — [arXiv 1805.08522](https://arxiv.org/abs/1805.08522)
- Morris et al. (2025), "How much do language models memorize?" (Meta, DeepMind, Cornell, NVIDIA):
  - memorisation splits into unintended memorisation and generalisation;
  - measured **capacity ≈ 3.6 bits per parameter** for GPT-style transformers trained in bf16 on random bitstrings;
  - models memorise until capacity fills, after which grokking or generalisation begins and unintended memorisation falls (snippet).
  - [arXiv 2505.24832](https://arxiv.org/abs/2505.24832); [VentureBeat](https://venturebeat.com/ai/how-much-information-do-llms-really-memorize-now-we-know-thanks-to-meta-google-nvidia-and-cornell)

### Inferences
- **Upper bounds, not floors.** Compression bounds bound how much *information* a trained LLM needs to encode its pretraining competence. That the bounds are non-vacuous after heavy compression means much of a big model's parameter budget is **not** load-bearing information.
  - Finzi et al.'s "integrates information more slowly than capacity grows" says the same from the other side: compute-optimal big models are under-filled.
  - This supports "shrink the model" (quantise, distil, prune) as a compute-saver *after* training. It does not show a small model could *learn* the same function with less compute. Big models may be easier to optimise (Edelman: width = parallel search / lottery tickets).
- **What simplicity bias licenses.** The theoretical licence for a *general* learner is that the world is low-complexity and nets are simplicity-biased (Goldblum et al.). The Legg result (theory_limits.md) says the learner itself must be complex. The two fit together:
  - the world's regularities have short descriptions;
  - an efficient learner of them must carry a sizeable prior or program (Legg);
  - compression bounds say that prior is much smaller than today's parameter counts.
- **A capacity ceiling on bits stored.** Morris et al.'s ~3.6 bits/param is an empirical maximum for storing arbitrary bits in a trained transformer; Allen-Zhu & Li's ~2 bits/param is for structured knowledge (Q6). Any architecture with N parameters holds at most ~2–4 N bits of learned content in practice (bf16's information-theoretic ceiling is 16 N). So parametric memory must scale with the knowledge the system must hold.

### Gaps
- No compression bound yields a *lower* bound on model size for a given competence level. The minimal-size question is answered only empirically (Q6).
- I could not retrieve the exact bits-per-token or bits-per-parameter figures in Finzi et al. or the numeric bounds in Lotfi et al.
- A 2026 "Incompressible Knowledge Probes" paper estimates black-box LLM parameter counts from factual capacity. I saw only its title in search results. — [arXiv 2604.24827](https://arxiv.org/pdf/2604.24827) (snippet: title only)

## Q6. Knowledge vs reasoning: minimum bits and parameters for human-level world knowledge, and the compute to acquire it

### Takeaway
- **Knowledge capacity**: transformers store about **2 bits of factual knowledge per parameter** (Allen-Zhu & Li), and only when each fact is seen **~1000 times**. Raw memorisation capacity is **~3.6 bits/param** (Morris et al.).
- **Human learned knowledge**: Landauer's classic estimate of functional long-term memory is **~1e9 bits**, from people retaining "very nearly two bits per second" over a lifetime.
- **Parameter floor**: human-level knowledge therefore needs only ~**3–5e8 parameters**. Allen-Zhu & Li estimate that a 7B model (1.4e10 bits) could hold more than English Wikipedia plus textbooks.
- **Reasoning is a separate budget**: looping improves knowledge *manipulation*, not capacity.
- **Compute cost**: dense SGD pays for knowledge roughly ∝ K² (my derivation). That puts human-level knowledge at ~1e21–1e22 FLOP and encyclopedic superhuman knowledge at ~1e24, matching empirical frontiers.
- **Theory's recommendation**: store knowledge in external or sparse memory (write-once, ~O(K) compute) and keep a small, deep, looped reasoning core.

### Cited Findings
- Allen-Zhu & Li (2024), "Physics of language models: Part 3.3, knowledge capacity scaling laws" (snippet):
  - GPT-2, LLaMA and Mistral architectures consistently reach **~2 bits of knowledge per parameter**, "even when quantized to int8", when each knowledge piece is exposed **~1000 times**;
  - "a 7B model can store 14B bits of knowledge, surpassing the English Wikipedia and textbooks combined based on their estimation";
  - the paper also studies training duration, architecture, quantisation, MoE sparsity and data signal-to-noise ratio.
  - [arXiv 2404.05405](https://arxiv.org/abs/2404.05405); [alphaXiv overview](https://www.alphaxiv.org/overview/2404.05405v1)
- Further details from the same paper, all **(from memory, verify)**:
  - with only ~100 exposures, capacity falls to ~1 bit/param, and architecture matters more (GPT-2 with rotary beats gated-MLP variants in that regime);
  - int4 quantisation drops capacity to ~0.7 bits/param;
  - MoE loses only a small factor of capacity per *total* parameter;
  - large amounts of "junk" data sharply reduce capacity for useful knowledge, and prepending a domain tag mostly restores it.
  - [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- Allen-Zhu & Li, Part 3.2 ("knowledge manipulation"): models that store facts still fail at inverse search and at classification or comparison without CoT, whatever their size. Knowledge *use* is a distinct capability. **(from memory, verify)** — [arXiv 2309.14402](https://arxiv.org/abs/2309.14402)
- Morris et al. (2025): **~3.6 bits/param** raw memorisation capacity (bf16). — [arXiv 2505.24832](https://arxiv.org/abs/2505.24832)
- Landauer (Cognitive Science 1986), "How much do people remember?" (snippet):
  - people "remembered very nearly two bits per second under all experimental conditions", which "continued over a lifetime would produce somewhat over **10⁹ bits**";
  - other methods, based on the informational demands of memory-based performance, give similar estimates of ~1e9 bits;
  - Landauer speculated that flexible retrieval relies on a large ratio of hardware capacity to functional storage.
  - [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1004_4); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0364021386800143); [Merkle's summary](http://www.ralphmerkle.com/humanMemory.html)
- Ouro LoopLM: looping improves knowledge manipulation, not knowledge capacity (snippet). — [arXiv 2510.25741](https://arxiv.org/abs/2510.25741)
- Cross-reference from theory_limits.md: the Hutter Prize enwik9 record compresses 1 GB of Wikipedia XML to 110,793,128 bytes ≈ 8.9e8 bits. — [Hutter Prize](http://prize.hutter1.net/). Also Byrnes's argument that the brain stores ~1e11 bits of learned, incompressible information. — [Alignment Forum](https://www.alignmentforum.org/posts/LY7rovMiJ4FhHxmH5/thoughts-on-hardware-compute-requirements-for-agi)

### Inferences
All figures in this subsection are my estimates.

- **Parameter floor for knowledge.** N_min ≈ K / b, with b ≈ 2 bits/param (structured facts, well-exposed) up to ~3.6 (raw capacity).

  | Knowledge target K | Parameters needed | Stored as bf16 |
  |---|---|---|
  | Human-level, K ≈ 1e9 bits (Landauer) | ≈ **3–5e8** | ≈ 1 GB |
  | Encyclopedic, K ≈ 1.4e10 bits (Allen-Zhu's Wikipedia + textbooks estimate) | ≈ **4–7e9** | — |
  | Brain-content upper estimate, K ≈ 1e11 bits (Byrnes) | ≈ 3–5e10 | — |

  Human-level knowledge is therefore *not* what forces models to be large. Superhuman breadth, e.g. knowing all of Wikipedia at once, pushes toward ~1e10 parameters or equivalent external memory.

- **Compute to load knowledge by dense SGD** scales ∝ K².
  - Let E be exposures per fact (100–1000), ρ the new-knowledge bits per training token (assumed 0.1–1; unmeasured), and N = K/b. Then D ≈ E·K/ρ and C ≈ 6·N·D ≈ **6·E·K²/(b·ρ)**.
  - **Human-level K = 1e9**:
    - E = 1000, b = 2: C ≈ 3e21 (ρ = 1) to 3e22 (ρ = 0.1).
    - E = 100, b = 1: C ≈ 6e20 to 6e21.
  - **Encyclopedic K = 1.4e10**, E = 1000, b = 2: C ≈ 6e23 (ρ = 1) to 6e24 (ρ = 0.1). This is the same order as DeepSeek-V3's 3.3e24 and the ~1e24 brain lifetime anchor, a useful consistency check.
  - The K² scaling arises because every token's update touches every parameter. **MoE** cuts it by the active fraction; for DeepSeek-V3, 37B/671B ≈ 0.055, or ~18×.
  - **External write-once memory** (retrieval, kNN-LM, episodic stores) cuts it to ~O(K): each fact is stored once at ~1 exposure. Embedding 1e10 tokens with a 1e8-parameter encoder costs ~2e18 FLOP.
  - **Theory-grounded implication**: move knowledge out of dense parameters. That is 2–4 OOM of training compute at encyclopedic K.
- **Per-bit comparison with the brain.**

  | Learner | Compute | Bits stored | Compute per bit |
  |---|---|---|---|
  | Brain | ~1e24 FLOP lifetime | ~1e9 bits (Landauer) | ~1e15 FLOP/bit |
  | DeepSeek-V3 | 3.3e24 FLOP | 671e9 × 2 ≈ 1.3e12 bits (if saturated) | ~2.5e12 FLOP/bit |

  - LLM training is ~2–3 OOM *cheaper per stored bit of knowledge* than the brain.
  - Caveats: most brain compute serves perception and motor control, and V3 is likely not saturated.
  - **Knowledge storage is not where the brain's advantage lies.** Its advantage, if any, is reasoning and generalisation per FLOP, and one-shot episodic storage with ~1 exposure rather than 1000.
- **Reasoning is a separate, depth-limited budget** (Q3). Theory plus the Allen-Zhu 3.2 and Ouro results say:
  - parameters buy knowledge;
  - depth, loops and CoT buy reasoning;
  - manipulation of stored knowledge needs serial computation.
  - A minimum-compute system therefore decouples the two: a large but cheap-to-write memory (external or sparse), and a small, deep, iterated core. The active parameters per reasoning step can be ~1e8–1e9 rather than ~1e10–1e11.

### Gaps
- Landauer's ~1e9 bits measures consciously recallable declarative and recognition memory. It likely understates procedural, perceptual and motor knowledge. Byrnes's ~1e11 is one author's argument. Neither is a measurement of what "human-level general competence" requires.
- The knowledge-per-token density ρ of natural text is unmeasured, so the ρ-dependence above spans 1 OOM.
- The 2 bits/param law was measured on synthetic biographical facts. Transfer to procedural or skill knowledge is untested.
- The minimum parameters for *reasoning* competence at human level has no theoretical bound. Empirical anchors are Ouro-1.4B and TRM-7M (narrow), and both are self-reported or narrow.

## Q7. Bottom line: theory-grounded lower-bound estimates for a human-level general learner, and which architectural features theory says are necessary

### Takeaway
There is no theorem that sets a FLOP floor for general intelligence. Combining the pieces above with explicit assumptions gives a theory-consistent floor for a human-level general system:
- (a) **~3e8–1e9 parameters** of learned content for human-level knowledge, or equivalent external memory (~1e9–1e10 bits), plus a reasoning core that theory says can be small if it is **deep or looped**;
- (b) **~1e21–1e23 FLOP** of training if knowledge is externalised or sparse and a curriculum (human text or CoT) is available, rising to **~1e24** if encyclopedic knowledge is held in dense parameters;
- (c) **~1e12–1e14 FLOP per routine reasoning task**, set by (active parameters) × (serial steps). Hard tasks need search multipliers of 10–1000×.

Theory marks five features as necessary or strongly favoured:
- **unbounded serial depth** (CoT or recurrence);
- **memory that scales with knowledge, preferably write-once external or sparse**;
- **an amortised learned prior**;
- **curriculum/intermediate supervision or active experimentation** to escape SQ-hardness;
- **verification-guided search only for the residual few tens of bits per task**.

Together with the "data free, compute costed" ledger, theory mildly favours a **compact, looped cognitive kernel pretrained on abundant human data, with external memory and search**, over a purely experiential, sample-efficient learner. The experiential agent's decisive theoretical advantage (active queries) matters mainly *beyond* human data.

### Cited Findings
Building blocks, as established above:
- serial-depth necessity and CoT/looping expressivity — [Merrill & Sabharwal 2024](https://arxiv.org/abs/2310.07923); [Merrill & Sabharwal 2025 padding](https://neurips.cc/virtual/2025/poster/118324); [CoT lower bounds, arXiv 2502.02393](https://arxiv.org/abs/2502.02393);
- looped depth ≈ unrolled depth — [Saunshi et al.](https://arxiv.org/abs/2502.17416); [Ouro](https://arxiv.org/abs/2510.25741);
- knowledge capacity 2 bits/param at ~1000 exposures — [Allen-Zhu & Li](https://arxiv.org/abs/2404.05405); 3.6 bits/param raw — [Morris et al.](https://arxiv.org/abs/2505.24832); human ~1e9 bits — [Landauer 1986](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1004_4);
- SQ/parity frontier — [Barak et al.](https://arxiv.org/pdf/2207.08799); [Edelman et al.](https://arxiv.org/abs/2309.03800);
- more data speeds up training — [Shalev-Shwartz et al.](http://proceedings.mlr.press/v22/shalev-shwartz12.html);
- data reuse reaches information-theoretic samples — [Lee et al. 2024](https://arxiv.org/pdf/2406.01581);
- empirical compute→data exchange — [Prabhudesai et al.](https://arxiv.org/abs/2507.15857); [Kim et al.](https://arxiv.org/abs/2509.14786);
- train/inference trade-off — [Epoch](https://epoch.ai/blog/trading-off-compute-in-training-and-inference);
- non-vacuous compression bounds — [Lotfi et al. 2024](https://arxiv.org/abs/2407.18158); [Finzi et al. 2025](https://arxiv.org/abs/2504.15208).

### Inferences
All figures and judgements in this subsection are my estimates or syntheses, with assumptions stated.

**Theory-grounded estimate table** (human-level *general* competence, not superintelligence)

| Quantity | Estimate | Key assumptions | Status |
|---|---|---|---|
| (a1) Memory for knowledge | ≥ K/b: ~3–5e8 params (K ≈ 1e9 bits) up to ~4–7e9 (K ≈ 1.4e10, encyclopedic); or 1e9–1e11 bits external | b ≈ 2–3.6 bits/param; Landauer K | Empirical laws plus a classic psychometric estimate; the capacity bound per parameter is near-hard |
| (a2) Reasoning core (active parameters per step) | ~1e8–1e9 | Depth or looping substitutes for parameters (Saunshi, Ouro); TRM shows 7M suffices for narrow tasks | Heuristic; no theorem gives a minimum |
| (b1) Training compute, knowledge | Dense: ~1e21–3e22 (human K), ~1e24 (encyclopedic K); external/sparse: ≲1e20 for storage itself | C ≈ 6EK²/(bρ); E = 100–1000; ρ = 0.1–1 | My derivation; ρ unmeasured |
| (b2) Training compute, reasoning skills | Unknown. Polynomial if CoT or curriculum data exist; exponential in "globality/leap" if not. Empirical anchors: looped 1.4B at ~2.6e23 matches 4B dense at ~8.6e23; GPT-3.5-level at ~1e23 (DCLM-7B) | SQ/globality theory; Ouro self-report | Theory gives only the qualitative shape |
| (c) Inference per task | ≥ 2·N_active·S: ~2e12–2e14 for N_active = 1e9 and S = 1e3–1e5 serial steps; × search factor 10–1000 on hard tasks | CoT/serial-depth lower bounds; Epoch trade-off rates | Conditional theorems (TC⁰ ≠ P) plus arithmetic |
| Total, human-level general learner | Theory-consistent floor ~1e22–1e23; no theorem forbids ~1e21, but no demonstration below ~1e23 at GPT-3.5 level | Knowledge externalised; abundant curriculum data; looped core | Speculative synthesis |

For comparison: the brain costs ~6e16 FLOP per minute of thought, and R1 costs ~7e14 FLOP per AIME problem (round-2 report).

**Features theory says are necessary or strongly favoured**
1. **Unbounded serial depth at inference**, through CoT, recurrence or looping. Required under TC⁰ ≠ P, with explicit linear lower bounds for parity, multiplication and reachability. Width cannot substitute without super-polynomial size. Consequence: reasoning compute is paid per task at inference, and cannot all be prepaid in training.
2. **Memory that scales with knowledge.** Parametric storage is capped at ~2–4 bits/param and costs ∝ K² compute under dense SGD. Write-once external or episodic memory makes storage ~O(K) and avoids SGD's ~100–1000-exposure requirement. This is the theoretical case for complementary learning systems (hippocampus-like fast store plus slow cortex-like core).
3. **An amortised prior.** Multi-task theory says a shared representation pays for itself after ~k tasks. Levin-style savings are exponential in the bits the prior supplies. A general system facing unboundedly many related tasks must amortise; per-task search from a tiny prior is optimal only for few, unrelated, cheaply verifiable tasks.
4. **Curriculum/intermediate supervision or active experimentation.** Passive gradient (SQ) learners face n^Ω(k)-type costs on high-globality concepts. Decomposed supervision (CoT, textbooks, code) or membership-style queries (experiments) remove the barrier. Human text is a massive free curriculum under the ledger; interaction is the only curriculum available beyond human data.
5. **Replay/data reuse where data are scarce.** Proven to reach information-theoretic sample complexity in stylised models. Empirically worth up to ~16× (AR), ~500× (diffusion, at high compute) and ~5× (regularisation plus ensembles) in data-constrained regimes.
6. **Verification-guided search for the residual.** Useful for the last few tens of bits per task (ARC-style refinement, formal proof). Its cost is exponential in residual uncertainty, so it complements the prior and cannot replace it. Epoch's data put the trade at ~3 OOM of inference to replace 1 OOM of model scale.

**What this implies for the report's ranking** (theory-based adjustments, my judgement)
- **The sample-efficiency argument for the #1 direction (experiential world-model agent) is weaker in compute terms than it looks.** Statistical–computational gaps and the brain's iso-compute position show that sample efficiency is bought with compute. Under a ledger where human data are free, the compute-optimal move is to ingest abundant data cheaply.
- **The #1 direction's theoretical edge is active experimentation**, which provably collapses some SQ-hard problems. It becomes decisive only where no human curriculum exists, i.e. for *superhuman* knowledge. That supports framing the likely winner as a hybrid:
  - a text-pretrained looped kernel (≤1e23–1e24) with external memory;
  - plus an experiential, experimenting world-model loop for knowledge beyond human data.
- **The #2 direction (compact cognitive kernel) gets theoretical support** from three results: knowledge and reasoning decouple (2 bits/param vs depth); the K² cost of dense knowledge acquisition argues for retrieval and external memory; and looped depth ≈ unrolled depth. Its theoretical weakness is continual learning: amortised priors carry an amortisation gap and cannot absorb new tasks without per-instance refinement.
- **Tiny recursive models (#7) are validated as the reasoning *component* (depth, not parameters) but not as a general learner.** They lack the K-scaled memory, and per-task search from a small prior is optimal only for few, unrelated, verifiable tasks.
- **For superintelligence rather than human level**, the knowledge term (b1) grows ∝ K² in dense form and the serial-depth term (c) grows with problem hardness. The cheapest scaling path theory allows is therefore:
  - external memory growing linearly with knowledge;
  - inference-time depth or search growing with problem hardness;
  - a fixed, modest-size amortised core.
- **Superintelligence on a ~1e22–1e24 training budget** is consistent with theory only if its "super" comes mainly from memory breadth (cheap, linear) and inference-time depth or search (paid per task), not from a much larger core. This is a synthesis, not a demonstrated result.

### Gaps
- No theorem lower-bounds the FLOP for human-level general competence. Every floor above is conditional on stylised models, unproven complexity conjectures, or empirical laws measured at ≤ ~10B parameters.
- The two biggest unknowns in the estimate are the knowledge density of natural data (ρ) and the minimum reasoning-core size. No study fixes either.
- No experiment has tested a "small looped core + large external write-once memory + curriculum data" system at 1e22–1e23 FLOP on broad benchmarks, with full ledger accounting. This is the direct empirical test of the theory-favoured design. It complements the "brain-scale compute, child-scale data" experiment flagged in round 2.
- Items marked **(from memory, verify)** should be spot-checked before being quoted with exact constants. These are: the Du/Tripuraneni bounds, Berthet–Rigollet rates, Jones's 10×/15× ratio, Snell's 14× figure, the Allen-Zhu 100-exposure, int4 and MoE details, and Epoch's equal-split claim.
