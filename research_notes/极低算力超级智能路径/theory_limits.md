# Theoretical Foundations and Hard Limits on the Minimum Compute for General Intelligence / ASI (training + inference)

Method note for the report writer: this session's egress proxy blocked full-page fetches from epoch.ai, arxiv.org, lesswrong.com, wikipedia.org, aclanthology.org and coefficientgiving.org. Most findings below therefore come from search-engine extracts of primary sources, linked to the primary URL. Items marked **†** come from the primary paper's abstract or well-known headline numbers that I could not re-fetch in this session. They are standard, widely reproduced figures, but spot-check them before quoting them as exact. All arithmetic in "Inferences" is mine.

## Q1. Biological anchor: brain compute, energy, lifetime training compute and data vs. frontier training runs

### Takeaway
The best-studied estimate puts brain-equivalent compute at roughly 1e15 FLOP/s (plausible range 1e13–1e17, and unlikely above ~1e21) running on ~20 W. A human "lifetime training run" is therefore about 1e24 FLOP (range ~1e22–1e26), with fewer than 1e8 words of language input by age 13. Frontier 2025–2026 training runs (~1e26–1e27 FLOP, ~1.5e13 tokens) already exceed the lifetime anchor by 100–1000x in compute and ~1e5x in text. The large headroom is in **sample efficiency and energy per operation**, not raw FLOP.

### Cited Findings
**Brain compute (FLOP/s)**
- Carlsmith (Open Philanthropy, 2020) consulted 30+ experts and used four methods: mechanistic (neurons × compute per neuron), functional (retina/visual cortex task-equivalence), limit (energy/Landauer), and communication. He judges it "more likely than not that 1e15 FLOP/s is enough" to match the brain given the right software. The mechanistic method gives ~1e13–1e17 FLOP/s for standard neuron signalling plus learning, and the methods overall span roughly 1e15–1e21 FLOP/s. — [Coefficient Giving / Open Phil report](https://coefficientgiving.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/); [EA Forum summary](https://forum.effectivealtruism.org/posts/nGQJEYp5X2pCbeweg/new-report-on-how-much-computational-power-it-takes-to-match)
- Sandberg & Bostrom, *Whole Brain Emulation: A Roadmap* (FHI 2008): the workshop's three most plausible emulation levels need ~1e18, 1e22 and 1e25 FLOPS. The spiking-neural-network level is ~1e18 FLOPS. The full range across levels is ~1e15–1e30 FLOPS, a 15-order-of-magnitude spread. These are **emulation** costs, which are far above **functional-equivalence** costs such as Carlsmith's. — [AI Impacts: Brain performance in FLOPS](https://aiimpacts.org/brain-performance-in-flops/); [ORA Oxford](https://ora.ox.ac.uk/objects/uuid:a6880196-34c7-47a0-80f1-74d32ab98788)
- Sandberg, "Energetics of the brain and AI" (2016): the brain performs ~1e14–1e16 synaptic operations per second (depending on how synapses, the active fraction and firing rates are counted) on ~20 W. Energy is ~1e-10 J per action potential and ~1e-14 J per synaptic transmission, from ~1e-19 J per ATP. — [arXiv 1602.04019](https://arxiv.org/pdf/1602.04019); [Andart II](https://aleph.se/andart2/neuroscience/energetics-of-the-brain-and-ai/)

**Lifetime and evolution anchors (Cotra, "Forecasting TAI with biological anchors", 2020)**
- Lifetime anchor: the compute of a brain growing from birth to ~32 years is a median of ~1e24 FLOP (≈1e15 FLOP/s × ~30 years). — [Epoch: Grokking bio-anchors](https://epoch.ai/publications/grokking-bioanchors); [Alignment Forum mirror](https://www.alignmentforum.org/posts/wgio8E758y9XWsi8j/grokking-forecasting-tai-with-biological-anchors)
- Evolution anchor: the compute performed over evolution since the first neurons is a median of ~1e41 FLOP. It carries 10% weight in Cotra's model. — [Epoch: Grokking bio-anchors](https://epoch.ai/publications/grokking-bioanchors)
- Steven Byrnes argues that "brain-inspired AGI" would need training compute near the lifetime anchor rather than the evolution anchor. — [Alignment Forum: Brain-inspired AGI and the "lifetime anchor"](https://www.alignmentforum.org/posts/W6wBmQheDiFmfJqZy/brain-inspired-agi-and-the-lifetime-anchor)

**Human data budget**
- BabyLM Challenge: the Strict track is capped at 100M words, a developmentally plausible budget for a child by about age 13. The organisers report that models trained on <100M words beat those trained on <10M words, yet "fall short of child-level and LLM competence". At this scale, small design choices (tokenisation, sequence length, optimiser cadence) often matter as much as architecture changes. — [Findings of the BabyLM Challenge, arXiv 2504.08165](https://arxiv.org/abs/2504.08165); [Findings of the Third BabyLM Challenge (2025)](https://aclanthology.org/2025.babylm-main.28.pdf)
- LeCun's visual-bandwidth argument: a 4-year-old has ~16,000 waking hours × 3,600 s × 1e6 optic-nerve fibres × 2 eyes × ~10 bytes/s ≈ **1e15 bytes** of visual input. An LLM trained on ~1e13 tokens sees ≈1e13–2e13 bytes, so the child sees ~50x more raw data. Language bandwidth is <12 bytes/s. — [LeCun on X](https://x.com/ylecun/status/1750614681209983231?lang=en); [LeCun on X (language bandwidth)](https://x.com/ylecun/status/1766498677751787723)
- Byrnes argues the brain stores far less learned, incompressible information (~1e11 bits) than it has synapses (~1e14). The search extract attributes this to him, but I could not confirm the exact wording. — [Alignment Forum: Thoughts on hardware/compute requirements for AGI](https://www.alignmentforum.org/posts/LY7rovMiJ4FhHxmH5/thoughts-on-hardware-compute-requirements-for-agi)

**Frontier training runs, 2023–2026**
- Epoch puts GPT-4 at ~2e25 FLOP, Claude 3.5 Sonnet at ~3e25 and Gemini 1.5 Ultra at ~1e26. — [Epoch: models over 1e25 FLOP](https://epoch.ai/data-insights/models-over-1e25-flop?subset=large-scale)
- Per Epoch, the first model estimated above 1e26 FLOP was xAI's Grok 3 (Feb 2025). Epoch estimates Grok 4's training compute cost ~$0.5B, with energy enough to power a town of ~4,000 Americans. Trend extrapolation gives ~30 models above 1e26 FLOP by the start of 2027. — [Epoch: What did it take to train Grok 4?](https://epoch.ai/data-insights/grok-4-training-resources); [Epoch: model counts above thresholds](https://epoch.ai/blog/model-counts-compute-thresholds)
- Epoch reports GPT-5 was likely trained with **less** total compute than GPT-4.5, because returns to post-training (RL) were higher and OpenAI scaled post-training on a smaller base. Epoch expects GPT-6 to resume the scaling trend. — [Epoch: Why GPT-5 used less training compute than GPT-4.5](https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont)
- A non-primary consultancy source puts frontier 2026 training runs at 1e26–1e27 FLOP, costing $200–500M. This is low reliability and only directionally consistent with Epoch. — [Deluair Consultancy](https://deluair.com/consultancy/insights/frontier-ai-training-cost-2026)
- Frontier training compute has grown ~4–5x per year since 2010. **†** — [Epoch: training compute grows 4–5x/yr](https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year)
- DeepSeek-V3 (Dec 2024) is a 671B-parameter MoE model with 37B parameters active per token, pre-trained on 14.8T tokens. Full training took 2.788M H800 GPU-hours, including 119K for context extension and 5K for post-training, or ~$5.576M at $2/GPU-hour. Llama-3.1-405B used 30.84M GPU-hours for comparable or worse performance. — [DeepSeek-V3 Technical Report, arXiv 2412.19437](https://arxiv.org/abs/2412.19437)

### Inferences
- Lifetime compute at 1e15 FLOP/s is ≈1e24 FLOP over 30 years (≈9.5e8 s) or ≈4e23 FLOP by age 13. Across Carlsmith's mechanistic range it is 1e22–1e26 FLOP.
- **Compute gap**: GPT-4 (~2e25) is ~20x the lifetime anchor, and 2025–26 frontier runs (1e26–1e27) are ~100–1000x. DeepSeek-V3 is ≈6 × 37e9 × 14.8e12 ≈ **3.3e24 FLOP** (my 6ND estimate), within ~3x of the lifetime anchor. Frontier training compute is therefore only 1–3 orders of magnitude above the brain.
- **Data gap**: 1.48e13 tokens against <1e8 words (~1.3e8 tokens) is a **~1e5x** gap in linguistic data. The raw visual stream (1e15 bytes by age 4) is larger than LLM text corpora, so the "data efficiency" gap is modality-specific. The brain learns language from ~1e5x fewer tokens but draws on massive multimodal, embodied, interactive data and ~1e9 years of evolutionary prior.
- **Energy gap**: brain lifetime energy is 20 W × 9.5e8 s ≈ 1.9e10 J ≈ **5.3 MWh**. A 2e25 FLOP run on H100-class hardware (1.4e12 FLOP/J peak, ~40% utilisation) needs ≈3.6e13 J ≈ 10 GWh, **~2,000x** more. Newer 1e26–1e27 runs are 1e4–1e5x more. This is the largest measurable gap, and it is mostly hardware (see Q2).
- The human brain is an **existence proof** that human-level (not super-human) general intelligence can be learned with ~1e24 FLOP-equivalent of training, ~1e15 FLOP/s of inference, ~20 W and ~1e8 words, given the right learning algorithm plus an evolved prior (inductive biases compressed into the genome).

### Gaps
- I could not verify Epoch's exact current FLOP estimate for Grok 4, GPT-5, Gemini 3 or Claude Opus 4.x-class runs (epoch.ai blocked). Recollection is ~5e26 FLOP for Grok 4, but that is **unverified**.
- Cotra's overall median training-compute requirement under 2020 algorithms is reported inconsistently in secondary sources (one extract says ~1e36.5 FLOP, range 1e33–1e41). This is unverified; use the anchors, not the composite median.
- I found no robust primary estimate of the brain's total learned information content; the ~1e11-bit figure is one author's argument.

## Q2. Physical limits: Landauer, energy per operation (GPU vs brain vs theory), reversible computing, remaining hardware headroom

### Takeaway
Current GPUs run at ~1.4e12 FLOP/J (H100). If the brain does ~1e15 FLOP/s on 20 W, it achieves ~5e13 FLOP/J, about 30–40x better. Epoch's first-principles estimate of the CMOS ceiling is ~4.7e15 (FP4) operations/J, about 200x above today. Landauer's bound (kT ln2 ≈ 2.9e-21 J per bit erased at 300 K) sits several further orders of magnitude away and can in principle be evaded by reversible computing, which reached net-energy-recovery test silicon in 2025. Hardware energy efficiency therefore has **~2–3 OOM of headroom within CMOS and more beyond it**, which bounds how cheaply any algorithm can be run.

### Cited Findings
- Landauer's principle: erasing one bit costs at least kT ln2 ≈ 2.8e-21 J at 20 °C, equivalent to ~3.5e20 bit-erasures per joule at 300 K. — [Thermodynamic bounds on energy use in DNNs, arXiv 2503.09980](https://arxiv.org/html/2503.09980v2); [Energetics of the brain and AI](https://arxiv.org/pdf/1602.04019)
- An H100 at 700 W delivering just under 1e15 dense BF16 op/s spends ~7e-13 J per multiply-accumulate, and Blackwell improves on this by ~2x. Current Nvidia chips run at ~5e-13 J/FLOP. — [Sang Lee, "The Joule Cost of Knowing"](https://eigenstateofrna.substack.com/p/the-joule-cost-of-knowing) (secondary; consistent with the Epoch figure below)
- Epoch (Ho, Erdil, Besiroglu 2023), "Limits to the energy efficiency of CMOS microprocessors": modelling switching, interconnect and leakage from first principles gives a geometric-mean maximum of **~4.7e15 FP4 ops/J**, "roughly two hundred-fold more efficient than current microprocessors". The H100 is ~1.4e12 FLOP/J. — [Epoch blog](https://epoch.ai/blog/limits-to-the-energy-efficiency-of-cmos-microprocessors); [arXiv 2312.08595](https://arxiv.org/abs/2312.08595)
- The brain uses ~1e-14 J per synaptic transmission and ~1e-10 J per action potential, against a global average of ~25 nJ per operation for conventional computers. — [Sandberg 2016](https://arxiv.org/pdf/1602.04019)
- "The Cognitive Kardashev Scale" (2026) states that semiconductor AI hardware operates ~1e9x above the Landauer limit. It also reports a gap of up to two orders of magnitude between peak ML inference efficiency and the upper end of biological-brain efficiency bounds. — [arXiv 2605.22840](https://arxiv.org/pdf/2605.22840)
- **Conflicting claim**: a Neuroscience News article reports that the brain processes information at "79% energy efficiency", about 1.26x the Landauer limit. This clashes with Sandberg's ~1e-14 J per synaptic event, which is ~3.5e6 × kT ln2. The 79% figure probably refers to a narrow information-transmission measure, not whole-brain computation. Treat it with caution. — [Neuroscience News](https://neurosciencenews.com/information-processing-brain-efficiency-31090/); contradicted by [Sandberg 2016](https://arxiv.org/pdf/1602.04019)
- Jacob Cannell argues that brains are close to thermodynamic and interconnect limits, and therefore "not (much) more efficient than computers" at equal technology levels. — [LessWrong: Brain Efficiency](https://www.lesswrong.com/posts/xwBuoE9p8GE7RAuhd/brain-efficiency-much-more-than-you-wanted-to-know); [LessWrong: No, human brains are not (much) more efficient than computers](https://www.lesswrong.com/posts/KsKfvLx7nFBZnWtEu/no-human-brains-are-not-much-more-efficient-than-computers)
- Reversible computing: Vaire Computing's "Ice River" test chip (22 nm CMOS, announced March 2025) demonstrated **net energy recovery** in an adiabatic resonator system. Energy-recovery factors were 1.77 for a capacitor array and 1.41 for an adder, against a proof-of-concept threshold of 1.0, at a 500 MHz data frequency with 1 GHz targeted. Promotional claims of "4,000x" AI energy reduction are speculative. — [EE Times](https://www.eetimes.com/vaire-demos-energy-recovery-with-reversible-computing-test-chip/); [Vaire](https://vaire.co/); [311 Institute (speculative claim)](https://www.311institute.com/reversible-computing-breakthroughs-could-reduce-ai-energy-consumption-x4000-fold/)

### Inferences
- Brain efficiency depends on which FLOP/s estimate is used: 1e13 → 5e11 FLOP/J (worse than an H100), **1e15 → 5e13 FLOP/J (~36x an H100)**, 1e17 → 5e15 FLOP/J (≈ the CMOS ceiling). The widely cited "brain is 1e6x more efficient" claims are not supported under Carlsmith's central estimate. The brain's advantage is plausibly 1–2 OOM, consistent with the Kardashev-scale paper.
- On the Landauer ceiling for a 20 W brain: 20 W / 2.9e-21 J ≈ **7e21 irreversible bit-operations/s**. This is the physical basis of Carlsmith's "limit method" upper bound near 1e21 FLOP/s.
- Hardware headroom: ~200x (Epoch's CMOS limit at low precision) plus ~2x/yr from Blackwell-class and low-precision gains in the near term. Beyond CMOS, adiabatic/reversible logic could in principle reach below-Landauer dissipation per logical operation, but it is at the ~1.4–1.8x energy-recovery prototype stage. For 2026–2035 planning, **assume at most ~2–3 OOM** of hardware energy-efficiency gain.
- Implication for the "extremely low compute" question: the energy floor sits roughly where the brain already operates. The large remaining multipliers therefore have to come from algorithms (fewer operations per unit of capability), not from making each operation cheaper.

### Gaps
- No reliable published estimate converts Landauer's per-bit bound into a per-FLOP (e.g., FP8 MAC) bound for irreversible digital logic; figures vary with the accounting.
- No peer-reviewed data exists yet on reversible or adiabatic chips running ML workloads at scale.
- I did not find verified 2026 figures for Blackwell or Rubin FLOP/J; the "~2x over H100" figure is from a secondary source.

## Q3. Measured algorithmic progress ("intelligence per FLOP"): pretraining, vision, post-training, inference, and dramatic jumps

### Takeaway
Headline measurements say the compute needed for a fixed capability falls **~3x per year** in pretraining (8-month halving). Vision shows ~9-month halving. Post-training adds one-off 5–20x compute-equivalent gains at less than 1% of the cost, and inference prices for fixed capability fall **~10–200x per year**. A key 2025 correction (Gundlach et al.) is that most measured pretraining efficiency gains are **scale-dependent**: ~91% comes from LSTM→Transformer and Kaplan→Chinchilla re-balancing, whose benefit grows with compute. At small scale, total 2012–2023 gains are **under 100x**, not the ~22,000x headline. The trend is real, but it does not straightforwardly make tiny-compute ASI possible.

### Cited Findings
**Pretraining (language)**
- Ho, Besiroglu et al. (Epoch/MIT, 2024): the compute needed to reach a fixed LM performance **halved roughly every 8 months** (95% CI 5–14 months), faster than Moore's law. Shapley analysis attributes **60–95%** of performance gains to scaling compute and data and **5–40%** to algorithms, with algorithms' relative share falling after ~2018. — [arXiv 2403.05812](https://arxiv.org/pdf/2403.05812); [Epoch blog](https://epoch.ai/blog/algorithmic-progress-in-language-models); [MIT CSAIL news](https://www.csail.mit.edu/news/recurrent-networks-gpt-4-measuring-algorithmic-progress-language-models)
- Per a 2025 LessWrong analysis, Dario Amodei (2025) estimated current algorithmic progress at **~4x per year**. Commentary summarises the consensus as "the same performance with ~3x less compute each year". — [LessWrong: The nature of LLM algorithmic progress (v2)](https://www.lesswrong.com/posts/sGNFtWbXiLJg2hLzK/the-nature-of-llm-algorithmic-progress-v2); [Epoch: Software progress topic](https://epoch.ai/topics/software-progress)
- **Gundlach et al., "On the Origin of Algorithmic Progress in AI" (MIT FutureTech, arXiv 2511.21622, Nov 2025)**:
  - Small-scale ablations of key 2012–2023 innovations account for **<10x** of the widely cited ~22,000x efficiency gain. Innovations outside their ablations are estimated at a further <10x, giving a total **<100x**.
  - Two scale-dependent shifts, LSTM→Transformer and Kaplan→Chinchilla rebalancing, account for **91%** of extrapolated gains at the 2025 frontier.
  - Most other innovations give small, scale-invariant gains (<10x combined, <10% of the total).
  - "Algorithmic progress for small-scale models is several orders of magnitude smaller than previously thought", and measured progress rates depend strongly on the reference algorithm.
  - Sources: [arXiv 2511.21622](https://arxiv.org/abs/2511.21622); [HTML](https://arxiv.org/html/2511.21622); [MIT FutureTech Substack](https://mitfuturetech.substack.com/p/on-the-origins-of-algorithmic-progress)
- "Meek Models Shall Inherit the Earth" (arXiv 2507.07931, 2025) argues that diminishing marginal returns to compute scaling, combined with algorithmic efficiency gains, will narrow the gap between frontier and modest-compute models. I have only the abstract-level gist; the page fetch was blocked. — [arXiv 2507.07931](https://arxiv.org/pdf/2507.07931)

**Vision**
- Erdil & Besiroglu (Epoch, 2022): on ImageNet, compute-augmenting algorithmic innovations **halve compute requirements every ~9 months** (95% CI 4–25 months). Algorithms were roughly as important as compute scaling, and most gains are compute-augmenting rather than data-augmenting. — [arXiv 2212.05153](https://arxiv.org/abs/2212.05153)

**Compute-optimal allocation (Chinchilla)**
- Hoffmann et al. 2022: the same compute as Gopher (280B) with a 70B-parameter model and ~4x more data (1.4T tokens) gave Chinchilla, which outperformed Gopher, GPT-3 and MT-NLG and reached 67.5% on MMLU (>7 points over Gopher). The compute-optimal rule is ~20 tokens per parameter, scaling parameters and data equally (a ≈ b ≈ 0.5), whereas Kaplan used a ≈ 0.73. **†** — [arXiv 2203.15556](https://arxiv.org/abs/2203.15556); [Reconciling Kaplan and Chinchilla, arXiv 2406.12907](https://arxiv.org/html/2406.12907)

**Post-training "compute-equivalent gains" (CEG)**
- Davidson, Denain, Villalobos & Bas (2023): post-training enhancements (tool use, prompting, scaffolding, solution selection, data generation) typically give CEGs of **>5x**, sometimes **>20x**, at **<1%** of the original training cost. — [arXiv 2312.07413](https://arxiv.org/abs/2312.07413); [Epoch](https://epoch.ai/publications/ai-capabilities-can-be-significantly-improved-without-expensive-retraining)
- DeepSeek-R1 (Nature, Sept 2025) reports that the RL stage cost **~$294K**, about 512 H800s at $2/hr. R1-Zero ran ~198 h and R1 ~80 h on 648 H800s (sources differ between 512 and 648 GPUs). This sits on top of the ~$6M base model (V3). — [Scientific American](https://www.scientificamerican.com/article/secrets-of-chinese-ai-model-deepseek-revealed-in-landmark-paper/); [The Register (caveats)](https://www.theregister.com/2025/09/19/deepseek_cost_train/); [HyperAI summary](https://hyper.ai/en/news/44332)
- Distillation: DeepSeek-R1-Distill-Qwen-1.5B reaches ~28.9% on AIME 2024 and ~83.9% on MATH-500, beating GPT-4o and Claude 3.5 Sonnet on those math benchmarks. **†** — [arXiv 2501.12948](https://arxiv.org/pdf/2501.12948)

**Inference efficiency (price for fixed capability)**
- Epoch: across six benchmarks, the price of reaching a fixed performance level fell **9x–900x per year** depending on the milestone, with a median of ~50x/yr accelerating to ~200x/yr since Jan 2024. GPT-4-level PhD-science (GPQA) performance got ~40x/yr cheaper. — [Epoch: LLM inference price trends](https://epoch.ai/data-insights/llm-inference-price-trends)
- MIT FutureTech (arXiv 2511.23455): controlling for frontier-model confounds, the price for fixed benchmark performance falls **~5–10x per year** on knowledge, reasoning, math and SWE benchmarks. — [arXiv 2511.23455](https://arxiv.org/html/2511.23455v2)
- Stanford AI Index 2025: GPT-3.5-level MMLU inference fell from $20 to $0.07 per million tokens (Gemini-1.5-Flash-8B), a ~280x drop between Nov 2022 and Oct 2024. The smallest model scoring >60% on MMLU shrank from PaLM (540B, 2022) to Phi-3-mini (3.8B, 2024), a ~142x parameter reduction. **†** — [AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report)
- ARC Prize 2025:
  - The Kaggle-constrained winner hit **24% on ARC-AGI-2 at ~$0.20/task**.
  - Poetiq's refinement harness took Gemini 3 Pro from 31% ($0.81/task) to 54% ($31/task).
  - The 2025 theme was "refinement loops", meaning per-task iterative program optimisation.
  - Sources: [ARC Prize 2025 Technical Report, arXiv 2601.10904](https://arxiv.org/html/2601.10904v1); [ARC Prize results](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- A secondary blog claims the same 87.5% ARC-AGI-1 score cost ~$4,560/task in 2024 and ~$0.30 in 2026. This is unverified. — [R40](https://r40.io/blog/arc-agi-verified-cost-per-task-price-step/)

### Inferences
- Converting halving times to annual rates: an 8-month halving is ≈2.8x/yr (CI 5–14 months → 1.8–5.3x/yr), and a 9-month halving is ≈2.5x/yr for vision. Combined with 4–5x/yr hardware and spend growth, frontier **effective compute** grows ~10–15x/yr.
- **Naïve extrapolation**: at a sustained 3x/yr, bringing a 1e26 FLOP capability down to 1e23 FLOP takes ~6.3 years, and to 1e21 FLOP ~10.5 years. Gundlach et al. imply this does **not** follow automatically, because the biggest historic gains (Transformers, Chinchilla) pay off mostly at large scale. Low-compute ASI needs *different kinds* of innovation (data, objectives, search, priors), not a continuation of the measured curve.
- Post-training, RL and inference-time search are now the cheapest levers, with one-off 5–20x+ CEG at <1% cost (GPT-5 < GPT-4.5 compute; R1 RL ≈ $0.3M). Distillation plus inference-price declines of 10–200x/yr mean **inference** costs for a fixed capability collapse fastest. However, this relies on a large teacher having been trained first, so end-to-end compute is not reduced proportionally.
- Tiny recursive models (TRM, see Q6) and ARC-style refinement loops show that for *narrow* abstract reasoning, capability per FLOP can be 3–4 OOM better than general LLMs.

### Gaps
- I could not access Epoch's 2025–2026 updated software-progress estimates directly (e.g., any revision of the 8-month figure, or RL-era compute-efficiency measures).
- I found no rigorous, peer-reviewed measurement of the "compute-equivalent gain" of reasoning RL (o1/R1-style) in FLOP terms.
- The contribution of data quality (synthetic data, filtering) to measured algorithmic progress is not cleanly separated in published decompositions.

## Q4. Theory of optimal general intelligence (AIXI, Solomonoff, Levin search, Gödel machine, compression) and theoretical lower bounds

### Takeaway
Theory gives a precise picture of an *ideal* general learner: Bayesian prediction over all programs weighted by 2^-length (Solomonoff/AIXI). Its sample efficiency is optimal, with total errors bounded by the environment's Kolmogorov complexity, but it is **incomputable**. Computable approximations (AIXItl, Levin search, Hutter's M, the Gödel machine) are "optimal up to constants" that are exponential in program length. Legg proves that powerful computable predictors must themselves be complex. So theory says ideal generality is data-cheap but compute-expensive, and practical efficiency must come from **strong, learned or evolved priors plus search restricted to structured program spaces**, not from a short universal algorithm. No-free-lunch results show no learner is best over all problems; efficiency comes only from matching the real world's low-complexity structure.

### Cited Findings
- **Solomonoff induction**: expected cumulative prediction error (total mean squared distance from the true distribution) is upper-bounded by the Kolmogorov complexity of the data-generating distribution. Kolmogorov complexity and the universal prior are **incomputable**, though only limit-computable. — [Hutter, "New Error Bounds for Solomonoff Prediction", arXiv cs/9912008](https://arxiv.org/pdf/cs/9912008); [Hutter, "Convergence and Error Bounds for Universal Prediction of Nonbinary Sequences", arXiv cs/0106036](https://arxiv.org/pdf/cs/0106036)
- **Legg (2006), "Is there an elegant universal theory of prediction?"**: Solomonoff's model is incomputable. Powerful computable prediction algorithms exist, but they are **necessarily highly complex**. Beyond moderate complexity their analysis hits Gödel incompleteness, which limits mathematics' ability to analyse intelligent systems. — [arXiv cs/0606070](https://arxiv.org/pdf/cs/0606070)
- **Levin universal search / Hutter's fastest algorithm**: Levin search runs all programs p in parallel, giving program p the time share 2^-l(p). Total time is bounded by 2^l(p) · time_p(x), so it is optimal up to a multiplicative constant exponential in program length. Hutter (2002) describes an algorithm M that solves any well-defined problem as fast as the fastest provably-correct algorithm "save for a factor of 5 and low-order additive terms", where the additive constant is astronomically large. **AIXItl** takes time of order **t·2^l** per cycle. — [Hutter, "The Fastest and Shortest Algorithm for All Well-Defined Problems", arXiv cs/0206022](https://arxiv.org/pdf/cs/0206022); [hutter1.net](https://www.hutter1.net/ai/pfastprg.pdf)
- **Gödel machine** (Schmidhuber 2003): a self-referential universal problem solver that rewrites any part of its own code once it has *proven* the rewrite useful under its utility function. Such self-improvements are "provably optimal", but proof search is itself unbounded in cost. **†** — [arXiv cs/0309048](https://arxiv.org/pdf/cs/0309048)
- **Compression = prediction = intelligence**:
  - Delétang et al. (DeepMind, ICLR 2024): Chinchilla-70B, trained mostly on text, compresses ImageNet patches to 43.4% and LibriSpeech to 16.4% of raw size, beating PNG (58.5%) and FLAC (30.3%). This demonstrates the prediction–compression equivalence. — [arXiv 2309.10668](https://arxiv.org/abs/2309.10668)
  - Hutter Prize (€500K) on enwik9 (1 GB of Wikipedia): the record as of Oct 2025 is **110,793,128 bytes** (archive plus decompressor) by fx2-cmix, set 3 Sept 2024. Winning requires a ≥1% improvement. Prize rules impose tight CPU-only time and memory limits. — [Hutter Prize site](http://prize.hutter1.net/); [FAQ](http://prize.hutter1.net/hfaq.htm); [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html)
- **No-free-lunch**: averaged over all possible problems, all optimisation and search algorithms perform identically (Wolpert & Macready 1997). **†** — [IEEE TEC 1(1):67–82, doi:10.1109/4235.585893](https://doi.org/10.1109/4235.585893)

### Inferences
- **What the theory implies about a compute-efficient general learner**:
  - Sample efficiency is theoretically unbounded above what LLMs achieve. An ideal Bayesian learner's total errors scale with the *description length* of the world's regularities, not with corpus size. This is the formal basis for "~1e8 words should be enough" if the learner has the right prior.
  - Compute efficiency is where theory bites. Exact universal induction is incomputable, and every "universal" approximation pays a 2^(program length) search factor. Useful programs for general intelligence are long (Legg), so brute-force universal search is infeasible at any physical compute. The Landauer-limited budget of a planet-scale computer is still ≪ 2^1000.
  - The escape route both theory and biology suggest is (a) a strong prior that makes relevant programs short, whether evolved (brain) or pre-trained (foundation models), plus (b) guided search and refinement over a restricted program space (ARC refinement loops, program synthesis, RL on verifiable rewards), plus (c) meta-learning that amortises search across tasks. This matches the Gödel-machine and OOPS (Optimal Ordered Problem Solver) intuition of reusing previously found programs.
- The compression framing gives a measurable proxy. Hutter-Prize progress is slow (~1%-step records years apart under strict CPU limits), whereas LLM-based compressors do better but with huge compute. That is evidence that **good compression, and hence prediction, currently trades off steeply against compute**.
- **Lower bounds**:
  - No known theorem sets a minimum FLOP count for "general intelligence", because the task distribution is not formally specified.
  - Known constraints are indirect: incomputability of the ideal; NP-hardness of many sub-problems, so no general polynomial algorithm exists unless P = NP; NFL, which rules out universally superior learners; and Legg's complexity result, which rules out a tiny powerful universal predictor.
  - Together these say ASI cannot be *both* fully general and trivially cheap. It must exploit the specific low-complexity structure of our world.

### Gaps
- I found no rigorous result that lower-bounds the FLOP required for human-level performance on a realistic task distribution.
- I found no quantitative estimate of the size of the brain's "prior" (the genome is ~3e9 base pairs ≈ 6e9 bits raw, a standard figure not verified here), and no connection of it to AIXI-style description-length terms.
- I could not verify the exact current Hutter Prize resource limits (historically ~single core, ~10 GB RAM, ~50 h) because Wikipedia and the prize page could not be fetched.

## Q5. Scaling-law theory: why the exponents are what they are, and what could change them

### Takeaway
Empirical LM exponents are small (Kaplan: loss ∝ C^-0.050, N^-0.076, D^-0.095; Chinchilla: α ≈ 0.34, β ≈ 0.28 in the parametric fit). Halving reducible loss therefore costs ~1e6x compute at α_C = 0.05. Theory ties the exponents to data geometry and statistics: α ≈ 4/d for data-manifold dimension d, and Zipf-distributed "quanta" of knowledge. Because compute cost scales like (loss gain)^(1/α), **changing the exponent is worth vastly more than changing the constant**. Data pruning can in theory beat power laws (even exponentially), and 2024–2025 work suggests data quality changes exponents, not just constants. But the effect is limited in practice at large scale.

### Cited Findings
- Kaplan et al. 2020: the power-law exponents for LM cross-entropy are ≈0.076 (parameters), ≈0.095 (data) and ≈0.050 (compute). Exponents were largely insensitive to architecture details. **†** — [arXiv 2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. 2022 parametric fit: L(N, D) = E + A/N^α + B/D^β, with E ≈ 1.69, α ≈ 0.34, β ≈ 0.28. **†** Kaplan's and Chinchilla's allocation exponents differ partly because Kaplan counted non-embedding parameters and used small models. — [arXiv 2203.15556](https://arxiv.org/abs/2203.15556); [Reconciling Kaplan and Chinchilla, arXiv 2406.12907](https://arxiv.org/html/2406.12907)
- **Bahri, Dyer, Kaplan, Lee, Sharma (PNAS 2024; arXiv 2021), "Explaining neural scaling laws"**: they identify four regimes, variance-limited and resolution-limited, each for data and model size. Variance-limited scaling follows from a well-behaved infinite-data or infinite-width limit. Resolution-limited scaling arises from models resolving a smooth data manifold, and the exponents relate to the manifold's intrinsic dimension (they plot 4/α_D against d). Model- and data-size exponents are linked by a duality through kernel spectra. — [arXiv 2102.06701](https://arxiv.org/abs/2102.06701); [PNAS](https://www.pnas.org/doi/10.1073/pnas.2311878121)
- Sharma & Kaplan (2020): the scaling exponent is α ≈ 4/d, where d is the intrinsic dimension of the data manifold. **†** — [arXiv 2004.10802](https://arxiv.org/abs/2004.10802)
- Zipf and "quanta" explanations of power laws: Hutter (2021) "Learning curve theory" and Michaud et al. (2023) "The quantization model of neural scaling" both derive power laws from power-law (Zipfian) frequencies of features or skills. The loss exponent is set by the frequency exponent. **†** — [arXiv 2102.04074](https://arxiv.org/abs/2102.04074); [arXiv 2303.13506](https://arxiv.org/abs/2303.13506). A 2026 paper derives LM scaling laws directly from natural-language statistics. — [arXiv 2602.07488](https://arxiv.org/pdf/2602.07488); see also [Neural Scaling Laws Rooted in the Data Distribution, arXiv 2412.07942](https://arxiv.org/pdf/2412.07942)
- **Sorscher, Geirhos, Shekhar, Ganguli, Morcos (NeurIPS 2022), "Beyond neural scaling laws: beating power law scaling via data pruning"**: in theory, a high-quality pruning metric can turn power-law error scaling in dataset size into **exponential** scaling. They observed better-than-power-law scaling on CIFAR-10, SVHN and ImageNet (ResNets), and benchmarked ten pruning metrics on ImageNet, proposing a self-supervised metric. — [arXiv 2206.14486](https://arxiv.org/abs/2206.14486); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7b75da9b61eda40fa35453ee5d077df6-Abstract-Conference.html)
- Limitation: "Data pruning and neural scaling laws: fundamental limitations of score-based algorithms" shows score-based pruning fails in high-compression regimes. — [arXiv 2302.06960](https://arxiv.org/pdf/2302.06960)
- **Data quality changes exponents**: several 2024–2025 papers report that higher-quality data raises the model-scaling exponent. Bi et al. 2024 (DeepSeek LLM) found cleaner data shifts optimal allocation toward model size, and a 2025 ACL paper analysing 400+ models identified high data density and non-optimal allocation as sources of "sub-scaling". "gzip Predicts Data-dependent Scaling Laws" finds scaling-law parameters vary systematically with data compressibility. — [ACL 2025: Revisiting Scaling Laws… Role of Data Quality](https://aclanthology.org/2025.acl-long.1163/); [arXiv 2405.16684 (gzip)](https://arxiv.org/pdf/2405.16684); [arXiv 2410.03083 (quality data, parameter-constrained)](https://arxiv.org/pdf/2410.03083)
- **Data-constrained regime**: Epoch estimates the stock of public human text at ~300T tokens (90% CI 100T–1,000T). At current trends it is fully used between 2026 and 2032 (median ~2028). Multi-epoch training remains effective up to ~5 repetitions. — [Epoch: Will we run out of data?](https://epoch.ai/publications/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data); [arXiv 2211.04325](https://arxiv.org/abs/2211.04325)
- Muennighoff et al. 2023: up to ~4 epochs of repeated data is nearly as good as unique data, and returns decay to ~0 with much more repetition. **†** — [arXiv 2305.16264](https://arxiv.org/abs/2305.16264)
- phi-1 ("Textbooks Are All You Need", 2023): 1.3B parameters trained on ~7B tokens of "textbook-quality" filtered and synthetic data reached 50.6% HumanEval pass@1. That rivals much larger models, and is an example of a large constant-factor gain from data quality. **†** — [arXiv 2306.11644](https://arxiv.org/abs/2306.11644)

### Inferences
- How much exponents matter: for a reducible-loss reduction by factor r, compute scales as r^(1/α).

  | Exponent α | Compute to halve reducible loss |
  |---|---|
  | 0.05 (Kaplan compute exponent) | ~1e6x |
  | 0.10 | ~1e3x |
  | 0.20 | ~32x |

  Doubling the exponent is therefore worth ~3 OOM at the same target, while a better constant (e.g., phi-style data quality) gives a fixed 10–100x multiplier.
- If α ≈ 4/d, exponents are small because natural data (text, images) has high effective intrinsic dimension and heavy-tailed (Zipf) feature frequencies. Learning each rare "quantum" takes proportionally more data. Paths to better exponents therefore include:
  - (a) reducing effective dimension through strong priors and structure (world models, symbolic abstraction, compositional representations);
  - (b) reshaping the data distribution through curricula, active learning and pruning that over-sample rare, informative examples;
  - (c) moving from passive prediction to interactive or experimental data (RL, self-play, verification), where the learner chooses informative samples.

  Sorscher's exponential bound is the theoretical ceiling of (b).
- **Evidence so far**: data-quality work shows real exponent shifts, but mostly in the model-size exponent and in modest amounts. No one has demonstrated exponential scaling for large LMs. The most robust large-scale wins remain constant-factor (Chinchilla allocation, filtering, synthetic data, MoE).

### Gaps
- I found no study demonstrating that a curriculum or data-selection method changes the **compute** exponent of frontier-scale LMs by a large factor, e.g., 0.05 → 0.1.
- The 2026 derivations (arXiv 2602.07488) could not be read in full.
- I could not verify the replication status of Chinchilla's parametric fit (Epoch's 2024 replication attempt reportedly found different α and β values).

## Q6. Can ASI plausibly be reached at ≤1e21–1e23 FLOP total, on a single GPU, or at a brain-level energy budget? Who argues this, and with what evidence?

### Takeaway
There is **no empirical demonstration** of general human-level AI at ≤1e23 FLOP. The strongest argument that it is possible is the brain existence proof: ~1e24 FLOP-equivalent lifetime learning (±2 OOM) and ~1e15 FLOP/s at 20 W, which is roughly one H100 of inference. Researchers such as Steven Byrnes argue that a brain-like learning algorithm could yield human-level AGI on a retail gaming GPU, and then ASI with modest compute ("brain in a box in a basement"). Counter-evidence:
- measured algorithmic progress mostly helps at large scale (Gundlach et al.);
- data-limited models (BabyLM) still fall short of children;
- theory requires complex priors or search (Legg, Levin);
- tiny-compute successes (TRM, ARC winners) are narrow.

The ≤1e21–1e23 FLOP range for ASI is **speculation**. ~1e24 FLOP for *human-level* learning is a biologically grounded but unproven anchor.

### Cited Findings
- Byrnes, "Foom & Doom 1: 'Brain in a box in a basement'" (2025): argues for a scenario in which a small team builds a brain-like system that rapidly goes from "unimpressive" to ASI using **very little compute**. His earlier analysis concludes "a retail gaming GPU will probably be plenty for human-level human-speed AGI". — [Alignment Forum: Foom & Doom 1](https://www.alignmentforum.org/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement); [Alignment Forum: Thoughts on hardware/compute requirements for AGI](https://www.alignmentforum.org/posts/LY7rovMiJ4FhHxmH5/thoughts-on-hardware-compute-requirements-for-agi)
- Carlsmith's central estimate is ~1e15 FLOP/s for brain-equivalent inference. An H100 delivers just under 1e15 dense BF16 op/s at 700 W. — [Carlsmith](https://coefficientgiving.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/); [Sang Lee](https://eigenstateofrna.substack.com/p/the-joule-cost-of-knowing)
- Tiny Recursive Model (Jolicoeur-Martineau, Samsung SAIL Montreal, Oct 2025):
  - a 2-layer, **~7M-parameter** network that recursively refines a latent state and answer (up to 16 improvement steps);
  - scores **45% on ARC-AGI-1 and 8% on ARC-AGI-2**, beating DeepSeek-R1, o3-mini and Gemini 2.5 Pro with <0.01% of their parameters;
  - trained on ~4 H100s for ~2 days for **<$500**;
  - won an ARC Prize 2025 paper award.
  - Sources: [arXiv 2510.04871](https://arxiv.org/pdf/2510.04871); [GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels); [ARC Prize 2025 Tech Report](https://arxiv.org/html/2601.10904v1)
- The ARC Prize 2025 Kaggle winner scored 24% on ARC-AGI-2 at ~$0.20/task under strict compute limits. — [ARC Prize 2025 Tech Report](https://arxiv.org/html/2601.10904v1)
- LeCun argues text is too low-bandwidth and too scarce (~1e13 tokens ≈ all quality public text) to learn how the world works, and that redundant sensory data (video) is what self-supervised learning needs. His position favours world models and JEPA-style architectures over scaling LLMs. — [LeCun on X](https://x.com/ylecun/status/1750614681209983231?lang=en); [LinkedIn post](https://www.linkedin.com/posts/yann-lecun_animals-and-humans-get-very-smart-very-quickly-activity-7133567569684238336-szrF)
- Counter-evidence on scale-dependence: algorithmic gains at small scale total <100x (2012–2023), versus ~22,000x extrapolated at the frontier, and 91% of frontier gains come from scale-dependent innovations. — [Gundlach et al., arXiv 2511.21622](https://arxiv.org/abs/2511.21622)
- Counter-evidence on data limits: BabyLM models trained on ≤100M words still fall short of child-level and LLM competence. — [arXiv 2504.08165](https://arxiv.org/abs/2504.08165)
- Counter-evidence on theory: powerful predictors must be complex, so no short elegant universal algorithm exists. — [Legg 2006](https://arxiv.org/pdf/cs/0606070)
- The evolution anchor (~1e41 FLOP) shows how expensive it is to *discover* the brain's algorithm from scratch by blind search, as opposed to *running* it. — [Epoch: Grokking bio-anchors](https://epoch.ai/publications/grokking-bioanchors)

### Inferences
- **Decomposing "end-to-end" compute** (my framing, grounded in the figures above):
  - (i) *Discovery* of the right algorithm or prior: evolution took ~1e41 FLOP; human R&D plus frontier-model assistance is the realistic substitute, and the first discovery likely needs ≥1e26 FLOP of experiments by frontier labs.
  - (ii) *Training* one instance: the brain analogue is ~1e22–1e26 FLOP, centrally ~1e24.
  - (iii) *Inference*: the brain analogue is ~1e13–1e17 FLOP/s, centrally ~1e15 ≈ one current GPU.

  An "extremely low compute" ASI is most plausible as (ii) + (iii) near brain levels, **after** (i) has been paid once, possibly using today's large models as teachers (distillation) or as search engines for better algorithms.
- **Required algorithmic gain**: getting from today's ~1e26-FLOP frontier capability to ≤1e23 FLOP is a 1,000x gain, and to ≤1e21 is 1e5x. At a measured ~3x/yr this is roughly 6–11 years *if* gains transferred to small scale. Gundlach et al. say most historical gains did not, so reaching this range likely requires a paradigm shift in sample efficiency, where the ~1e5x language-data gap is the largest measured inefficiency, rather than continued incremental progress.
- **Superintelligence vs. human-level**: the brain anchor bounds *human-level* learning only. ASI could come from human-level learners that are cheap to run, copy, speed up and parallelise (1 GPU ≈ 1 brain at 1e15 FLOP/s), or from inference-time search and refinement (ARC-style loops, RL on verifiable tasks) layered on a compact learner. Both routes keep training compute near the brain anchor while spending extra *inference* compute. The latter is cheap and falling 10–200x/yr in price for fixed capability.
- Most-supported technical direction for very low end-to-end compute, which is a synthesis rather than an established finding:
  - (1) brain-like or world-model learners with strong structural priors, aimed at the ~1e5x sample-efficiency gap;
  - (2) small recursive or iterative models with test-time refinement and search (TRM, ARC-2025 refinement loops);
  - (3) data curation and active or interactive data to improve scaling *exponents* (Sorscher, data-quality exponent shifts);
  - (4) RL post-training and distillation as the cheapest capability multipliers (5–20x CEG at <1% cost; R1 RL ≈ $0.3M);
  - (5) hardware efficiency toward the ~1e15–1e16 op/J CMOS limit and, longer term, reversible logic.

  AIXI-style theory explains *why* (1)–(3) are necessary: priors plus guided search are the only way to evade the exponential cost of universal search.

### Gaps
- I found no peer-reviewed quantitative argument that ASI specifically (as opposed to human-level AGI) is achievable at ≤1e21–1e23 FLOP. Claims at that level (Byrnes, some Yudkowsky-style "home computer" arguments) are reasoned speculation without empirical demonstration.
- I found no measured result showing a general-purpose model reaching broad human-level competence on ≤1e8 words or ≤1e23 FLOP.
- Direct reads of Byrnes's and LeCun's latest (2026) positions and of Epoch's 2026 updates were blocked. Their latest quantitative claims may differ from the versions summarised here.
