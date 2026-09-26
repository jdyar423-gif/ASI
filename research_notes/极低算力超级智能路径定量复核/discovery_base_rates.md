# Discovery base rates: how and where major AI efficiency and capability breakthroughs were found, and with how much compute (calibrating the "discovery engine" for low-compute general superintelligence)

*Source status (26 Sep 2026): the egress proxy blocked arxiv.org, epoch.ai, icml.cc, openreview, huggingface.co, semanticscholar, lesswrong, EA Forum, pith.science, github.io and alphaxiv. The sources used were (a) raw GitHub READMEs, fetched directly and marked **(README)**; (b) GitHub-hosted arXiv digests and syntheses, marked **(GitHub digest)**; (c) search-engine snippets, marked **(snippet)**; (d) my own knowledge of the original papers, marked **(from memory, verify)**. Every row in the dataset links the original paper. Where the compute figure is from memory, the row says so. FLOP figures marked "est." are my own order-of-magnitude arithmetic (GPUs × time × peak FLOP/s × assumed ~30–40% utilization, or 6·N·D). They should be read as ±3–10×.*

**Important correction to earlier rounds.** arXiv 2507.10618, "Compute Requirements for Algorithmic Innovation in Frontier AI Models", is by **Peter Barnett (MIRI), sole author**. It was presented at the ICML 2025 Workshop on Technical AI Governance. It is *not* by "Fogelson et al." Alex Fogelson is a co-author of the *Gundlach et al.* paper (arXiv 2511.21622). The misattribution appears to come from a secondary synthesis that earlier rounds cited. The report should cite "Barnett (2025)". — [arXiv listing via GitHub digest (Lyken17/arXiv-stats)](https://github.com/Lyken17/arXiv-stats); [abstract via GitHub digest (CSQianDong/Awesome-arXiv-Daily-Reporter)](https://github.com/CSQianDong/Awesome-arXiv-Daily-Reporter); [research-digest summary](https://github.com/memgrafter/research-digests/blob/main/ml_research_analysis_2025/2507.10618_compute-requirements-for-algorithmic-innovation-in-frontier-ai-models_20260210_042037.md); workshop venue from a (snippet) of the [arXiv page](https://arxiv.org/abs/2507.10618); Gundlach author list from [GitHub daily digest](https://github.com/LIHUA919/AI-Agents-Daily-Research/blob/main/data/2025-11-27.md).

---

## Q1. Dataset: landmark ML innovations 2012–2026, with who found them, discovery compute, small-scale visibility and time to frontier adoption

### Takeaway
The dataset has 35 human-discovered landmark innovations and 8 AI-search-discovered ones. About 63% (22/35) of the human-discovered innovations were found in experiments of roughly ≤1e20 FLOP (≤ one 8-GPU node for days to weeks), with the benefit already visible at that scale. About 14% (5/35) were found small but showed their real value only at scale; their adoption lag ran up to 5–6 years. About 23% (8/35) needed large or frontier compute, or access to a frontier-pretrained model, to be discovered at all. Among the ~15 *paradigm-level* items, the split is roughly one third in each class. Over time, the "required large compute" class has become concentrated in the paradigm-level innovations of 2020–2026: in-context learning, Chinchilla, CoT, and RL-for-reasoning (o1/R1).

### Cited Findings

**Classification key.**
- **S**: discovered in small experiments (≲1e20 FLOP, ≲8–16 accelerators), with the benefit clearly visible there.
- **S→L**: originated in small experiments, but the value that made it a frontier standard showed only at scale. Typically there was a long adoption lag.
- **L**: discovery needed large or frontier compute (≳1e22 FLOP of experiments, or a frontier-pretrained model as the substrate).

**Org types.** Acad = university. Small = small company, startup or independent researcher. Big = large industrial lab.

| # | Innovation (first paper) | Year | Originating org (type) | Size of discovery experiments (as reported / est.) | Benefit visible at small scale? | Time to frontier adoption | Class | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | AlexNet (deep CNN on GPUs) | 2012 | U. Toronto (Acad) | 2× GTX 580, ~5–6 days; ~5e17 FLOP (from memory, verify) | Yes. ImageNet top-5 error ~15% vs ~26% for the runner-up (from memory) | <1 yr: ImageNet 2013 entries were CNNs | S | [NeurIPS 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) |
| 2 | Dropout | 2012 | U. Toronto (Acad) | Single GPUs; MNIST/CIFAR/TIMIT, plus use in AlexNet; ≤1e18 (est.) | Yes. Benefit *shrinks* at large-data scale, and modern LLM pretraining mostly drops it (from memory) | Immediate (AlexNet) | S | [arXiv 1207.0580](https://arxiv.org/abs/1207.0580) |
| 3 | Seq2seq (LSTM encoder-decoder) | 2014 | Google (Big) | 8 GPUs, ~10 days (from memory); ~1e19 (est.) | Yes (WMT'14 En-Fr) | ~2 yr (GNMT 2016) | S | [arXiv 1409.3215](https://arxiv.org/abs/1409.3215) |
| 4 | Attention for NMT (Bahdanau) | 2014 | U. Montréal (Acad) | Single GPU, ~5 days per model (from memory); ~1e17–1e18 (est.) | Yes (long-sentence BLEU) | ~2 yr (GNMT); ~3 yr (Transformer) | S | [arXiv 1409.0473](https://arxiv.org/abs/1409.0473) |
| 5 | Adam (→ AdamW 2017, U. Freiburg) | 2014 | U. Amsterdam / U. Toronto (Acad) | MNIST, IMDB, CIFAR-10 on single GPUs; ≤1e17 (est.) | Yes | <2 yr; Adam(W) became the LLM default | S | [arXiv 1412.6980](https://arxiv.org/abs/1412.6980) |
| 6 | Batch Normalization | 2015 | Google (Big) | Inception on ImageNet; ~1e18–1e19 (est.) | Yes ("14× fewer training steps", from memory) | <1 yr in CNNs (ResNet). Not used in Transformers | S | [arXiv 1502.03167](https://arxiv.org/abs/1502.03167) |
| 7 | ResNet / residual connections | 2015 | Microsoft Research Asia (Big) | ImageNet on a multi-GPU node; ~1e19 FLOP for ResNet-152 (from memory, ±3×). CIFAR-10 experiments on 1–2 GPUs | Yes (CIFAR-10, 110 layers) | <1 yr; residuals universal in Transformers from 2017 | S | [arXiv 1512.03385](https://arxiv.org/abs/1512.03385) |
| 8 | LayerNorm (→ RMSNorm 2019, Edinburgh/Zurich) | 2016 | U. Toronto (Acad) | Small RNN experiments; ≤1e18 (est.) | Yes | ~1 yr (Transformer). RMSNorm took ~4 yr (LLaMA 2023) | S | [arXiv 1607.06450](https://arxiv.org/abs/1607.06450) |
| 9 | Transformer | 2017 | Google Brain/Research (Big) | 8× P100. Base 12 h, big 3.5 days; the paper's Table 2 gives training cost **3.3e18 (base) and 2.3e19 (big) FLOP** (from memory, verify) | Partly. SOTA BLEU at lower cost, but the size of the advantage is scale-dependent: **6.28× over LSTM at 1e15 FLOP, extrapolated to ≫100× at frontier scale** (secondary; see Q2) | ~1 yr (GPT-1 Jun 2018, BERT Oct 2018) | S→L | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762); scale numbers: [secondary synthesis of Gundlach et al.](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/algorithmic-efficiency-trends.md) |
| 10 | Sparsely-gated Mixture-of-Experts | 2017 | Google Brain (Big) | LSTM-MoE up to 137B params on multi-GPU clusters (16–128 K40-class, from memory); ~1e19–1e20 (est.) | Modest at small scale; gains grow with data and params | ~6 yr (GShard 2020, Switch 2021; Mixtral, DeepSeek-V2/V3 2023–24) | S→L | [arXiv 1701.06538](https://arxiv.org/abs/1701.06538) |
| 11 | Generative pretraining (GPT-1) | 2018 | OpenAI (then a small lab) | "8 GPUs for 1 month", ~0.96 PF-days ≈ 8e19 FLOP (from memory, verify) | Yes (SOTA on 9/12 tasks, from memory); the *scaling* payoff came later | Months (BERT, GPT-2) | S→L | [OpenAI GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 12 | BERT (bidirectional MLM pretraining) | 2018 | Google (Big) | BERT-Large: 16 Cloud TPUs (64 chips), 4 days; ~3e20 (est., from memory) | Yes (BERT-Base) | Months; later superseded at the frontier by decoder-only models | S (moderate) | [arXiv 1810.04805](https://arxiv.org/abs/1810.04805) |
| 13 | Neural scaling laws (Kaplan) | 2020 | OpenAI (Big) | Models ≤1.5B non-embedding params; sweep total ~1e21 (est., ±10×) | Yes by design: extrapolation *from* small models. But the fitted N:D allocation was wrong at scale | Immediate (GPT-3 design) | S→L | [arXiv 2001.08361](https://arxiv.org/abs/2001.08361) |
| 14 | In-context / few-shot learning at scale (GPT-3) | 2020 | OpenAI (Big) | **3.14e23 FLOP** final run (from memory, verify) | No; few-shot ability strengthens sharply with scale | Immediate | L | [arXiv 2005.14165](https://arxiv.org/abs/2005.14165) |
| 15 | Compute-optimal scaling (Chinchilla) | 2022 | DeepMind (Big) | >400 models of 70M–16B params plus Chinchilla-70B at **~5.76e23 FLOP**; sweep total ~1e24 (est.) | No. Kaplan's small-model sweep got the allocation wrong | Months (LLaMA 2023, which then deliberately over-trained) | L | [arXiv 2203.15556](https://arxiv.org/abs/2203.15556) |
| 16 | RLHF (Christiano et al.) → InstructGPT | 2017 → 2022 | OpenAI + DeepMind (Big) | 2017: Atari/MuJoCo, ≤1e18 (est.). InstructGPT 175B PPO-ptx: **~60 PF-days (~5e21)**, vs 3,640 PF-days for GPT-3 pretraining (from memory, verify) | In RL toy tasks, yes. For LLM usefulness it needed a strong pretrained base; InstructGPT 1.3B was preferred over GPT-3 175B (from memory) | ~5 yr to frontier (ChatGPT, Nov 2022) | S→L | [arXiv 1706.03741](https://arxiv.org/abs/1706.03741); [arXiv 2203.02155](https://arxiv.org/abs/2203.02155) |
| 17 | Chain-of-thought prompting | 2022 | Google Brain (Big) | Inference only on LaMDA-137B, PaLM-540B and GPT-3. Marginal FLOP tiny, but it *required* a ~2.5e24-FLOP pretrained model (PaLM) as substrate (from memory) | No; described as an emergent ability of ~100B-parameter models (from memory) | Immediate; foundation of reasoning models | L (borrowed) | [arXiv 2201.11903](https://arxiv.org/abs/2201.11903) |
| 18 | LoRA | 2021 | Microsoft (Big) | RoBERTa/DeBERTa/GPT-2 plus GPT-3 175B fine-tuning; small to moderate | Yes | ~1 yr (ubiquitous fine-tuning method) | S | [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) |
| 19 | RoPE (rotary position embedding) | 2021 | Zhuiyi Technology, Shenzhen (Small) | Small Chinese RoFormer models; ≤1e19 (est.) | Yes (Gundlach: 1.44×, secondary) | ~1 yr (GPT-J 2021, PaLM 2022, LLaMA 2023) | S | [arXiv 2104.09864](https://arxiv.org/abs/2104.09864); gain: [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/theory-vs-scale-in-ai-progress.md) |
| 20 | SwiGLU (GLU variants) | 2020 | Google (Big), single author | T5-base-size runs; ~5e19 per variant (est.) | Yes (Gundlach: 1.17×, secondary) | ~2 yr (PaLM 2022, LLaMA 2023) | S | [arXiv 2002.05202](https://arxiv.org/abs/2002.05202) |
| 21 | µP / µTransfer | 2022 | Microsoft Research + OpenAI (Big) | Hyperparameters tuned on a ~40M proxy and transferred to GPT-3 6.7B; tuning cost ~7% of pretraining (from memory, verify) | Yes. It is *designed* to make small-scale findings transfer | ~1–2 yr (Cerebras-GPT and others; frontier usage mostly undisclosed) | S | [arXiv 2203.03466](https://arxiv.org/abs/2203.03466) |
| 22 | FlashAttention | 2022 | Stanford (Acad) | Kernel benchmarks plus GPT-2/BERT training on 8×A100; ~1e19–1e20 (est.) | Yes (exact attention, wall-clock speedup) | <1 yr. The README says it was "widely adopted in such a short time after its release" | S | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135); [README](https://github.com/Dao-AILab/flash-attention) |
| 23 | Speculative decoding | 2022–23 | Google Research; DeepMind (Big) | Inference experiments (T5-XXL; Chinchilla-70B) | Yes (it is exact) | ~1 yr; standard in serving | S | [arXiv 2211.17192](https://arxiv.org/abs/2211.17192); [arXiv 2302.01318](https://arxiv.org/abs/2302.01318) |
| 24 | DPO | 2023 | Stanford (Acad) | Models ≤6B; ≤1e20 (est.) | Yes | ~1 yr (Zephyr 2023; Llama 3 2024, from memory) | S | [arXiv 2305.18290](https://arxiv.org/abs/2305.18290) |
| 25 | Mamba / selective SSMs | 2023 | CMU + Princeton (Acad) | Largest run 2.8B params × 300B tokens ≈ 5e21 (6ND, est.); most evidence from ≤1.4B sweeps | Yes at ≤3B | Partial: hybrids (Jamba 2024, NVIDIA Nemotron-H 2025). No pure-SSM frontier model known | S (moderate) | [arXiv 2312.00752](https://arxiv.org/abs/2312.00752); Nemotron-H: [arXiv 2504.03624](https://arxiv.org/abs/2504.03624) |
| 26 | Multi-head Latent Attention (MLA) | 2024 | DeepSeek (mid-size lab, ~10k GPUs) | Developed inside DeepSeek-V2 (236B total / 21B active, 8.1T tokens ≈ 1e24, est.) | KV-cache saving at any scale; quality parity had to be shown at scale | ~0.5–1 yr (DeepSeek-V3; Kimi K2-style models) | L | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) |
| 27 | GRPO → R1-Zero recipe (pure RL elicits long CoT) | 2024 → Jan 2025 | DeepSeek | GRPO on DeepSeekMath-7B, then R1-Zero on DeepSeek-V3-Base (671B MoE; V3 pretraining ~3–4e24 FLOP, est.) | Only *after* the frontier result did small reproductions appear. TinyZero reproduced the "aha moment" on Qwen2.5-3B for "< $30", but "for Qwen2.5-0.5B base, we know it fails to learn reasoning" (README) | Months (industry-wide in 2025) | L | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300); [arXiv 2501.12948](https://arxiv.org/abs/2501.12948); [TinyZero README](https://github.com/Jiayi-Pan/TinyZero) |
| 28 | RL-trained test-time reasoning (o1) | 2024 | OpenAI (Big) | Undisclosed; frontier-scale RL (≥1e25, est.). Small precursors were STaR (2022) and PRMs (2023) | No (scaling curves shown on frontier models) | Months | L | [OpenAI o1 post](https://openai.com/index/learning-to-reason-with-llms/) |
| 29 | Muon optimizer | Oct 2024 | Independent researcher (K. Jordan) with collaborators (Small) | NanoGPT speedrun on **8×H100**; the record that introduced Muon took **24.9 min** (README); ~1e18–1e19 FLOP per run (est.) | Yes. Moonshot's scaling-law study: "∼2× computational efficiency compared to AdamW" (README) | **~9 months**: Kimi K2 (1T params, 15.5T tokens) "Trained with the Muon optimizer" (README) | S | [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt); [Moonlight README](https://github.com/MoonshotAI/Moonlight); [Kimi-K2 README](https://github.com/MoonshotAI/Kimi-K2) |
| 30 | AlphaGo Zero / AlphaZero (self-play + MCTS, tabula rasa) | 2017 | DeepMind (Big) | AlphaZero: ~5,000 TPUv1 for self-play plus 64 TPUv2 for training (from memory). AlphaGo Zero ~3e23 FLOP (from memory, verify) | No; needs massive self-play | Open community replications (Leela, KataGo) took ~1–2 yr and distributed volunteer compute (from memory) | L | [arXiv 1712.01815](https://arxiv.org/abs/1712.01815) |
| 31 | MuZero (learned model + planning) | 2019 | DeepMind (Big) | Board games: 16 TPUv3 for training, 1,000 TPUv3 for self-play (from memory) | No | No LLM-frontier adoption. Niche deployment (video compression, from memory) | L | [arXiv 1911.08265](https://arxiv.org/abs/1911.08265) |
| 32 | DreamerV3 (world-model RL, fixed hyperparameters) | 2023 | DeepMind / U. Toronto (Big + Acad) | One GPU per run, including Minecraft diamonds (from memory, verify); ~1e19–1e20 (est.) | Yes | No LLM-frontier adoption; influential in world-model agents | S | [arXiv 2301.04104](https://arxiv.org/abs/2301.04104) |
| 33 | JEPA (I-JEPA) | 2023 | Meta FAIR (Big) | ViT-H/14 on 16 A100s for <72 h (from memory); ~5e20 (est.) | Yes (label-efficient representations) | No frontier LLM adoption as of 2026 (not verified) | S | [arXiv 2301.08243](https://arxiv.org/abs/2301.08243) |
| 34 | Diffusion models (DDPM; roots in Sohl-Dickstein 2015 and Song & Ermon 2019) | 2020 | UC Berkeley (Acad) | CIFAR-10 on a TPU v3-8 for ~10 h (from memory); ~1e19 (est.) | Yes (CIFAR-10 FID 3.17, from memory) | ~2 yr (DALL-E 2, Imagen, Stable Diffusion in 2022) | S | [arXiv 2006.11239](https://arxiv.org/abs/2006.11239) |
| 35 | HRM (2025) / TRM (2025): tiny recursive reasoners | 2025 | Sapient Intelligence (Small); Samsung SAIL Montréal, single author (Small) | HRM: Sudoku in "~10 hours on a RTX 4070 laptop GPU"; ARC runs "assume an 8-GPU setup" (README). TRM: 7M params, ARC-AGI-1 run on "4 H-100 GPUs", "~3 days" (README), ≤1e21 peak-FLOP-equivalent (est.) | Yes, by construction. TRM reports "45% on ARC-AGI-1 and 8% on ARC-AGI-2" (README, self-reported) | None known at frontier as of Sep 2026 | S | [HRM README](https://github.com/sapientinc/HRM); [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) |

**AI-search / machine-discovered ML innovations (the "AI as discovery engine" rows).**

| # | Innovation | Year | Org | Discovery engine and compute | Gain | Frontier adoption? | Source |
|---|---|---|---|---|---|---|---|
| A1 | Swish / SiLU activation | 2017 | Google Brain | Exhaustive + RL-controller search over activation formulas (search cost not found). SiLU had also been written down by humans earlier (Hendrycks & Gimpel 2016) | Small (beats ReLU on deeper models) | **Yes, indirectly.** SiLU is the gate in SwiGLU (PaLM, Llama, Mistral, Gemma…). "Ramachandran, Zoph, and Le rediscovered it independently in 2017 via a neural-architecture search at Google" (snippet) | [arXiv 1710.05941](https://arxiv.org/abs/1710.05941); [zeroentropy explainer](https://zeroentropy.dev/concepts/silu/) (snippet) |
| A2 | NAS-found CNNs (NASNet, AmoebaNet → EfficientNet) | 2016–2019 | Google Brain | RL/evolutionary NAS; Zoph & Le ~800 GPUs; NASNet ~500 GPUs × 4 days; AmoebaNet ~450 GPUs × 7 days (from memory, verify) | SOTA ImageNet accuracy per FLOP | Vision SOTA 2018–2020, superseded by ViTs. No LLM-frontier adoption | [arXiv 1611.01578](https://arxiv.org/abs/1611.01578); [arXiv 1905.11946](https://arxiv.org/abs/1905.11946) |
| A3 | Evolved Transformer | 2019 | Google Brain | Evolutionary NAS on TPUs. Strubell et al.'s cost estimate was "off by 88X" per Jeff Dean (snippet). Per Patterson et al., its reuse in Meena "saved 48.5 tCO2e", "~15X larger than the energy cost of running the search" (snippet) | Modest | Used in Meena (2020) only | [arXiv 1901.11117](https://arxiv.org/abs/1901.11117); [Patterson et al. arXiv 2104.10350](https://arxiv.org/abs/2104.10350) (snippet) |
| A4 | Primer (squared ReLU, depthwise conv on Q/K/V) | 2021 | Google | Evolutionary search over TensorFlow programs (cost not verified) | "The most effective modification found by Primer's search is using a square ReLU" (snippet) | **Yes, near-frontier.** Nemotron-4 340B and Nemotron-H use "squared ReLU (So et al., 2022)" (snippet). The modded-nanogpt record (15.2 min) uses ReLU² (README) | [arXiv 2109.08668](https://arxiv.org/abs/2109.08668); [Nemotron-4 340B](https://arxiv.org/abs/2406.11704) (snippet) |
| A5 | Lion optimizer | 2023 | Google Brain + UCLA | Symbolic program search over a 45-function space (search cost not found) | Self-reported: "saves up to 5x the pre-training cost on JFT-300M"; "up to 2x compute" on LM; "up to 2.3x" on diffusion (README) | Google search-ads CTR model (README, a line that is now commented out). Not seen as the default in frontier LLMs, which use AdamW or Muon (not verified) | [README](https://github.com/google/automl/tree/master/lion); [arXiv 2302.06675](https://arxiv.org/abs/2302.06675) |
| A6 | AlphaEvolve kernel and scheduling discoveries | 2025 | Google DeepMind | Gemini-powered evolutionary coding agent with automated evaluators | Gemini matmul kernel "speeding up the kernel by 23% and reducing overall Gemini training times by 1%". FlashAttention kernel "32.5%" speedup (snippet) | **Yes, at the frontier** (Gemini training) | [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (snippet); [Medium review](https://artgor.medium.com/paper-review-alphaevolve-a-coding-agent-for-scientific-and-algorithmic-discovery-5732a876c2e2) (snippet) |
| A7 | DiscoRL (meta-learned RL update rule) | 2025 | Google DeepMind | Meta-learning over agent populations. "Discovered within 3 simulations of the agent's lifetimes (200 million steps) per game" (snippet, via prior-round notes) | Disco103 (a 754K-param rule) beats hand-designed rules on held-out benchmarks | No LLM-frontier adoption known | [Nature](https://www.nature.com/articles/s41586-025-09761-x); prior-round [red_team.md](../极低算力超级智能路径深化/red_team.md) |
| A8 | ASI-Arch (autonomous linear-attention architecture search) | 2025 | SJTU / GAIR (Acad) | ~1,773 experiments, ~20k GPU-hours (from memory, verify) | Modest; critiqued (see prior-round notes) | No | [arXiv 2507.18074](https://arxiv.org/abs/2507.18074) |

**Summary counts (my tally of the 35 human-discovered rows).**
- By class: S = 22 (63%), S→L = 5 (14%), L = 8 (23%).
- By originating org:
  - Academic: 10 rows (AlexNet, dropout, attention, Adam, LayerNorm, FlashAttention, DPO, Mamba, DDPM, DreamerV3 co-affiliation).
  - Small company / independent: 3 rows (RoPE, Muon, HRM/TRM).
  - Big lab: 22 rows (GPT-1 counted as big-lab, although OpenAI was small in 2018).
- By period (S / S→L / L):
  - 2012–16: 8 / 0 / 0.
  - 2017–20: 3 / 5 / 3.
  - 2021–23: 9 / 0 / 2.
  - 2024–26: 2 / 0 / 3 (Muon and HRM/TRM as S; MLA, R1-Zero, o1 as L).
- Paradigm-level subset (15 items: AlexNet, attention, Adam, ResNet, diffusion, Transformer, GPT pretraining, MoE, Kaplan scaling, RLHF, GPT-3 ICL, Chinchilla, CoT, AlphaZero, o1/R1 reasoning RL): 5 S, 5 S→L, 5 L.

**Adoption lags.**
- Ideas that were S and visible small: median ~1–2 yr (Transformer ~1, RoPE ~1, FlashAttention <1, DPO ~1, Muon ~0.75, SwiGLU ~2, diffusion ~2).
- S→L ideas: long lags (MoE ~6 yr, RLHF ~5 yr).
- L ideas found by big labs: months (Chinchilla, o1, R1).

### Inferences
- **The small-compute share is high for components, lower for paradigms, and falling for paradigms.** Almost all the constant-factor "tricks" (normalization, optimizers, positional encodings, kernels, PEFT, preference losses) came from ≤1e20-FLOP experiments. Every paradigm-level item from 2020 on in this list except diffusion (in-context learning, Chinchilla allocation, CoT, reasoning RL) needed a frontier-scale model either as the experiment or as the substrate.
- **"Borrowed-compute discovery" is a distinct, growing category.** CoT and TinyZero-style reproductions were cheap at the margin but needed a big pretrained model. In the report's full-pipeline ledger this matters: a compact learner found by poking a frontier model is "cheap discovery on borrowed compute".
- **Small-scale discovery plus scale-dependent value produces long latent periods.** MoE and RLHF sat for 5–6 years. A compact general learner discovered small but only weakly superior at small scale could likewise be ignored for years. One that is decisively better at small scale, like AlexNet, would be adopted within months.
- **The list is biased toward LLM-frontier adoption.** Innovations that looked great small and failed at scale, such as many linear-attention variants and Lion as an LLM default, are under-represented. That inflates the S share among *adopted* innovations but not necessarily among *attempted* ones.

### Gaps
- For big-lab papers, the reported experiments are the final ones. Exploratory compute is not disclosed. Epoch estimates only ~10% of OpenAI's 2024 R&D compute went to final training runs (see Q3), so "discovery compute" for big-lab rows is probably understated by ~10×.
- The compute figures marked "(from memory, verify)" could not be checked against arXiv because of egress blocks. In particular: AlexNet ~5e17; Transformer 3.3e18/2.3e19; GPT-1 0.96 PF-days; GPT-3 3.14e23; Chinchilla 5.76e23; InstructGPT 60 PF-days; AlphaZero hardware; µP 7%; NAS GPU counts.
- The search costs of Swish, Primer and Lion were not found.

---

## Q2. What exactly do Barnett (2025; misattributed earlier as "Fogelson et al.") and Gundlach et al. (2025) find, and what other studies address "innovation discovery vs compute"?

### Takeaway
Barnett catalogs 36 *pretraining* innovations in Llama 3 and DeepSeek-V3. Even a GPT-2-compute cap or an 8×H100 hardware cap "could still have allowed for half" of them, and compute-intensive innovations double their requirements each year. Gundlach et al. find that small-scale ablations explain <10× of the claimed ~22,000× efficiency gain. Scaling experiments account for 6,930×, mostly through the scale-dependent LSTM→Transformer transition. As a result, "algorithmic progress for small models has been far slower than previously assumed." The two results are compatible: *most innovations* are small-discoverable, but *most of the value* sits in a few scale-dependent ones. The lab-panel econometrics of Whitfill & Wu can be read either way (compute and labor as substitutes, σ≈2.58, or as near-perfect complements, σ≈−0.10), depending on whether frontier-scale experiments are assumed necessary.

### Cited Findings
**Barnett (2025), arXiv 2507.10618** (abstract verbatim, via GitHub digest):
- Abstract: "We catalog 36 pre-training algorithmic innovations used in Llama 3 and DeepSeek-V3. For each innovation we estimate both the total FLOP used in development and the FLOP/s of the hardware utilized. Innovations using significant resources double in their requirements each year… Our analysis suggests that compute caps alone are unlikely to dramatically slow AI algorithmic progress. Even stringent compute caps -- such as capping total operations to the compute used to train GPT-2 or capping hardware capacity to 8 H100 GPUs -- could still have allowed for half of the cataloged innovations." — [abstract via GitHub digest](https://github.com/CSQianDong/Awesome-arXiv-Daily-Reporter); [arXiv](https://arxiv.org/abs/2507.10618)
- A review quotes trend lines of "FLOP growing ~2.5x/year, hardware ~2.1x/year". — (snippet) [Pith Review](https://pith.science/paper/2507.10618)
- More details come only from a secondary synthesis that misattributes authorship, so treat them as unverified. They are: "25% of innovations (9 of 36) required negligible training FLOP", mostly systems work such as ZeRO and FlashAttention; "2.53x/year for FLOP"; and "by 2028, projected median requirements reach 10^24 FLOP". — [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/theory-vs-scale-in-ai-progress.md)
- Scope limits:
  - Only *pretraining* innovations are included, with no post-training, RL or reasoning innovations.
  - Only innovations *adopted* in two 2024 models are included, which is a survivorship-conditioned sample.
  - A digest lists as limitations that the "sample size of 36 innovations may not capture full diversity" and that the "focus on large language model pre-training may not generalize".
  - — [research-digests summary](https://github.com/memgrafter/research-digests/blob/main/ml_research_analysis_2025/2507.10618_compute-requirements-for-algorithmic-innovation-in-frontier-ai-models_20260210_042037.md)

**Gundlach, Fogelson, Lynch, Trisovic, Rosenfeld, Sandhu, Thompson (MIT FutureTech), arXiv 2511.21622**. Abstract content via a Chinese translation in a GitHub digest:
- Small-scale ablations of the key 2012–2023 innovations explain **less than 10×** of the estimated 22,000× gain.
- Other innovations in the literature add **less than 10×** more, for a total **under 100×**.
- Scaling experiments showed that LSTM and Transformer have *different exponents* in their compute-optimal scaling laws, "while finding little scaling difference for many other innovations".
- With extrapolation, the authors account for **6,930×**, "with the scale-dependent LSTM-to-Transformer transition accounting for the majority".
- Conclusion: "algorithmic progress for small models has been far slower than previously assumed, and … measures of algorithmic efficiency are strongly reference-dependent".
- Sources: [GitHub digest (fan-jj24)](https://github.com/fan-jj24/fan-jj24.github.io/blob/master/2025/12/01/2025-11-26/index.html); [search snippet of arXiv abstract](https://arxiv.org/abs/2511.21622)

Numbers from secondary syntheses. These are not verified against the paper, and they are internally inconsistent:
- LSTM→Transformer accounts for "68%" of frontier-scale gains, and together with Kaplan→Chinchilla for "91%".
- Transformer gain: "6.28x" at 1e15 FLOP, versus "725x" at 2e23 FLOP in one version and "100x+" in another.
- Individual component gains: SwiGLU 1.17×, pre-RMSNorm 1.87×, RoPE 1.44×, AdamW vs SGD 1.87×.
- All post-2017 scale-invariant innovations combined: "~3.5x" in one version, "1.33x" in another.
- Progress at ~1e18 FLOP: "~20x", "below hardware progress rates".
- Quoted: "limits to compute scaling pose obstacles not only to realizing efficiency gains but also to discovering them."
- Sources: [theory-vs-scale synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/theory-vs-scale-in-ai-progress.md); [algorithmic-efficiency-trends synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/algorithmic-efficiency-trends.md)

**Other studies on innovation vs compute:**
- **Ho, Besiroglu, Erdil et al. (Epoch, 2024).** Compute needed for a given LM performance halves every ~8 months (95% CI 5–14). By Shapley decomposition, 60–95% of gains over 2012–2023 came from compute and data scaling. — [arXiv 2403.05812](https://arxiv.org/abs/2403.05812) (via [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/theory-vs-scale-in-ai-progress.md))
- **Hernandez & Brown (OpenAI, 2020).** Compute to reach AlexNet-level ImageNet accuracy fell 44× from 2012 to 2019, a 16-month halving time. — [arXiv 2005.04305](https://arxiv.org/abs/2005.04305) (via [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/algorithmic-efficiency-trends.md))
- **Epoch on research compute.**
  - OpenAI spent "~$7B on cloud compute in 2024 — mostly R&D". Only "about $500 million – roughly 10% – went to the final training runs", with the rest going to "scaling experiments, synthetic data generation, basic research". (snippet)
  - A companion piece is titled "Final training runs account for a minority of R&D compute spending". (snippet)
  - Sources: [Epoch data insight](https://epoch.ai/data-insights/openai-compute-spend); [Epoch Gradient Update](https://epoch.ai/gradient-updates/r-and-d-vs-training-compute)
- **Whitfill & Wu, "Will Compute Bottlenecks Prevent an Intelligence Explosion?" (arXiv 2507.23181).**
  - Data: a panel of OpenAI, DeepMind, Anthropic and DeepSeek, 2014–2024.
  - "The baseline CES-in-compute regression yields σ = 2.58 (substitutes), while a frontier-experiments re-parameterization yields σ = -0.10, statistically indistinguishable from zero and implying near-perfect complementarity." (snippet)
  - "If progress hinges on frontier-scale experiments, then compute constraints may remain a binding bottleneck." (snippet)
  - Source: [arXiv 2507.23181](https://arxiv.org/abs/2507.23181)
- **Davidson (Forethought).** He concludes "there's a good chance that compute bottlenecks don't slow down a software intelligence explosion until its late stages". He cites Davidson & Houlden (2025): "1.2 doublings of output for every doubling of cognitive input, with a range of 0.4 to 3.6". (snippet) — [Forethought](https://www.forethought.org/research/will-compute-bottlenecks-prevent-a-software-intelligence-explosion)
- **Meta, "Automated LLM Speedrunning Benchmark" (2025).**
  - "Even recent, leading reasoning models, like R1 and o3-mini, when combined with a state-of-the-art agent scaffold, still struggle" to reproduce NanoGPT-speedrun records.
  - The gap "persists even when these strong baseline agents are provided with detailed explanations describing the exact code changes".
  - The benchmark has 19 tasks. (snippet)
  - Sources: [arXiv 2506.22419](https://arxiv.org/abs/2506.22419); [GitHub repo](https://github.com/facebookresearch/llm-speedrunner)

### Inferences
- **Barnett and Gundlach answer different questions.**
  - Barnett counts innovations: half are small-discoverable.
  - Gundlach weights them by frontier-scale value: almost all of the value comes from two scale-dependent shifts.
  - Combined: small-compute research is *productive in count* but historically *weak in value at scale*. Measured at ≤1e18 FLOP, the entire decade's progress is only tens of × (secondary figure).
- **The key disanalogy for the report's question.** A *compact low-compute general learner* is by definition an innovation whose value appears **at low compute**. The Gundlach worry, that small-scale evaluation hides the winners, applies with much less force here, because the target metric (capability at ≤X FLOP) can be measured at X.
  - What still requires compute is *verification that it is general and superhuman at its intended budget*. If X ≈ 1e23–1e24 FLOP, a single verification run costs ~100–1,000 H100s for weeks. That is far beyond academic budgets but trivial for frontier labs.
  - So the search can be small; the final proof is mid-scale.
- **Gundlach's other finding cuts the other way.** Small-model efficiency has improved little in 2012–2023 (only tens of × at ~1e18 FLOP, secondary). Either (a) the low-compute regime is under-explored, because incentives pointed at scale, or (b) big low-compute gains are genuinely hard to find. The historical record cannot distinguish these. TRM/HRM-type results and the NanoGPT speedrun are evidence for (a) in narrow settings. The speedrun reports "Under 75 seconds on 8xH100 (the llm.c GPT-2 replication needed 45 minutes)" and "under 400M tokens (the llm.c GPT-2 replication needed 10B)" (README). That is a ~36× wall-clock and ~25× token gain in ~2 years at 124M scale ([modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)). Much of it is hardware-level or systems engineering, and it transfers only partly to scale.
- **The frontier-experiments-as-complement result (σ≈0) is the strongest quantitative support for "big-compute discoverers win".** It rests on the assumption that the relevant experiments must be near-frontier. For a *low-compute learner*, the relevant experiments are by construction sub-frontier. That moves the problem toward the substitutes regime (σ≈2.6), where cognitive labor (human or AI) substitutes for compute.

### Gaps
- Barnett's per-innovation table was not accessible: which 18 of the 36 needed more than GPT-2 compute, and their FLOP estimates.
- I could not read Gundlach's full text to settle the 1.33× vs 3.5× and 100× vs 725× discrepancies. The report should cite only the abstract-level claims (<10× from ablations, <100× total without scale-dependence, 6,930× with it, majority from LSTM→Transformer).
- I found no study that measures discovery compute for *post-training/RL* innovations systematically, and none that does so for non-LLM paradigms (RL, world models).
- I found no Erdil/Besiroglu study (beyond Ho et al.) that directly estimates the share of innovations requiring frontier compute.

---

## Q3. Who discovered the breakthroughs (academia vs industry), what share required frontier compute, and the trend 2012 → 2026

### Takeaway
Academia and small groups originated most of the 2012–2016 deep-learning toolkit. From 2017, big industrial labs originated most paradigm-level advances, and every compute-bound one. By 2024–2025 industry produced ~90% of *notable models*. Academia remains "the top source of highly cited research", and small groups keep producing adopted components: RoPE, FlashAttention, DPO, Mamba, and Muon (by an independent researcher on 8 GPUs, adopted at the 1T-parameter scale within ~9 months). Big labs spend ~90% of R&D compute on experiments rather than final runs, which structurally favors their discovery of scale-dependent innovations.

### Cited Findings
- AI Index 2025: "Nearly 90% of notable AI models in 2024 came from industry, up from 60% in 2023, while academia remains the top source of highly cited research." (snippet) — [Stanford HAI AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report)
- AI Index 2026: "over 90% of notable models coming from industry… in 2025". "Epoch AI tracked 87 notable model releases from industry in 2025, compared to just seven from all other sources". Training details "have quietly stopped being disclosed for several of the most resource-intensive systems". (snippet) — [Stanford HAI AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)
- Ahmed, Wahed & Thompson (Science 379, 2023): "Industry increasingly dominates the three key ingredients of modern AI research: computing power, large datasets, and highly skilled researchers", and this is "translating into AI research outcomes". (snippet) — [Science](https://www.science.org/doi/abs/10.1126/science.ade2420); [MIT Sloan summary](https://mitsloan.mit.edu/ideas-made-to-matter/study-industry-now-dominates-ai-research)
- OpenAI 2024 compute allocation: ~10% to final training runs, with the rest to experiments, synthetic data and research. (snippet) — [Epoch](https://epoch.ai/data-insights/openai-compute-spend)
- Small-group examples of adoption at frontier scale:
  - Muon was introduced by @kellerjordan0 and @jxbz on 10/04/24 in an 8×H100 speedrun record ([README](https://github.com/KellerJordan/modded-nanogpt)).
  - Moonshot identified two "crucial techniques for scaling up Muon" and found "∼2× computational efficiency" ([Moonlight README](https://github.com/MoonshotAI/Moonlight)).
  - Kimi K2 (1T params, 15.5T tokens) was "Trained with the Muon optimizer" with "zero training instability" ([Kimi-K2 README](https://github.com/MoonshotAI/Kimi-K2)).
- A small-group example that has not (yet) been adopted: TRM, whose README argues "The idea that one must rely on massive foundational models trained for millions of dollars … is a trap" and reports 45%/8% on ARC-AGI-1/2 with 7M params (self-reported). — [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- Dataset-based trend: see the Q1 counts. Academic or small-group origin by period:
  - 2012–16: 5/8.
  - 2017–20: 1/11 (DDPM).
  - 2021–23: 4/11 (RoPE, FlashAttention, DPO, Mamba).
  - 2024–26: 2/5 (Muon; HRM/TRM), none of them paradigm-level so far.
  - Required-large-compute (L): 0/8 → 3/11 → 2/11 → 3/5.

### Inferences
- **Trend for paradigm-level breakthroughs.** From "nearly all small, mostly academic" (2012–16), to "originated small but valued at scale" (2017–20), to "frontier-compute-dependent" (2020–26: ICL, Chinchilla, CoT, reasoning RL). Counting the *established* paradigm-level items in my table whose discovery needed frontier-scale compute or a frontier-scale substrate:
  - 0/4 (2012–16: AlexNet, attention, Adam, ResNet);
  - 2/8 (2017–20: GPT-3 ICL and AlphaZero; 5 of the 8 were S→L);
  - 3/3 (2021–26: Chinchilla, CoT, o1/R1; small n).
  
  If the not-yet-proven small-compute paradigm contenders (Mamba/SSMs, JEPA, DreamerV3, TRM/HRM) are counted as candidates, the 2021–26 frontier-dependent share is ~40–50%. My central estimate for "paradigm-level discoveries that now need frontier compute or a frontier substrate" is **~60–70%**.
- **The trend for components is flat.** About half to two thirds remain small-discoverable (Barnett's ~50%; my 63%). Barnett's "doubling each year" for demanding innovations suggests that the compute-intensive *half* keeps moving further out of academic reach.
- **Why big labs dominate L discoveries.**
  - They hold the substrate: frontier pretrained models.
  - They hold ~10× more experimental compute than final-run compute (Epoch).
  - Their training details are increasingly undisclosed (AI Index 2026), so small groups cannot even build on L discoveries quickly.
- **What the compact-learner question most resembles.** The historical case closest to "a compact learner that wins at low compute" is AlexNet/ResNet/DDPM/Muon-type work: small-compute, often academic or independent, and adopted within months *when the small-scale win is decisive*. The historical case closest to "general superintelligence" is ICL/CoT/o1: frontier-compute discoveries. The target combines both, which is why the base rate is genuinely split.

### Gaps
- There is no systematic dataset of *innovations* (as opposed to *models*) by sector over time. The AI Index and Epoch count notable models, which is a biased proxy for discovery.
- The exact Ahmed et al. figures (e.g., the industry share of the largest models, the PhD-flow percentages) could not be verified beyond the snippet.
- I found no 2026 data on the share of highly cited ML papers from academia vs industry.

---

## Q4. Has any AI system already discovered an ML innovation adopted at the frontier? Compute used vs gain

### Takeaway
Yes, but so far only **constant-factor, component-level** ones:
- Swish/SiLU (2017 automated search), which lives on inside SwiGLU in essentially all Llama-lineage LLMs;
- Primer's squared ReLU (2021 evolutionary search), used in NVIDIA Nemotron-4 340B and Nemotron-H;
- AlphaEvolve's Gemini matmul kernel (23% kernel speedup, 1% of Gemini training time).

No paradigm-level innovation has yet been machine-discovered. DiscoRL (2025) is the closest in kind, a discovered *learning rule*, but it has no frontier adoption. Search costs have ranged from moderate (hundreds to thousands of GPU/TPU-days for NAS) to small multiples of a single training run (DiscoRL: ~3 agent-lifetimes per game). Payback can be large relative to search cost: the Evolved Transformer search was repaid ~15× by a single reuse. As of Sep 2026, lab-internal AI research labor is large and rising: OpenAI reports 3.1 agent-workdays per human workday. Agents still struggled to *reproduce* known speedrun innovations in 2025.

### Cited Findings
- **Swish.** Automated search: "leveraging automatic search techniques to discover new activation functions, using a combination of exhaustive and reinforcement learning-based search". SiLU "has quietly become the default activation inside most modern open-weight transformers — Llama, Mistral, Falcon, Gemma, and PaLM all use it". SiLU was "first written down by Hendrycks and Gimpel in the 2016 GELU paper", so human discovery was parallel. (snippet) — [Semantic Scholar entry](https://www.semanticscholar.org/paper/Searching-for-Activation-Functions-Ramachandran-Zoph/c8c4ab59ac29973a00df4e5c8df3773a3c59995a); [zeroentropy](https://zeroentropy.dev/concepts/silu/)
- **Primer → Nemotron.** "Nemotron-4-340B-Base uses squared ReLU activations in the MLP layers, and Nemotron-H-8B uses squared ReLU activation as well", citing So et al. (snippet) — [Nemotron-4 340B report](https://arxiv.org/abs/2406.11704); [Nemotron-H](https://arxiv.org/abs/2504.03624)
- **Lion.** Discovered "by symbolic program search". Self-reported gains of "up to 5x" pretraining cost (JFT), "up to 2x compute" (LM) and "up to 2.3x" (diffusion). A commented-out README line says it was "deployed in production systems such as Google's search ads CTR model". — [google/automl Lion README](https://github.com/google/automl/tree/master/lion)
- **Evolved Transformer.** Strubell et al.'s emissions estimate was "off by 88X" (Jeff Dean, snippet). Reusing the Evolved Transformer in Meena "saved 48.5 tCO2e… ~15X larger than the energy cost of running the search to discover it" (Patterson et al., snippet). — [arXiv 2104.10350](https://arxiv.org/abs/2104.10350); [Jeff Dean on X](https://x.com/JeffDean/status/1843493504347189746)
- **AlphaEvolve.**
  - It "modified a core matrix multiplication helper in Gemini's architecture, speeding up the kernel by 23% and reducing overall Gemini training times by 1%". On FlashAttention kernels it achieved a "32.5%" speedup. (snippet)
  - The 2026 follow-up reports DNA-sequencing error −30%, AC-OPF feasibility 14%→88%, and other results. (snippet)
  - A secondary claim that the freed compute is "worth an estimated $500 million per year" is unverified.
  - Sources: [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [AlphaEvolve impact](https://deepmind.google/blog/alphaevolve-impact/) (snippets)
- **DiscoRL** (Nature, Oct 2025). A meta-learned RL rule (Disco103, 754K params) outperformed hand-designed rules on unseen benchmarks. Discovery cost was "within 3 simulations of the agent's lifetimes… per game" (snippet). — [Nature](https://www.nature.com/articles/s41586-025-09761-x); details compiled in prior-round [red_team.md](../极低算力超级智能路径深化/red_team.md)
- **Ability to reproduce known innovations (2025).** "No frontier model is capable of reproducing the human-driven speedrun, even when given the pseudocode hints" (R1 / o3-mini era). (snippet) — [arXiv 2506.22419](https://arxiv.org/abs/2506.22419)
- **Lab-internal AI research labor (Sep 2026).**
  - OpenAI announced on 6 Sep 2026 that it reached its "automated research intern" goal, with "3.1 agent-workdays of effort for every workday of human labor".
  - The median researcher runs ">$600 a day of inference on internal agents".
  - The company says it is making progress toward an "automated AI researcher by March of 2028".
  - (Self-reported; snippet.) — [Help Net Security](https://www.helpnetsecurity.com/2026/09/07/openai-research-automation-intern/); [Engadget](https://www.engadget.com/2251859/openai-says-it-reached-its-goal-of-creating-an-automated-research-intern/)
- Prior-round context:
  - METR measured >2× in-lab uplift (Jul 2026).
  - Cunningham et al. (arXiv 2609.15802) judge the feedback loops "not yet self-sustaining".
  - AIDE² showed multi-generation self-improvement of research agents.
  - — [red_team.md](../极低算力超级智能路径深化/red_team.md) and the sources cited there.

### Inferences
- **What AI discovery has found so far, and where.**
  - Its track record is **0 of ~15 paradigm-level innovations** and **~3 of ~40 adopted components**.
  - Every success came where there is (a) a cheap, automatic, trustworthy evaluator at small scale (activation functions on proxy tasks, kernels with exact speed measurements, RL rules on Atari lifetimes) and (b) a well-specified search space.
  - "Train a learner from scratch within budget X and score its generality" is exactly that kind of evaluator, *if* generality can be scored cheaply. That makes the compact-learner problem unusually well suited to AI search, much more than scale-dependent frontier innovations are.
- **Returns on search compute have been good.** Search typically costs tens to thousands of accelerator-days, while payoffs are constant factors reused many times (15× payback for the Evolved Transformer; AlphaEvolve's 1% of Gemini training). A compact general learner would have a payoff orders of magnitude larger, so even very expensive AI-driven search (≥1e26 FLOP of AI-researcher inference plus many small training runs) would be rational for labs.
- **The 2025 speedrun result and the 2026 intern milestone bracket the transition.** In 2025 agents could not reproduce human innovations even with hints. By Sep 2026 agents do most of the code-level research labor at one frontier lab, though still under human direction. "Discovered mainly by AI" in a strict sense (the AI originates the key idea) has not happened for any paradigm-level ML advance as of Sep 2026.

### Gaps
- There is no public accounting of which (if any) 2025–2026 frontier-model innovations were AI-originated. Labs do not disclose this, and the AI Index 2026 notes shrinking disclosure.
- Search compute for Swish, Primer, Lion and DiscoRL could not be verified in FLOP terms.
- No 2026 rerun of the Automated LLM Speedrunning Benchmark with current models was found.

---

## Q5. Bottom line: (a) share of paradigm-level breakthroughs discoverable at small compute, (b) trend, (c) calibrated probability for the compact general learner's discoverer

### Takeaway
- **(a)** Historically about **one third** of paradigm-level breakthroughs were both discovered *and* recognizable at small compute. About **two thirds originated** in small experiments. About **one third needed frontier compute or a frontier model** to be found at all.
- **(b)** Among established *paradigm-level* advances, the share needing frontier compute or a frontier substrate rose from 0/4 (2012–16) to 2/8 (2017–20) to 3/3 (2021–26, small n). The share recognizable at small scale fell from 4/4 to 1/8 to 0/3. The component-level small-discoverable share stayed ~50–65%.
- **(c)** For a compact low-compute general learner, conditional on it being found (by ~2040), my calibrated split is:
  - **(i) small-compute human research ~15–20%**;
  - **(ii) large-compute, human-led research ~30%**;
  - **(iii) AI research systems as the main discoverer ~50% (range 35–65%)**.
  
  This is slightly *below* the earlier ~60–70% for (iii). It keeps AI discovery as the plurality mode, but is more explicit that the target's small-scale evaluability revives the small-compute human route.

### Cited Findings
- Component-level small-discoverability: Barnett, "half of the cataloged innovations" under GPT-2 or 8×H100 caps ([arXiv 2507.10618](https://arxiv.org/abs/2507.10618)); my dataset gives 63% S (Q1 table).
- Value concentration in scale-dependent innovations: Gundlach et al. (<10× explained by small-scale ablations; 6,930× with scaling; majority from LSTM→Transformer) ([arXiv 2511.21622](https://arxiv.org/abs/2511.21622), abstract via [GitHub digest](https://github.com/fan-jj24/fan-jj24.github.io/blob/master/2025/12/01/2025-11-26/index.html)).
- Compute and cognitive labor in lab research: σ = 2.58 (baseline) vs −0.10 (frontier-experiments model) ([arXiv 2507.23181](https://arxiv.org/abs/2507.23181), snippet). Returns to cognitive input ~1.2 (0.4–3.6) ([Forethought](https://www.forethought.org/research/will-compute-bottlenecks-prevent-a-software-intelligence-explosion), snippet).
- AI research labor ramp: 3.1 agent-workdays per human workday at OpenAI, Sep 2026 ([Help Net Security](https://www.helpnetsecurity.com/2026/09/07/openai-research-automation-intern/), snippet, self-reported). Automated-researcher target: March 2028.
- Industry concentration: >90% of notable models from industry in 2025 ([AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report), snippet). ~90% of lab R&D compute goes to experiments ([Epoch](https://epoch.ai/data-insights/openai-compute-spend), snippet).
- Small-compute human discoveries adopted at frontier scale within about a year: Muon, 8×H100 → Kimi K2 1T in ~9 months ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), [Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) READMEs).

### Inferences
**(a) Share of paradigm-level breakthroughs discoverable at small compute (my estimate from the Q1 table; ±15 points given list selection).**
- Originated in ≤1e20-FLOP experiments: **~65%** (10/15).
- Recognizable as paradigm-level at small compute: **~33%** (5/15: AlexNet, attention, Adam, ResNet, diffusion).
- Required frontier compute or a frontier substrate to discover: **~33%** (5/15: ICL, Chinchilla, CoT, AlphaZero, o1/R1).

**(b) Trend.** Paradigm-level small-discoverability, in the sense of originating in small experiments, went from 4/4 (2012–16) to 6/8 (2017–20, but 5 of these were S→L with long lags) to 0/3 established items (2021–26; ~40–50% if unproven small-compute contenders count). Component-level small-discoverability is flat at ~50–65%, but the compute-intensive half doubles its requirements yearly (Barnett). The *locus* of discovery moved from academia (5/8 of 2012–16 items) to big labs. Small groups are persistent but mainly contribute components.

**(c) Calibrated probability for who discovers a compact low-compute general learner (conditional on it being found by ~2040).** Definitions:
- (i) Human-led work with ≲1e21 FLOP of research compute: academic, independent or startup, possibly AI-assisted.
- (ii) Human-directed work at large labs using large compute; AI tools as assistants.
- (iii) AI research systems (running on ≥1e26-FLOP-class infrastructure) generate the key idea and/or run the search, with humans mainly supervising and verifying.

Time-conditional judgment (all mine):

| If found… | P(time bucket) | (i) small-compute human | (ii) large-compute human-led | (iii) AI research systems |
|---|---|---|---|---|
| before 2029 | ~0.20 | 35% | 45% | 20% |
| 2029–2035 | ~0.45 | 15% | 30% | 55% |
| 2035–2040 | ~0.35 | 10% | 20% | 70% |
| **Weighted** | 1.0 | **~17%** | **~30%** | **~53%** |

Why (iii) is not higher than ~50–55%:
- The base rate of AI-originated paradigm-level ML innovations to date is zero.
- Machine-discovered components have been constant-factor gains.
- 2025 agents failed to reproduce known innovations.
- Attribution will be blurry: most 2029+ discoveries will be "human-directed, AI-executed", and some would be scored (ii).

Why (iii) is not lower:
- The compact-learner problem has exactly the structure where machine search has succeeded: a cheap automatic evaluator (train small, score generality) and a codifiable search space (DiscoRL, Lion, Primer, AlphaEvolve).
- AI research labor already exceeds human labor at one frontier lab by 3:1 in workdays (self-reported).
- If the learner were easy for humans, the decade of low-compute work (Gundlach: slow small-model progress) should already have found more of it. That shifts probability toward later, AI-heavy discovery.

Why (i) keeps ~15–20%:
- For this target, the Gundlach objection is weak: value is measured at low compute.
- Small groups have repeatedly found adopted components with ≤8 GPUs (RoPE, FlashAttention, DPO, Muon, TRM-type results), and often become AI-amplified individuals themselves.
- Verification of *superintelligence* at a 1e23–1e24 budget still needs mid-scale compute, which caps (i) unless compute is borrowed or donated.

Why (ii) is ~30%:
- Big labs own frontier substrates (for borrowed-compute discovery, as with CoT and R1).
- They own ~10× experimental compute (Epoch).
- They have fast adoption.
- The σ≈0 frontier-experiments result supports them *if* the key experiments turn out to need near-frontier scale.

**Implications for the report.**
- The earlier "~60%, AI systems on ≥1e26 FLOP" figure is defensible as an upper-middle estimate. A better-calibrated statement is **"~50% (35–65%) AI research systems; ~30% large-lab human-led; ~15–20% small-compute humans"**.
- The historical base rate by itself (zero AI-discovered paradigm shifts; about one third of paradigm shifts small-discoverable) would put (iii) far lower. The forward-looking evidence (research-labor automation curves, the fit between the task and automated search) is what pulls it up. The report should say so openly.
- **Ledger note:** under (iii), discovery compute is large (AI-researcher inference plus many small training runs), but the report's instance ledger excludes it. Under (i), the discovery ledger is small, but a borrowed frontier model may have been used for ideation or data. Under the report's accounting that is "borrowed compute" if it enters the artifact (e.g., distillation), and not if it only produced the idea.

### Gaps
- Selection bias: my 35-item list was hand-picked from famous, adopted innovations. A systematic sample (e.g., all components of Llama 3 / DeepSeek-V3 / Qwen3 / Kimi K2, plus post-training) would give tighter base rates. Barnett's per-innovation table would be the natural starting point but was inaccessible.
- The time-bucket probabilities and discoverer splits are my judgment, not derived from a model. The largest uncertainty is how attribution will be assigned in human-AI hybrid research after 2028.
- I found no empirical study of whether *AI-discovered* innovations are more or less scale-dependent than human ones. That would directly test whether AI search at small scale can find innovations that matter at scale.
