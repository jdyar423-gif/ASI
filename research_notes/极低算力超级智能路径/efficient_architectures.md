# Compute-Efficient Architectures and Training Methods (Mainstream Deep Learning): How Far Can Training + Inference Compute for Capable General Models Be Pushed Down? (as of Sept 2026)

Scope note for the report writer: these notes cover capability-per-FLOP multipliers from data, sparsity, sequence architecture, precision, optimizers, small models/distillation, test-time compute and retrieval/memory, plus how they stack. Almost all headline numbers are **vendor/author self-reported**. Where I could find independent checks, they're flagged. Verification limits: arXiv, Hugging Face, Epoch and LessWrong pages were blocked for direct fetch in this session, so many figures come from search-engine extracts of those pages or from well-known paper abstracts (cited by their arXiv URL). GitHub READMEs were fetched directly. The search budget ran out before a few 2026 items could be cross-checked; those items are listed under Gaps.

Frequently used conversions: 8-month halving ≈ 2.8×/yr; 60×/yr ≈ 1.78 OOM/yr ≈ 2-month halving; training FLOPs ≈ 6 × active params × tokens.

---

## Q1. Data quality & curriculum (Phi, FineWeb-Edu, DCLM, pruning, synthetic data), and hidden upstream compute

### Takeaway
Better data is one of the largest and best-documented multipliers. DCLM showed roughly 6.6× less training compute for near-equal MMLU, and Phi-1/Phi-4 showed small models matching or beating much larger ones on targeted skills. The strongest results (Phi-4) depend on synthetic data from a frontier teacher (GPT-4o), which brings in upstream compute that nobody has disclosed. Filtering gains are also bounded, because aggressive filtering pushes training into a repetition-limited regime.

### Cited Findings
- **Phi-1 (June 2023):** 1.3B params, trained for about 4 days on 8 A100s. Data was about 6B tokens of "textbook-quality" filtered web data plus about 1B tokens of GPT-3.5-generated synthetic textbooks and exercises. It scored 50.6% pass@1 on HumanEval, which at the time matched models more than 10× larger trained on 100× more data. — [Gunasekar et al., "Textbooks Are All You Need", arXiv 2306.11644](https://arxiv.org/abs/2306.11644)
- **Phi-4 (Dec 2024):** a 14B model built on a data-quality-centred recipe, using about 400B unique synthetic tokens across 50 dataset types. GPT-4o rewrote web, code, paper and book snippets into exercises, Q&A and structured reasoning. Phi-4 beats its teacher GPT-4o on GPQA (56.1 vs 50.6) and MATH (80.4 vs 74.6). The authors argue this shows the method goes "beyond distillation". — [Phi-4 Technical Report (Microsoft Research)](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/P4TechReport.pdf); [DeepLearning.AI summary](https://www.deeplearning.ai/the-batch/microsofts-phi-4-blends-synthetic-and-organic-data-to-surpass-larger-models-in-math-and-reasoning-benchmarks); [alphaXiv 2412.08905](https://www.alphaxiv.org/abs/2412.08905)
  - Phi-4 training scale (from the model card as I recall it; not re-fetched this session): about 9.8T tokens on 1,920 H100s for 21 days, which is about 0.97M H100-hours and about 8×10²³ FLOPs by 6ND. The GPT-4o inference compute used to generate the 400B synthetic tokens is **not disclosed**. — [microsoft/phi-4 model card](https://huggingface.co/microsoft/phi-4)
- **DCLM (June 2024):** a 7B model on the filtered 2.6T-token DCLM-Baseline corpus reaches **64% MMLU 5-shot**. That is close to Mistral-7B-v0.3 (63%) and Llama 3 8B (66%), which used up to **6.6× more compute**. The ratio follows from 6ND: 8B×15T vs 7B×2.6T. The paper also reports +6.6 MMLU points over MAP-Neo with 40% less compute. — [DataComp-LM, arXiv 2406.11794](https://arxiv.org/abs/2406.11794); [EmergentMind summary](https://www.emergentmind.com/topics/datacomp-lm-benchmark)
  - Conflict flag: one aggregator states "3,700 H100 GPU-hours vs 6,900 for Llama 3 8B". That can't be right, because Meta reported roughly 1.3M GPU-hours for Llama 3 8B. Use only the 6.6× ratio. — [EmergentMind DCLM page](https://www.emergentmind.com/topics/datacomp-lm-benchmark)
- **FineWeb-Edu vs DCLM (2024):** FineWeb-Edu scores higher on knowledge/education benchmarks (MMLU, ARC, OpenBookQA). DCLM-Baseline scores higher on HellaSwag and CommonsenseQA. Contamination rates are similar. — [DataComp-LM, arXiv 2406.11794](https://arxiv.org/abs/2406.11794)
- **Data pruning theory (2022):** with a high-quality pruning metric, the error-vs-dataset-size relationship can in principle move from power-law to exponential scaling. This was shown mainly on vision/ImageNet, and good metrics are the bottleneck. — [Sorscher et al., "Beyond neural scaling laws", arXiv 2206.14486](https://arxiv.org/abs/2206.14486)
- **Limit on filtering (data-constrained scaling, 2023):** at fixed compute, up to about 4 epochs of repeated data costs almost nothing compared with unique data. With more repetition, the value of extra compute decays toward zero. Filtering the web down to a small high-quality subset therefore eventually runs into repetition limits. — [Muennighoff et al., "Scaling Data-Constrained Language Models", arXiv 2305.16264](https://arxiv.org/abs/2305.16264)
- **Hidden upstream compute:** analyses of "catch-up" algorithmic progress explicitly drop distilled models from compute-efficiency fits. The reason is that capability-vs-final-run-compute is distorted by "additional compute sources, such as from distillation or substantial quantities of synthetic data generation". The same analysis argues that for catch-up progress, using earlier models via synthetic data or logit distillation is legitimate. — [MIRI TGT / LessWrong, "Catch-Up Algorithmic Progress Might Actually be 60× per Year"](https://www.lesswrong.com/posts/yXLqrpfFwBW5knpgc/catch-up-algorithmic-progress-might-actually-be-60-per-year)

### Inferences
- A reasonable compute multiplier from filtering and quality alone is **about 2–7×** at fixed benchmark performance. The low end comes from the MAP-Neo comparison; the high end is DCLM vs Llama 3 8B. The Llama 3 comparison overstates it, because Llama 3 8B was intentionally overtrained far past Chinchilla-optimal for inference reasons.
- Synthetic-data recipes (Phi) give larger *apparent* gains on reasoning/STEM benchmarks, but only by amortizing a frontier teacher's pretraining and inference. For an "entire pipeline low compute" question they aren't free: the teacher has to exist. They're cheap only in the catch-up sense, once a frontier model has been trained, and that cost can be spread over many students.
- Benchmark-targeted synthetic data (Phi-4 on GPQA/MATH) may raise benchmark scores more than general capability. Phi-series models have historically been criticised for weaker performance off-benchmark. I didn't re-verify this in this session, so treat it as a caution rather than a finding.

### Gaps
- The widely quoted figure that FineWeb-Edu matches baseline MMLU "with about 10× fewer tokens" could not be verified this session (HF blog/arXiv blocked, search budget exhausted).
- There is no public accounting of the teacher inference FLOPs used for Phi-4's synthetic corpus, or for any major synthetic-data pipeline.
- I found no independent (non-Microsoft) replication of Phi-4-level STEM results from a comparably small synthetic-data recipe that doesn't use a frontier teacher.

---

## Q2. Sparsity & conditional computation (MoE): demonstrated compute-equivalent multipliers

### Takeaway
Fine-grained MoE is now the default frontier recipe and is the most robust single architectural multiplier. Controlled scaling-law work finds **more than 7× compute leverage** over dense models at about 3% activation. Production reports put the combined MoE+hybrid gain at roughly 10× training GPU-hours for equal quality (Qwen3-Next). DeepSeek-V3 trained a GPT-4-class-or-better model for about 2.8M H800-hours. The catch is that total parameters (memory) grow even as active FLOPs fall.

### Cited Findings
- **DeepSeek-V3 (Dec 2024):** 671B total / 37B active, 14.8T tokens, **2.788M H800 GPU-hours** total, or about $5.576M at $2/GPU-hour. This covers the final run only and excludes prior research and ablations. It used FP8 mixed-precision training and multi-token prediction. — [DeepSeek-V3 Technical Report, arXiv 2412.19437](https://arxiv.org/abs/2412.19437); cost reconfirmed in [DeepSeek V4 guide (secondary)](https://www.morphllm.com/deepseek-v4)
- **DeepSeek-V4 (2026, secondary sources):** V4-Pro is reportedly **1.6T total / 49B active** trained on more than 33T tokens. V4-Flash is 284B total / 13B active. **DeepSeek has not disclosed V4 training compute or cost.** — [MorphLLM DeepSeek V4 page](https://www.morphllm.com/deepseek-v4); [Kili Technology V4 guide](https://kili-technology.com/blog/data-story-deepseek-v4). Treat these as unverified: the sources are secondary, and one of them (deepseek.ai) isn't DeepSeek's official domain.
- **Kimi K2 (July 2025):** 1.04T total / 32B active, 15.5T tokens, trained with the MuonClip optimizer with "zero loss spikes" (vendor-reported). — [Kimi K2 report, arXiv 2507.20534](https://arxiv.org/abs/2507.20534)
- **Qwen3 (Apr–May 2025):** 36T pretraining tokens. The flagship is Qwen3-235B-A22B. Vendor claims: Qwen3-30B-A3B beats QwQ-32B with about 10× fewer activated parameters, and Qwen3-4B "can rival" Qwen2.5-72B-Instruct. — [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/abs/2505.09388); [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- **Qwen3-Next-80B-A3B (Sept 2025):** 80B total, about 3B active, with a hybrid of Gated DeltaNet and gated attention at 3:1. Qwen reports the base model matches or slightly beats dense Qwen3-32B while using **less than 10% of its training cost (GPU-hours)**. It also gets **more than 10× inference throughput** at contexts beyond 32K. Vendor-reported. — [Qwen blog: Qwen3-Next](https://qwen.ai/blog?from=research.latest-advancements-list&id=4074cca80393150c248e508aa62983f9cb7d27cd); [HF model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct); [secondary summary](https://colinmcnamara.com/blog/qwen3-next-ultimate-training-inference-efficiency-guide)
- **MoE "Efficiency Leverage" scaling law (Ant/Ling, July 2025; ICLR 2026):** EL is the dense-equivalent compute multiplier. It is driven mainly by the activation ratio and the total compute budget, both following power laws. Expert granularity has an optimal range. An MoE with 3.1% activation and granularity 12 is predicted to reach **more than 7× EL at 1e22 FLOPs**. It was validated by Ling-mini-beta (0.85B active), which matched a 6.1B dense model on the same 1T tokens with **more than 7× less compute**. — [Towards Greater Leverage, arXiv 2507.17702](https://arxiv.org/abs/2507.17702); [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/32b640528f5b67975562210f00c131ed-Paper-Conference.pdf)
- **Fine-grained MoE scaling laws (Feb 2024):** MoE consistently beats dense Transformers, and the efficiency gap *widens* as model size and training budget grow. Granularity is a key hyperparameter. — [Krajewski et al., arXiv 2402.07871](https://arxiv.org/abs/2402.07871)
- **Memory side of MoE (Feb 2025):** joint scaling laws suggest MoE can be memory-efficient too, not only compute-efficient, when configured properly. — [Joint MoE Scaling Laws, arXiv 2502.05172](https://arxiv.org/abs/2502.05172)
- **Nemotron 3 Nano (Dec 2025):** hybrid Mamba-2 + GQA + MoE with **31.6B total / 3.2B active**. It matches or beats GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507 on AIME25, SWE-Bench, LCBv6, RULER@1M and other benchmarks. It delivers 2.2× and 3.3× throughput respectively at 8K input / 16K output. Vendor-reported. — [Nemotron 3 Nano, arXiv 2512.20848](https://arxiv.org/abs/2512.20848)

### Inferences
- Dense-equivalent **training-compute multiplier from MoE: about 3–10×** at today's scales. It rises with budget and sparsity, from EL > 7× at 1e22 FLOPs up to about 10× combined with hybrid attention in Qwen3-Next.
- **Inference FLOPs** scale with active params, so a 3B-active model runs at roughly 3B-dense FLOP cost. Memory footprint and bandwidth scale with *total* params, however. For edge or low-power inference, memory rather than FLOPs becomes the binding constraint, which is where Engram-style offloading (Q7) or low-bit weights (Q4) come in.
- Rough compute comparison. DeepSeek-V3 is about 6×37B×14.8T ≈ 3.3×10²⁴ FLOPs. Epoch's commonly cited estimate for original GPT-4 is about 2×10²⁵ FLOPs (not re-verified this session). That suggests **about 6× less training compute for a stronger model**, roughly 20 months later, before counting FP8's roughly 2× cheaper FLOPs. It is consistent with the approximately 3×/yr frontier algorithmic-progress trend (Q8).

### Gaps
- There is no independent (non-vendor) replication of the Qwen3-Next "<10% GPU-hours" claim or of the Ling ">7× EL" claim at frontier scale.
- DeepSeek-V4, Kimi K3 and other 2026 frontier MoE training compute figures are not publicly disclosed or verified.
- I found no clean study of MoE leverage at compute budgets above about 1e24 FLOPs with a matched dense baseline, because nobody trains the dense control.

---

## Q3. Alternative sequence architectures (SSMs/Mamba, linear attention/RWKV-7, hybrids, diffusion LMs)

### Takeaway
Pure SSM or linear-RNN models haven't displaced attention: they lag on recall, copying and in-context learning. By 2025–2026, though, **hybrids (about 3:1 linear-to-full attention) have become mainstream** at Qwen (Qwen3-Next, Qwen3.5), Moonshot (Kimi Linear), and NVIDIA (Nemotron-H, Nemotron 3). They match or slightly beat full attention on quality, with **3–10× inference throughput and about 75% smaller KV cache at long context**. Training-FLOP savings are modest. Diffusion LMs give 5–10× decoding speed from parallelism but still trail on hard reasoning.

### Cited Findings
- **Mamba (Dec 2023):** selective SSM with about **5× higher inference throughput** than Transformers and linear scaling in sequence length. Mamba-3B beats same-size Transformers and matches Transformers twice its size in pretraining and downstream evaluation. — [Gu & Dao, arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
- **Mamba-2 / SSD (May 2024):** the core layer is **2–8× faster** than Mamba's selective scan while staying competitive with Transformers. — [Dao & Gu, arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
- **Controlled 8B study (NVIDIA, June 2024):** pure Mamba/Mamba-2 lag Transformers on tasks that need strong copying or in-context learning, such as 5-shot MMLU and phonebook lookup. An 8B hybrid of about 43% Mamba-2, 7% attention and 50% MLP layers **beats the Transformer on all 12 standard tasks (+2.65 points on average)** and is predicted to be **up to 8× faster** at generation. — [Waleffe et al., arXiv 2406.07887](https://arxiv.org/abs/2406.07887); see also [Jelassi et al., "Repeat After Me: Transformers are Better than SSMs at Copying", arXiv 2402.01032](https://arxiv.org/abs/2402.01032)
- **Nemotron-H (Apr 2025):** hybrid Mamba-Transformer family (8B/56B) with accuracy on par with or better than similarly sized Transformers (Qwen-2.5, Llama-3.1). It runs **up to 3× faster at inference**. The 56B model was trained in FP8. — [Nemotron-H, arXiv 2504.03624](https://arxiv.org/abs/2504.03624)
- **Jamba (Mar 2024):** Transformer-Mamba-MoE hybrid with 52B total / 12B active, 256K context, and fits on a single 80GB GPU. — [Jamba, arXiv 2403.19887](https://arxiv.org/abs/2403.19887)
- **RWKV-7 "Goose" (Mar 2025):** a 2.9B RNN with constant memory and constant per-token time. It sets a new 3B state of the art on multilingual tasks and matches the 3B state of the art on English. It was trained on 5.6T tokens, versus 18T for Qwen2.5-3B. Note the inconsistency: its World v3 corpus is described as 3.1T tokens, so the 5.6T figure presumably counts cumulative or continued training. — [RWKV-7, arXiv 2503.14456](https://arxiv.org/abs/2503.14456)
- **Kimi Linear (Oct 2025):** Kimi Delta Attention (a refined Gated DeltaNet) plus MLA at 3:1. The model is 48B total / 3B active and was trained on 5.7T tokens, with fair comparisons at 1.4T tokens. It **beats full MLA attention** in fair comparisons across short-context, long-context and RL settings. It cuts **KV cache by up to 75%**, gives **up to 6× decoding throughput at 1M context**, and a 3.98× speedup at 128K on RULER (84.3). Vendor-reported. — [Kimi-Linear GitHub](https://github.com/MoonshotAI/Kimi-Linear); [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)
- **Qwen3.5 (Feb 16 – Mar 2, 2026):** the whole family, from the 397B-A17B flagship down to the 0.8B/2B/4B/9B small models, uses a "Gated Delta Networks combined with sparse MoE" design. Secondary sources describe a 3:1 DeltaNet:attention ratio. — [QwenLM/Qwen3.5 GitHub](https://github.com/QwenLM/Qwen3.5); [Medium explainer](https://medium.com/@milesk_33/qwen-3-5-small-models-0-8b-9b-the-architecture-shift-that-lets-edge-ai-finally-compete-ab3b807273ff)
- **Diffusion LMs:**
  - LLaDA 8B (Feb 2025) was trained from scratch on 2.3T tokens and is competitive with LLaMA3-8B on in-context learning. — [Nie et al., arXiv 2502.09992](https://arxiv.org/abs/2502.09992)
  - Inception's Mercury (Feb 2025) was the first commercial-scale diffusion LLM. Mercury Coder runs at about 1,100 tok/s (Mini) and about 740 tok/s (Small) on an H100. — [Inception blog](https://www.inceptionlabs.ai/blog/introducing-mercury)
  - Mercury 2 is reported at 1,009 tok/s on one Blackwell GPU, and Mercury 2.5 at about 1,107 tok/s. — [secondary: Medium](https://buzzgrewal.medium.com/beyond-next-token-how-diffusion-llms-like-mercury-2-and-llada-hit-1-000-tokens-per-second-in-2026-996b52cd4fce), [tech-insider](https://tech-insider.org/mercury-2-5-vs-gemini-diffusion-llm-speed-2026/). A claimed "91.1 AIME 2025" for Mercury 2 appears only in a Medium post and is unverified.
  - Gemini Diffusion (May 2025) runs at about 1,479 tok/s, roughly 5× Gemini 2.0 Flash-Lite, and roughly matches it on code (HumanEval 89.6 vs 90.2; MBPP 76.0 vs 75.8). — [secondary summary](https://tech-insider.org/mercury-2-5-vs-gemini-diffusion-llm-speed-2026/)
  - **DiffusionGemma (Aug 2026)** was obtained by fine-tuning the Gemma 4 MoE (25.2B total / 3.8B active) with **less than 10% of the AR model's training token budget**. It uses SFT for bidirectional denoising, then RL plus sampler distillation. It generates about 20 tokens per forward pass and reaches **about 1,500 tok/s on one H100**. — [DiffusionGemma Technical Report, arXiv 2608.00146](https://arxiv.org/abs/2608.00146)
  - ParallelBench (Oct 2025) documents quality loss from parallel decoding when output tokens are interdependent. — [arXiv 2510.04767](https://arxiv.org/abs/2510.04767)

### Inferences
- Hybrids are now a mainstream, production-validated **inference multiplier of about 3–10×**, larger at long context and in agentic or RL rollouts with long traces. The quality effect is neutral to slightly positive. Their *training*-compute effect is small on its own, perhaps 1–1.5× from slightly better loss per FLOP; most of Qwen3-Next's 10× training saving is plausibly MoE.
- Linear or constant-state models matter a great deal for reasoning models, because test-time compute scales sequence length. The KV-cache and quadratic costs of full attention are exactly what long chains of thought blow up. This couples Q3 with Q6.
- Diffusion LMs mostly give **latency and parallelism** gains, not total-FLOP reductions. Each forward pass is bidirectional over a block and denoising takes several steps. They can be converted cheaply from AR models (DiffusionGemma used less than 10% of the tokens), so they add to AR pretraining rather than replacing it. Their reasoning quality still trails AR models.

### Gaps
- Independent third-party evaluations of hybrid vs full-attention quality at frontier scale are scarce; nearly all comparisons are vendor-run.
- The Gemini Diffusion model page reportedly shows it lagging Flash-Lite considerably on GPQA Diamond and Global MMLU. I couldn't re-verify the exact numbers this session.
- Search results reference "Kimi K3" reportedly using KDA linear attention for 1M context at frontier scale, but I couldn't verify its size, compute or results. The Kimi-Linear README doesn't mention K3.
- I found no reliable FLOPs-per-quality comparison of diffusion vs AR LMs at matched training compute beyond about 8B.

---

## Q4. Low precision & hardware-aware training (FP8, FP4, BitNet ternary), energy per token

### Takeaway
FP8 training is production-proven at frontier scale (DeepSeek-V3, Nemotron-H), and 4-bit NVFP4 pretraining matched FP8 on a 12B / 10T-token run in 2025–26. Together these give roughly 2–4× cheaper FLOPs per unit of hardware. Ternary BitNet b1.58 gives large **inference** memory and energy savings: about 6–23× lower estimated energy per token and 3.5–12× smaller memory at 2B scale, at slight quality cost. It doesn't cut training compute on current hardware. Scaling-law work also warns that low precision and heavy overtraining interact badly.

### Cited Findings
- **DeepSeek-V3 FP8 (Dec 2024):** first validation of FP8 mixed-precision training at extremely large scale. Relative loss error versus BF16 stayed below 0.25%. — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- **NVFP4 pretraining (NVIDIA, Sept 2025; v2 Mar 2026):** a 12B hybrid Mamba-Transformer trained on **10T tokens in 4-bit NVFP4**, the longest public 4-bit run. It reached **62.58% MMLU-Pro 5-shot vs 62.62% for the FP8 baseline**. It used random Hadamard transforms, 2D block scaling, stochastic rounding and a few high-precision layers. MXFP4 needed 36% more tokens (1.36T vs 1T) to match NVFP4 loss. — [Pretraining LLMs with NVFP4, arXiv 2509.25149](https://arxiv.org/abs/2509.25149); [MarkTechPost, May 2026](https://www.marktechpost.com/2026/05/18/nvidia-introduces-a-4-bit-pretraining-methodology-using-nvfp4-validated-on-a-12b-hybrid-mamba-transformer-at-10t-token-horizon/)
- **Scaling laws for precision (Nov 2024):** training in lower precision reduces a model's "effective parameter count". Post-training quantization degradation *increases* with more pretraining data, so heavily overtrained models quantize worse. The compute-optimal pretraining precision is predicted at about 7–8 bits. — [Kumar et al., arXiv 2411.04330](https://arxiv.org/abs/2411.04330)
- **BitNet b1.58 (Feb 2024):** ternary {−1, 0, +1} weights. A 3B model matches FP16 LLaMA-3B perplexity and end-task scores while being 2.71× faster with 3.55× less GPU memory, and gains grow with scale. — [Ma et al., arXiv 2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet b1.58 2B4T (Apr 2025):** about 2B parameters (2.4B per the GitHub README) trained on 4T tokens.
  - Non-embedding memory is **0.4 GB vs 1.4–4.8 GB** for comparable models, and estimated decoding energy is **0.028 J/token vs 0.186–0.649 J**.
  - It averages 54.19 across 16 benchmarks versus 55.23 for Qwen2.5-1.5B, which leads on MMLU and code. It wins on ARC-C, GSM8K and WinoGrande, and beats INT4-quantized Qwen2.5-1.5B.
  - Vendor-reported (Microsoft). — [BitNet b1.58 2B4T Technical Report, arXiv 2504.12285](https://arxiv.org/abs/2504.12285); [DAIR.AI summary](https://academy.dair.ai/papers/bitnet-b1-58-2b4t)
- **bitnet.cpp (Microsoft GitHub, updated July 2026):**
  - Speedups over full-precision inference are 1.37–5.07× on ARM and 2.37–6.17× on x86.
  - Energy reductions are 55.4–70.0% on ARM and 71.9–82.2% on x86.
  - A 100B BitNet b1.58 model runs on a single CPU at 5–7 tok/s, about human reading speed.
  - BitNet embedding models (0.6B, 270M) were released July 20, 2026. — [microsoft/BitNet GitHub](https://github.com/microsoft/BitNet)

### Inferences
- **Training:** FP8→FP4 roughly doubles, then doubles again, peak tensor throughput per chip generation. H100 FP8 dense peak is about 2× BF16, and Blackwell adds FP4. That lowers **cost and energy per FLOP**, not FLOPs per capability. Plausible realized training cost multiplier versus BF16 is **about 1.5–3×**.
- **Inference:** ternary or 2-bit weights give about **5–20× energy and memory savings** at small scale. That is the strongest "cheap inference" lever for edge deployment. BitNet training still keeps latent high-precision weights, so the training cost is similar to a normal model with the same parameter count and tokens.
- Tension with the small-model route: Kumar et al. imply that the "overtrain a small model on huge token counts" recipe (Q5) makes low-bit quantization *harder*. The two multipliers aren't fully multiplicative; native low-bit training, as in BitNet, is the way around this.
- For the report: the energy-per-token figures are *estimated matmul energy* from arithmetic-operation models, not wall-plug measurements.

### Gaps
- There is no independent large-scale (>10B) ternary model trained from scratch that matches the full-precision state of the art; BitNet's largest native release is about 2B.
- I found no reliable measured (wall-plug) energy-per-token figures for frontier models this session.
- There is no public evidence yet of FP4 pretraining at frontier (>100B active) scale.

---

## Q5. Small language models (1–10B) in 2025–2026 vs earlier frontier; lag; distillation's hidden upstream compute

### Takeaway
Frontier capability reaches models that run on a single consumer GPU in about **6–12 months** (Epoch). "Capability density" per parameter doubles about every **3.5 months** (densing law). Catch-up algorithmic progress, which includes distillation and synthetic data, has been estimated at **16–60× per year**, far faster than the approximately 3×/yr frontier rate. Much of this speed comes from distilling or learning from frontier models, so it depends on continued expensive frontier training.

### Cited Findings
- **Epoch (Aug 2025):** a single top gaming GPU (RTX 5090, under $2.5K) can run models that match the absolute frontier of 6–12 months earlier, with about a 9-month lag. That covers ≤28B params on an RTX 4090 or ≤40B on an RTX 5090. Epoch attributes this to open models scaling at frontier rates, distillation, and better GPUs. Epoch predicted home-runnable models matching Grok 4 by Q2 2026. — [Epoch data insight: consumer GPU gap](https://epoch.ai/data-insights/consumer-gpu-model-gap); [Epoch on X](https://x.com/EpochAIResearch/status/1956468513817915598)
- **Epoch (2026):** since January 2026, the best open-weight models have trailed closed frontier models by about **4 months** (about 8 ECI points). — [Epoch: open-closed ECI gap](https://epoch.ai/data-insights/open-closed-eci-gap)
- **Densing law (arXiv Dec 2024; Nature Machine Intelligence 2025):** across 51 open base models, maximum capability density (capability per parameter) doubles about every **3.5 months**, so an equal-quality model needs half the parameters about 3.5 months later. — [Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01137-0); [arXiv 2412.04315](https://arxiv.org/abs/2412.04315)
- **Catch-up algorithmic progress (2025):** estimated at **16×–60× per year** for 2023–2025, including post-training. The headline is 1.76 OOM/yr, or about 60×, which is about a 2-month halving time for compute to reach a fixed capability. Distilled models are excluded from the main fit because of hidden upstream compute. — [MIRI TGT blog](https://techgov.intelligence.org/blog/catch-up-algorithmic-progress-might-actually-be-60x-per-year); [LessWrong](https://www.lesswrong.com/posts/yXLqrpfFwBW5knpgc/catch-up-algorithmic-progress-might-actually-be-60-per-year)
- **Qwen3 (Apr 2025, vendor):** Qwen3-4B "can rival" Qwen2.5-72B-Instruct, and Qwen3-30B-A3B beats QwQ-32B. — [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- **Qwen3.5 small models (Mar 2, 2026):** 0.8B, 2B, 4B and 9B, natively multimodal, 262K context, Apache 2.0, using a Gated DeltaNet hybrid.
  - Vendor claims the 9B is "very close or even outperforms models 10× its size".
  - On MMMU-Pro, the 9B scores 69.2 and the 4B 65.4, versus 56.6 for Qwen3-VL-8B.
  - Artificial Analysis Intelligence Index is about 32 (9B) and 27 (4B), roughly double other sub-10B models. — [Qwen3.5 GitHub](https://github.com/QwenLM/Qwen3.5); [Artificial Analysis article](https://artificialanalysis.ai/articles/qwen3-5-small-models) (via search extract); [Medium](https://medium.com/@milesk_33/qwen-3-5-small-models-0-8b-9b-the-architecture-shift-that-lets-edge-ai-finally-compete-ab3b807273ff)
  - One secondary blog claims Qwen3.5-9B scores **81.7% on GPQA Diamond** and 82.5% on MMLU-Pro. **This is unverified.** If true, it would put a 9B model above o1 (Dec 2024) on GPQA. — [Labellerr blog](https://www.labellerr.com/blog/best-small-language-models-under-10b-parameters/)
- **Gemma 4 (Apr 2026):** Effective-2B, Effective-4B (using per-layer embeddings), a 26B MoE and a 31B dense model (secondary source). DiffusionGemma confirms that the Gemma 4 MoE is 25.2B total / 3.8B active. — [Labellerr blog](https://www.labellerr.com/blog/best-small-language-models-under-10b-parameters/); [arXiv 2608.00146](https://arxiv.org/abs/2608.00146)
- **Distillation beats RL for small models (DeepSeek-R1, Jan 2025):**
  - DeepSeek-R1-Distill-Qwen-7B scores 55.5% on AIME 2024 and R1-Distill-Qwen-32B scores 72.6%. The latter compares with about 47% for Qwen-32B trained directly with large-scale RL (R1-Zero-style).
  - The authors conclude that distilling stronger models into small ones works very well, whereas small models relying on large-scale RL "require enormous computational power and may not even achieve the performance of distillation". — [DeepSeek-R1, arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- **Phi-4-mini-reasoning (Apr 2025):** a 3.8B math reasoner trained with a multi-stage distillation plus RL recipe that beats roughly 2× larger R1-distilled reasoning models on math benchmarks. — [arXiv 2504.21233](https://arxiv.org/abs/2504.21233)

### Inferences
- Lag of small models behind frontier: about 6–12 months for models at or under 40B that fit a consumer GPU, and plausibly **about 12–24 months for 4–9B models**. That is roughly **1.5–2 OOM fewer parameters** for the same capability within about 1–2 years, going by densing law and the Qwen3-4B vs Qwen2.5-72B pairing.
- The *per-model* training compute for small models isn't small, because they're heavily overtrained (Qwen3 used 36T tokens). A 4B model on 36T tokens is about 6×4e9×3.6e13 ≈ 8.6×10²³ FLOPs, the same order as Phi-4 and about 1/4 of DeepSeek-V3. The compute saving is overwhelmingly on the **inference** side.
- Distillation is the main engine of catch-up efficiency, and it's **not a substitute for frontier compute**; it amortizes it. For the question of reaching ASI with low compute across the whole pipeline, distillation and synthetic data move compute from the student to the teacher. It cuts the *marginal* cost of each additional capable model, but not the cost of first reaching a new capability level.

### Gaps
- I couldn't verify the Qwen3.5-4B/9B GPQA, AIME or LiveCodeBench scores from the official model card, because the HF model card was blocked and the GitHub README gives numbers only in images.
- Training token counts and compute for Qwen3.5 small models, Gemma 4 and SmolLM3 weren't verified this session.
- There is no public accounting of the teacher compute used to distil Qwen3.5 or Gemma 4 small models.

---

## Q6. Test-time compute vs training compute: exchange rate and where it breaks

### Takeaway
Under favourable conditions (medium-difficulty problems, a good verifier, low inference volume), test-time compute can substitute for about **1 OOM of training compute** at a cost of about 1–2 OOM more inference. Snell et al. found compute-optimal allocation more than 4× better than best-of-N, and small models plus search beating a 14× larger model at matched FLOPs. The exchange breaks down on the hardest problems, where test-time compute can't create missing capability. It also breaks down at high query volume, because inference cost recurs, and without reliable verifiers.

### Cited Findings
- **Snell et al. (Aug 2024; ICLR 2025):** compute-optimal test-time scaling, which adapts per-prompt between verifier-guided search with process reward models and sequential revision, is **more than 4× more efficient than best-of-N**.
  - In a FLOPs-matched evaluation, a small model with extra test-time compute **beats a 14× larger model** on problems where the small model already has non-trivial success.
  - Effectiveness depends heavily on prompt difficulty. On the hardest problems, spending FLOPs on pretraining is better, and the trade depends on the ratio of inference tokens to pretraining tokens. — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314); [EmergentMind](https://www.emergentmind.com/papers/2408.03314)
- **Inference scaling laws (Wu et al., Aug 2024):** smaller models plus advanced inference such as tree search give Pareto-optimal cost/performance. For example, Llemma-7B with tree search consistently beats Llemma-34B on MATH at equal inference compute. — [arXiv 2408.00724](https://arxiv.org/abs/2408.00724)
- **Repeated sampling (Brown et al., July 2024):** coverage grows log-linearly with the number of samples. DeepSeek-Coder-V2 on SWE-bench Lite went from 15.9% (1 sample) to 56% (250 samples), above the single-attempt state of the art of 43%. Without an automatic verifier, however, selection methods such as majority vote and reward models plateau after a few hundred samples. — [Large Language Monkeys, arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- **Epoch (2023):** across five techniques (model scaling, MCTS, pruning, resampling, chain of thought), one can spend **1–2 OOM more inference compute to save about 1 OOM of training compute**. Resampling for code saves 1 OOM of training for about 1.5 OOM of inference. Conversely, overtraining spends about 2 OOM more training to save 1 OOM of inference. — [Epoch: Trading off compute in training and inference](https://epoch.ai/publications/trading-off-compute-in-training-and-inference)
- **Cheap post-training for reasoning:**
  - s1 (Jan 2025) fine-tuned Qwen2.5-32B on only **1,000 examples, 26 minutes on 16 H100s**. With "budget forcing" at test time it beat o1-preview on competition math (MATH, AIME24) by up to 27%. This shows the base model already held the capability; test-time compute and tiny SFT unlocked it. — [s1, arXiv 2501.19393](https://arxiv.org/abs/2501.19393)
  - The DeepSeek-R1 Nature paper (Sept 2025) reported R1's reasoning RL stage cost about **$294K** on 512 H800s, on top of the roughly $5.6M V3 base. — [Nature, DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z)

### Inferences
- The exchange rate is **sub-unity in total FLOPs**: 1 OOM of training saved costs 1–2 OOM of inference per query. For a whole pipeline with many queries, *inference dominates lifetime compute*, so moving compute to test time *raises* the total unless deployment volume is small. For an ASI serving enormous query volumes, test-time scaling is a capability lever, not a compute saver.
- Test-time compute can't substitute for missing knowledge or skills; it works when p(correct) is non-trivial. It amplifies a competent base model rather than creating one. So the cheap route is a strong but small base, plus RL-trained reasoning, plus efficient long-context architectures (Q3) to make long reasoning traces cheap.
- Verifiers are the bottleneck. Domains with cheap automated verification (math, code, formal proofs) get the best exchange rates; open-ended domains don't.

### Gaps
- I found no updated (2025–2026) quantitative exchange-rate study for RL-trained reasoning models (o-series/R1-style) comparable to Snell et al.'s PRM/revision setting.
- I have no reliable data on what share of frontier labs' 2026 compute goes to RL and test-time inference versus pretraining.

---

## Q7. Memory / retrieval: offloading knowledge to cheap storage ("knowledge vs reasoning" separation)

### Takeaway
There's consistent evidence that **factual knowledge can be moved out of dense FLOP-heavy weights into cheap lookup structures**. Examples are RETRO (about 25× fewer parameters for similar perplexity), Atlas (50× fewer parameters than PaLM on few-shot QA), Meta's memory layers (beating dense models with 2× compute), and DeepSeek's 2026 Engram (hash-lookup n-gram memory, a 100B-param table in host DRAM with under 3% throughput penalty). This is a real, largely orthogonal multiplier for knowledge-heavy tasks. It doesn't make reasoning itself cheaper, and some headline gains (RETRO) were partly inflated by train/test overlap.

### Cited Findings
- **RETRO (DeepMind, Dec 2021):** retrieval from a 2T-token database. A 7.5B RETRO performs comparably to GPT-3 (175B) and Jurassic-1 (178B) on the Pile despite **25× fewer parameters**. The paper itself analyzes how much of the gain comes from overlap between retrieved training data and evaluation data. — [Borgeaud et al., arXiv 2112.04426](https://arxiv.org/abs/2112.04426)
- **RETRO reproduction (NVIDIA, Apr 2023):** pretraining with retrieval improves perplexity and factual accuracy, but downstream gains outside knowledge-intensive tasks are modest. — [Wang et al., "Shall We Pretrain Autoregressive LMs with Retrieval?", arXiv 2304.06762](https://arxiv.org/abs/2304.06762)
- **Atlas (Aug 2022):** an 11B retrieval-augmented model reaches over 42% on NaturalQuestions with 64 examples. That beats 540B PaLM by 3 points with **50× fewer parameters**. — [Izacard et al., arXiv 2208.03299](https://arxiv.org/abs/2208.03299)
- **Memory Layers at Scale (Meta, Dec 2024):** trainable key-value memory layers scaled to about 128B memory parameters. The augmented models beat dense models with **more than 2× the compute budget**, and beat MoE matched on compute and parameters, especially on factual tasks. — [Berges et al., arXiv 2412.09764](https://arxiv.org/abs/2412.09764)
- **DeepSeek Engram (Jan 2026):** "conditional memory" via O(1) hashed n-gram embedding lookup, presented as a second sparsity axis alongside MoE.
  - Under strict iso-parameter and iso-FLOP constraints, Engram-27B beats MoE baselines on knowledge, reasoning, code and math. A U-shaped law governs the split of sparse capacity between MoE experts and memory.
  - Deterministic addressing allows offloading huge tables to host memory. Secondary reports say a 100B-parameter table was offloaded to DRAM with **under 3% throughput penalty**, benchmarks gained about 3–5 points, and needle-in-a-haystack rose from 84.2% to 97%. — [deepseek-ai/Engram GitHub](https://github.com/deepseek-ai/Engram); [arXiv 2601.07372](https://arxiv.org/abs/2601.07372); [Introl blog (secondary)](https://introl.com/blog/deepseek-engram-conditional-memory-architecture-january-2026); [SDxCentral](https://www.sdxcentral.com/news/deepseek-looks-to-offload-simple-llm-tasks-to-save-billions-of-parameters/)
  - A follow-up (July 2026) proposes a training-free, SSD-backed Engram variant. — [TF-Engram, arXiv 2607.07388](https://arxiv.org/abs/2607.07388)

### Inferences
- For knowledge-heavy capability, retrieval and memory give roughly **10–50× parameter savings** (RETRO, Atlas) and about **2× compute-equivalent** in end-to-end LM training (memory layers, Engram). The memory lives in DRAM, SSD or disk, which is orders of magnitude cheaper per byte and per joule than HBM plus FLOPs.
- This is the most plausible mechanism for a "small reasoning core plus large cheap memory" design. It is largely **orthogonal to MoE, data quality and low precision**, since it stacks as a separate sparsity axis in Engram's framing. Reasoning FLOPs still have to be paid.
- Caveat: RETRO's 25× is a perplexity result with partial leakage. Downstream reasoning gains from retrieval are much smaller than knowledge-recall gains.

### Gaps
- There is no independent replication of Engram's results. Its precise benchmark deltas and allocation ratios couldn't be read from the primary PDF because arXiv was blocked.
- I found no evidence of a frontier-class model whose *reasoning* ability is preserved while most of its knowledge is externalized. That separation is still unproven at scale.

---

## Q8. Optimizers, training tricks, and measured algorithmic progress (Muon, µP, modded-nanogpt speedrun, Epoch/Gundlach)

### Takeaway
Measured algorithmic progress in LMs is about **3×/yr at the frontier** (8-month halving). Small-scale speedruns show striking wall-clock gains: modded-nanogpt went from **45 min to 1.126 min on 8×H100** (about 40×, Aug 2026) and cut training tokens about 25×. Rigorous re-analysis shows the *scale-invariant* innovations (optimizers, normalization, activations and so on) add up to **under 10×**. Most historical gains came from scale-dependent changes (Transformer, Chinchilla rebalancing). Optimizer "2×" claims shrink to about 1.1–1.4× against well-tuned baselines as scale grows.

### Cited Findings
- **modded-nanogpt speedrun (Keller Jordan et al.):** target is ≤3.28 validation loss on FineWeb (GPT-2-small quality) on 8×H100.
  - The llm.c baseline was **45 min (May 28, 2024)**.
  - Milestones: 7.8 min (Nov 8, 2024); **2.992 min (Jan 16, 2025)**; 2.717 min (Sept 5, 2025); 2.128 min (Dec 18, 2025); 1.521 min (Feb 3, 2026).
  - **Record #91 is 1.126 min (Aug 6, 2026)**, about 40× faster overall.
  - Ingredients include rotary embeddings, QK-norm, ReLU², the Muon optimizer, FlashAttention-3, FP8 matmuls and custom kernels.
  - Per the README extract, training tokens fell from **10B to about 0.4B** (about 25×).
  - GPT-2-Medium track (target 2.92 loss): from 5.8 h to 17.35 min (about 20×).
  - — [KellerJordan/modded-nanogpt GitHub](https://github.com/KellerJordan/modded-nanogpt)
  - Caveats: record #91's change was "mask logits for infeasible token continuations *during validation*", which is an evaluation-side trick rather than a training improvement. The fetch tool's summary also gave a "~40–50% algorithmic vs ~50–60% systems" split that is **not reliable** (likely model-generated), so it shouldn't be cited.
- **Muon / Moonlight (Moonshot, Feb 2025):** Muon achieves about **2× computational efficiency** versus AdamW under compute-optimal training, matching AdamW with **about 52% of the training FLOPs**. Moonlight is 16B total / 3B active on 5.7T tokens and scores MMLU 70.0, versus 65.6 for Qwen2.5-3B, which was trained on 18T tokens. Vendor-reported. — [MoonshotAI/Moonlight GitHub](https://github.com/MoonshotAI/Moonlight); [arXiv 2502.16982](https://arxiv.org/abs/2502.16982)
- **Independent optimizer benchmark (Stanford, Sept 2025):** with carefully tuned baselines, speedups of new optimizers (Muon, SOAP and others) over AdamW are much smaller than claimed. They fall from about 1.4× at 0.1B to about 1.1× at 1.2B parameters, and many "2×" claims come from under-tuned baselines. — [Wen et al., "Fantastic Pretraining Optimizers and Where to Find Them", arXiv 2509.02196](https://arxiv.org/abs/2509.02196)
- **Kimi K2 (July 2025):** MuonClip was used at 1T-param / 15.5T-token scale with no loss spikes, so Muon-family optimizers are production-viable at frontier scale. — [arXiv 2507.20534](https://arxiv.org/abs/2507.20534)
- **µP / µTransfer (Mar 2022):** zero-shot hyperparameter transfer from a 40M proxy beat published GPT-3 6.7B results, at a tuning cost of about 7% of pretraining compute. It removes the costly full-scale hyperparameter sweeps from the pipeline. — [Yang et al., Tensor Programs V, arXiv 2203.03466](https://arxiv.org/abs/2203.03466)
- **Chinchilla (Mar 2022):** 70B trained on 1.4T tokens beats 280B Gopher at the same compute. Compute-optimal rebalancing is a pure allocation gain. — [Hoffmann et al., arXiv 2203.15556](https://arxiv.org/abs/2203.15556)
- **Epoch / Ho et al. (Mar 2024):** compute needed to reach a fixed LM performance **halves about every 8 months** (95% CI 5–14 months), or about 3×/yr, since 2012. Compute scaling contributed more to progress than algorithms. — [Ho et al., arXiv 2403.05812](https://arxiv.org/abs/2403.05812); [MIT CSAIL news](https://www.csail.mit.edu/news/recurrent-networks-gpt-4-measuring-algorithmic-progress-language-models)
- **OpenAI (2020), historical vision baseline:** 44× less compute to reach AlexNet-level ImageNet performance in 2019 than in 2012, a 16-month halving. — [Hernandez & Brown, "Measuring the Algorithmic Efficiency of Neural Networks"](https://cdn.openai.com/papers/ai_and_efficiency.pdf)
- **Gundlach et al. (MIT, Nov 2025):** algorithms are estimated to have raised training FLOP efficiency about **22,000×** over 2012–2023. Ablation experiments find most evaluated innovations give **small, scale-invariant gains totalling under 10×**. Two strongly **scale-dependent** changes, LSTM→Transformer and Kaplan→Chinchilla rebalancing, account for **91%** of efficiency gains when extrapolated to the 2025 compute frontier. As a result, algorithmic progress for *small* models has been far slower than assumed, and efficiency measures are strongly reference-dependent. — [arXiv 2511.21622](https://arxiv.org/abs/2511.21622); [Semantic Scholar](https://www.semanticscholar.org/paper/On-the-Origin-of-Algorithmic-Progress-in-AI-Gundlach-Fogelson/6e3dde1f3c16c761d5b171e051bf7ae36a06b473)
- **Inference-side progress:** Epoch found the price of reaching a fixed benchmark score fell **9×–900× per year** depending on task, with a median of about 50×/yr and about 200×/yr for post-2024 models. The price of GPT-4-level PhD-science performance fell about 40×/yr. — [Epoch: LLM inference price trends](https://epoch.ai/data-insights/llm-inference-price-trends); [Epoch on X](https://x.com/EpochAIResearch/status/1900264630473417006)

### Inferences
- The speedrun's roughly 40× (wall-clock, fixed hardware) and roughly 25× (tokens) over about 26 months show what aggressive tuning can do at **tiny scale (about 124M params, about 1e17–1e18 FLOPs)**. Gundlach et al. and Wen et al. suggest such small-scale gains **don't transfer one-to-one** to frontier scale, and some of them shrink.
- Optimizers: plan on **about 1.1–2×**, with the low end independent and the high end vendor-reported. µP-style transfer removes tuning overhead, a pipeline saving that doesn't show up in final-run FLOPs.
- A key strategic point for the report: the biggest historical efficiency gains are **scale-dependent**; they pay off *more* at larger compute. That cuts against the premise of "extremely low compute ASI", because the multipliers that exist are largest exactly where compute is large.

### Gaps
- There is no rigorous published decomposition of the modded-nanogpt speedup into algorithmic (per-FLOP) vs systems (per-second) parts.
- Epoch's newer (2025–2026) algorithmic-progress revisions ("Revisiting algorithmic progress") couldn't be fetched, so their updated numbers aren't included.
- I have no independent verification of Muon's 2× at scales above 3B active.

---

## Q9. Overall: which gains multiply, which overlap, and what total OOM reduction is plausible?

### Takeaway
Stacking the demonstrated, mostly orthogonal levers gives roughly **1.5–2.5 OOM (about 30–300×) lower training cost** for a fixed capability level, versus a 2022-style dense BF16/AdamW Chinchilla baseline. The levers are data quality (2–7×), fine-grained MoE (3–10×), optimizer/µP (1.1–2×), low-precision hardware FLOPs (1.5–3× cheaper per FLOP) and modest architecture gains. This is roughly consistent with observed frontier trends (about 3×/yr, and DeepSeek-V3 ≈ 1/6 of GPT-4's estimated FLOPs about 20 months later).

**Inference** has larger multipliers: 2–3+ OOM through active-param sparsity, distillation into small models, 1–2-bit weights, linear or hybrid attention and memory offload. That matches Epoch's 9–900×/yr price declines. Several of the biggest "savings" (distillation, synthetic data, catch-up) **import upstream frontier compute**, and the largest historical multipliers are **scale-dependent**. Both facts undercut the idea that a very low-compute pipeline could reach new frontier capability rather than just re-derive existing capability cheaply.

### Cited Findings
- Frontier algorithmic progress is about 3×/yr (8-month halving). — [Ho et al., arXiv 2403.05812](https://arxiv.org/abs/2403.05812)
- Catch-up progress, including distillation and post-training, is about 16–60×/yr. — [MIRI TGT](https://techgov.intelligence.org/blog/catch-up-algorithmic-progress-might-actually-be-60x-per-year)
- Scale-invariant innovations total under 10×, while scale-dependent changes (Transformer, Chinchilla) account for 91% of gains at the 2025 frontier. — [Gundlach et al., arXiv 2511.21622](https://arxiv.org/abs/2511.21622)
- Optimizer speedups shrink with scale (about 1.4×→1.1× from 0.1B→1.2B). — [Wen et al., arXiv 2509.02196](https://arxiv.org/abs/2509.02196)
- MoE leverage grows with compute budget (EL > 7× at 1e22 FLOPs). — [arXiv 2507.17702](https://arxiv.org/abs/2507.17702)
- Low precision conflicts with heavy overtraining: post-training quantization degradation rises with data. — [Kumar et al., arXiv 2411.04330](https://arxiv.org/abs/2411.04330)
- Data filtering is bounded by repetition limits beyond about 4 epochs. — [Muennighoff et al., arXiv 2305.16264](https://arxiv.org/abs/2305.16264)
- Training↔inference trade is about 1 OOM saved for 1–2 OOM spent. — [Epoch](https://epoch.ai/publications/trading-off-compute-in-training-and-inference)
- A real stacked system: Qwen3-Next (MoE plus a hybrid linear-attention mix) used under 10% of dense Qwen3-32B's GPU-hours for equal quality, with more than 10× long-context throughput. — [Qwen blog](https://qwen.ai/blog?from=research.latest-advancements-list&id=4074cca80393150c248e508aa62983f9cb7d27cd)
- A second stacked system: Nemotron 3 Nano (hybrid Mamba + MoE, 3.2B active) matched Qwen3-30B-A3B-Thinking with 3.3× throughput. — [arXiv 2512.20848](https://arxiv.org/abs/2512.20848)
- NVIDIA's NVFP4 12B/10T run combined 4-bit training with a hybrid Mamba-Transformer. — [arXiv 2509.25149](https://arxiv.org/abs/2509.25149)
- Inference price for fixed capability falls 9–900×/yr (median about 50×). — [Epoch](https://epoch.ai/data-insights/llm-inference-price-trends)
- Frontier capability reaches a single consumer GPU in about 6–12 months. — [Epoch](https://epoch.ai/data-insights/consumer-gpu-model-gap)

### Inferences

**Approximate stacking table (training compute at fixed capability, vs a 2022 dense BF16 AdamW Chinchilla-optimal baseline; my synthesis of the cited findings):**

| Lever | Training multiplier | Inference multiplier | Orthogonality / overlap notes |
|---|---|---|---|
| Data filtering/quality (DCLM, FineWeb-Edu) | 2–7× | indirect (enables smaller models) | Mostly orthogonal to architecture; bounded by repetition limits; overlaps with synthetic data |
| Synthetic data / distillation (Phi, R1-distill) | large apparent (≥10× for the student) | large (enables small models) | **Imports teacher compute**; overlaps with data quality; not a frontier-advancing saving |
| Fine-grained MoE | 3–10× (grows with scale) | ≈ active/total ratio in FLOPs (10–30×), but memory ∝ total | Largely orthogonal to data (Ling EL measured on the same data) |
| Hybrid linear attention / SSM | ~1–1.5× | 3–10× at long context; 75% less KV | Orthogonal to MoE (Qwen3-Next and Nemotron 3 stack both); matters more for long reasoning traces |
| Optimizer (Muon) + µP | 1.1–2× | none | Gains shrink with scale; overlap with tuning quality |
| FP8 → FP4 training | 1.5–3× cheaper FLOPs (hardware) | FP8/FP4 serving 2–4× | Hardware-dependent; conflicts with heavy overtraining for post-training quantization |
| Ternary/1–2-bit weights (BitNet) | ~1× (latent full-precision weights) | 5–20× energy and memory | Proven only at about 2B scale natively |
| Retrieval / memory (RETRO, Engram) | ~2× compute-equivalent; 10–50× fewer params for knowledge | cheap DRAM/SSD instead of HBM | Separate sparsity axis; mostly helps knowledge, not reasoning |
| Test-time compute | can save ~1 OOM training | costs 1–2 OOM more per query | Trade, not saving; net negative at high query volume |

- **Training:** multiplying the midpoints (about 4 × 5 × 1.3 × 1.3 × 2) gives about 70×. The plausible range is **about 30–300× (1.5–2.5 OOM)** in cost at fixed capability. It isn't fully multiplicative: optimizer gains shrink at scale, filtering hits repetition limits, and low-bit conflicts with overtraining. This matches the observed about 3×/yr trend over about 4 years (3⁴ ≈ 80×).
- **Inference:** small, distilled, MoE, hybrid and low-bit stacks plausibly give **2.5–4 OOM** lower cost per query at fixed capability within about 2 years of a frontier release, consistent with Epoch's 50–200×/yr price declines. Much of this, however, relies on distillation from frontier models.
- **Across the entire pipeline:** the cheapest demonstrated path to a *given* capability is "frontier trains once, then distil into small, sparse, low-bit, hybrid, retrieval-augmented students". Reaching a *new* capability level (ASI) cheaply is not demonstrated by any of these levers. The largest multipliers (Transformer, Chinchilla, MoE leverage) grow with scale, so shrinking compute also shrinks the gains (Gundlach et al.).
- For the "most likely direction" question: the evidence favours a **stacked, sparse, modular design**, meaning fine-grained MoE plus conditional memory (Engram or retrieval) plus hybrid linear attention plus low-precision training and serving, trained on curated or synthetic data and amplified by RL and test-time reasoning with verifiers. None of this is a single breakthrough. At documented rates (about 3×/yr frontier; tens of × per year for catch-up), algorithmic efficiency alone gives about 1 OOM every 2 years at the frontier. Claims of 3+ OOM total reductions for *novel* capability have no support in the mainstream-DL evidence gathered here.

### Gaps
- No controlled, published experiment stacks *all* levers against a matched dense baseline at a frontier-relevant scale above 1e24 FLOPs. Qwen3-Next and Nemotron 3 are partial stacks with vendor-only evaluation.
- Upstream compute for synthetic data and distillation (teacher inference plus teacher pretraining amortization) is undisclosed across the industry, so "full-pipeline" compute can't be computed for any distilled small model.
- Epoch's 2025–2026 updated algorithmic-progress estimates, and the International AI Safety Report 2026's efficiency figures ([arXiv 2602.21012](https://arxiv.org/abs/2602.21012)), couldn't be read in full this session.
- 2026 frontier details (DeepSeek-V4 compute, Kimi K3 architecture, Qwen3.5 small-model official scores, Gemma 4 training tokens) remain unverified or undisclosed.
