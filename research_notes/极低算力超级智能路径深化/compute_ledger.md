# Full-Pipeline Compute Ledger for Representative AI Systems, and Re-verification of Load-Bearing Numbers (as of Sept 2026)

**Method and labelling (read first).** In this session epoch.ai, arxiv.org, huggingface.co, *.substack.com, alphaxiv, openreview, semanticscholar and archive.org were all blocked, both for direct fetch and for WebFetch. The shared WebSearch budget also ran out partway through. So nearly every number below comes from one of these routes:
- primary files on GitHub: model READMEs, the NVARC paper PDF, and LaTeX/text mirrors of papers such as the DeepSeek-R1 v2, Qwen3, Densing Law, KataGo, EfficientZero, Dreamer 4, V-JEPA 2 and BabyLM-2025 Findings texts;
- **GitHub mirrors of Epoch AI's "Notable AI Models" export**, with the same columns as Epoch's CSV, including Epoch's own "Training compute notes";
- a few search snippets from earlier in the session.

Labels used throughout:
- **[S]** sourced from a primary text or Epoch data export.
- **[S2]** secondary source quoting a primary.
- **[snip]** search-snippet only.
- **[A]** my assumption.
- **[A†]** a standard spec value I did not re-fetch.
- **[E]** my arithmetic or estimate.

Hardware constants used:
- H100/H800 dense BF16 = 989 TFLOP/s [S]; the 1,979 figure is with sparsity ([h100-vs-4090 text](https://github.com/bojieli/ai-infra-book/blob/main/references/text/h100-vs-4090.txt)).
- A100 = 312 TFLOP/s [S], same source.
- RTX 4090 = 330 TFLOP/s FP16 [S], same source.
- TPU v1 = 92 TOPS INT8 [S] ([TPU v1 paper text](https://github.com/bojieli/ai-infra-book/blob/main/references/text/tpu-v1.txt)).
- TPU v5p = 459 TFLOP/s BF16 [S] ([Google v5p doc text](https://github.com/bojieli/ai-infra-book/blob/main/references/text/google-v5p.txt)).
- V100 = 125 TFLOP/s FP16 tensor [A†].
- RTX 3090 = 71 TFLOP/s FP16 with FP32 accumulate [A†].
- RTX 4070 = 29 TFLOP/s FP32 [A†].
- L4 = 121 TFLOP/s BF16 dense [A†].

Conversions: training ≈ 6·N·D; inference ≈ 2·N_active per token; hardware-time FLOP = GPU-seconds × peak × utilisation (MFU).

"Full-pipeline" compute is (a) own pretraining + (b) borrowed upstream compute + (c) own RL/post-training + (d) inference, with (d) given per task or episode. Borrowed compute is reported three ways:
- **full attribution**: the whole teacher and base-model pretraining is charged to the student;
- **inference-only**: only the teacher's data-generation FLOP is charged;
- **own-only**: nothing upstream is charged.

---

## Q1. Per-system arithmetic: pretraining, borrowed compute, RL/post-training, and inference per task

### Takeaway
Full-pipeline compute for the ledger rows spans about 22 orders of magnitude. The BabyLM 2025 winner uses about 1.5e18 FLOP, TRM about 1e20, KataGo about 1e21, DeepSeek-V3/R1 about 3.5e24, GPT-4 2.1e25, Grok 4 5e26, and evolution about 1e41. The two structural facts that dominate the ledger:
1. **Epoch now excludes self-play and synthetic-data generation from "training compute."** AlphaGo Zero fell from 3.4e23 to 6.5e20, but self-play is about 99.8% of its real cost.
2. **Every small model that reaches LLM-class skill (Phi-4, R1-distills, NVARC's Qwen3-4B) carries 95–99.99% borrowed compute under full attribution.**

### Cited Findings

**Epoch dataset (mirrors of Epoch's Notable-AI-Models export)**
- Epoch's current AlphaGo Zero entry is **6.49e20 FLOP**, with the note: *"Updating this compute estimate to only account for direct training compute, not synthetic data generation compute."* The arithmetic in the note: forward pass 2 × 17,048,888,774 = 3.41e10 FLOP per position; 3.1M mini-batches × 2,048 = 6.35e9 positions; training = 3 × 3.41e10 × 6.35e9 = 6.5e20. The dataset-size note keeps the older "200 × 29e6 = 5.8e9" positions figure. Parameters: 46.4M. Hardware: TPU v1; 480 h. — [Epoch export mirror with notes](https://github.com/leizungjyun/spring2023/blob/main/collections/research-talks/neuro-group-opening-keynote/docs/evidence/epoch-selected-models.csv); same value in [Epoch export mirror (2026)](https://github.com/soeunpark98/ai-carbon-visualization/blob/main/public/data/part3/3a.csv)
- Epoch's AlphaZero entry is **1.06e20 FLOP "direct training compute"**. The note says the Go version trained for 34 h using 5k TPUv1 for data generation and 64 TPUv2 for parameter updates. — [Epoch export mirror with notes](https://github.com/leizungjyun/spring2023/blob/main/collections/research-talks/neuro-group-opening-keynote/docs/evidence/epoch-selected-models.csv)
- Other Epoch values [S] from the export mirror updated to early 2026 ([3a.csv](https://github.com/soeunpark98/ai-carbon-visualization/blob/main/public/data/part3/3a.csv)):

  | Model | Epoch training compute (FLOP) | Notes |
  |---|---|---|
  | KataGo | 2.32e19 | 2.5M params |
  | MuZero | 4.8e19 | |
  | DreamerV3 | 2.2032e20 | 200M params |
  | V-JEPA 2 | 9.06e21 | 1B params |
  | V-JEPA (2024) | 1.64e21 | |
  | Phi-4 | 9.32e23 | |
  | phi-3-mini | 7.52e22 | |
  | DeepSeek-V3 | 3.3e24 | |
  | DeepSeek-R1 | 3.5e24 | |
  | Qwen3-4B | 8.64e23 | |
  | Qwen3-32B | 7.08e24 | |
  | Qwen3-235B-A22B | 4.75e24 | |
  | Qwen2.5-Math-1.5B | 1.75e23 | |
  | Qwen2.5-Math-7B | 8.64e23 | |
  | Qwen2.5-32B | 3.51e24 | |
  | gpt-oss-120b | 4.94e24 | 116.83B params |
  | Llama 3-8B | 7.2e23 | |
  | Chinchilla | 5.76e23 | |
  | GPT-3 175B | 3.14e23 | |
  | OpenAI Five | 6.7e22 | |
  | AlphaStar | 1.08e23 | |
  | GPT-4 | **2.1e25** | 1.8T params |
  | Grok 3 | **3.5e26** | |
  | Grok 4 | **5.0e26** | |
  | Kimi K2.5 (2026) | 5.8e24 | |
  | GLM-5 (2026) | 6.84e24 | |
- A second Epoch-derived table gives **GPT-4.5 = 3.8e26** and Claude 3.5 Sonnet = 2.7e25. — [ai-dispatch models.csv](https://github.com/suveer-dhawan/ai-dispatch/blob/main/data/models.csv)

**Self-play game agents**
- **AlphaGo Zero.** The 20-block run used 4.9M self-play games and 700k × 2,048 training steps; the 40-block run used 29M games and 3.1M × 2,048. Search used 1,600 MCTS simulations per move; inference ran on 4 TPUs. [S2, compilation of the Nature paper] — [DeepMind run configs](https://github.com/HenryTBCassidy/AlphaBlokus/blob/main/docs/research/deepmind-run-configs.md). It reached superhuman level after 3 days (earlier-round source) — [LessWrong](https://www.lesswrong.com/posts/shnSyzv4Jq3bhMNw5/alphago-zero-and-the-foom-debate)
- **AlphaZero.** 5,000 first-generation TPUs ran self-play and 64 second-generation TPUs trained. Games: chess 44M, shogi 24M, Go 21M. Search used 800 simulations per move; training ran 700k steps × 4,096. The network is a 20×256 residual tower with 119 chess input planes and a 4,672 (8×8×73) chess policy. Wall-clock: chess 9 h, shogi 12 h, Go 34 h. It passed prior state-of-the-art at chess 4 h, shogi under 2 h and Go 8 h. [S2] — [DeepMind run configs](https://github.com/HenryTBCassidy/AlphaBlokus/blob/main/docs/research/deepmind-run-configs.md)
- **KataGo (2019).**
  - The main run "lasted for 19 days using a maximum of 28 V100 GPUs at any time (averaging 26–27) and generated about 241 million training samples across 4.2 million games". It "surpasses ELF's final model after only 19 days on fewer than 30 GPUs", a "50x reduction in computation over comparable methods".
  - For comparison, AlphaZero's Go run was "about 41 TPU-years", and ELF OpenGo used "2000 V100 GPUs for about 13–14 days, or about 74 GPU-years". [S]
  - — [KataGo paper LaTeX source](https://github.com/FoAKTEE/az/blob/main/ref-paper/arxiv-1902.10565/src/Accelerating_Self_Play_Learning_In_Go_2020.tex); [arXiv 1902.10565](https://arxiv.org/abs/1902.10565)
  - The 2026 README says "a training run using only a *single* top-end consumer GPU could possibly train a bot from scratch to superhuman strength within a few months." [S, author's speculation] — [KataGo README](https://github.com/lightvector/KataGo)
- **EfficientZero (NeurIPS 2021).**
  - It scores 194.3% mean and 109.0% median human-normalised on Atari 100k (26 games, 2 h of real-time play). The paper's own Table 1 text says 1.904 / 1.160 instead.
  - DQN reaches 220% / 96% with 500× more data (200M frames).
  - "To train an Atari agent for 100k steps, it only needs 4 GPUs to train 7 hours." [S]
  - — [EfficientZero paper text](https://github.com/qpwo/rules/blob/main/writings/arxiv.org__pdf__2111.00210.pdf.txt). The GPUs are RTX 3090s (4 GPUs × 3090) [S] — [EfficientZero README](https://github.com/YeWR/EfficientZero)

**World-model agents**
- **DreamerV3 (Nature 2025).** Each agent uses a single A100 and a default size of 200M params. Minecraft Diamond runs 100M environment steps, and "all Dreamer seeds collect diamonds" with no human data or curriculum. VPT used "720 GPUs for 9 days; Dreamer uses 1 GPU for 9 days". [S2, knowledge-base summary of the Nature paper] — [DreamerV3 KB](https://github.com/ABA-WM-Knowledge-Base/ABA-WM-Knowledge-Base/blob/main/papers/dreamerv3/paper.md)
- **Dreamer 4 (Sept 2025).** [S] — [Dreamer 4 LaTeX mirror](https://github.com/edwhu/dreamer4-jax/blob/main/docs/main.txt); [arXiv 2509.24527](https://arxiv.org/abs/2509.24527)
  - "We train models with 2B parameters—400M for the tokenizer and 1.6B for the dynamics model—on 256 to 1024 TPU-v5p." Minecraft uses 256 spatial tokens per frame with 192 frames of context.
  - Data is the VPT contractor set: 2,541 h of 360p video at 20 FPS with actions. It uses about 100× less data than VPT.
  - Results: diamonds in **0.7% of episodes**, iron pickaxe 29%, over 90% success up to the stone pickaxe. VPT (finetuned) reaches only sticks (53%).
  - The world model runs at 20 FPS real-time on one H100. Total training duration is not stated in the text I obtained.
- **V-JEPA 2 (June 2025).** [S] — [V-JEPA 2 paper text](https://github.com/sfreedoms2035/-AgenticWorkflowDataGenerationProblemTasks/blob/main/Input/2506.09985v1.txt); [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
  - Encoder: ViT-g, 1B params. Data: over 1M hours of video plus 1M images.
  - Schedule: 252K iterations (12K warmup, 228K constant, 12K cooldown) at global batch 3,072. Clips are 16 frames at 256² until the cooldown, then up to 64 frames at 384².
  - "Training our ViT-g model on 64 × 384 × 384 inputs would require roughly 60 GPU-years"; progressive resolution gives "an 8.4× reduction in GPU time".
  - The AC stage uses 62 h of Droid robot video.

**ARC-AGI systems**
- **TRM.** 7M params, "45% on ARC-AGI-1 and 8% on ARC-AGI-2" (self-reported). The ARC-AGI-1 run assumes 4 H100 GPUs with "*Runtime:* ~3 days"; config H_cycles=3, L_cycles=4, L_layers=2. [S] — [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels). ARC Prize verified **40% ARC-AGI-1 at $1.76/task and 6.2% ARC-AGI-2 at $2.10/task** (earlier-round notes) — [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis). It won 1st place in the ARC Prize 2025 Paper Award — [ARC Prize 2025 Technical Report mirror](https://github.com/tiendungchs/PersonalWiki/blob/main/raw/ARC%20Prize%202025%20Technical%20Report.md)
- **HRM.** ARC-1 uses 960 examples and ARC-2 uses 1,120; experiments "assume an 8-GPU setup" with "*Runtime:* ~24 hours" for ARC. Sudoku takes about 10 h on an RTX 4070 laptop GPU. [S] — [HRM README](https://github.com/sapientinc/HRM). ARC Prize verified 32% on ARC-AGI-1 against a self-reported 40.3% (earlier-round notes) — [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)
- **CompressARC.** It has 76K parameters, "achieves 20% on the ARC-AGI-1 evaluation set, processing each puzzle in approximately 20 minutes on a single NVIDIA RTX 4070 GPU". It reaches 20–34% on ARC-AGI-1 and 4% on ARC-AGI-2 with no pretraining, and won 3rd place in the Paper Award. [S] — [ARC Prize 2025 Technical Report mirror](https://github.com/tiendungchs/PersonalWiki/blob/main/raw/ARC%20Prize%202025%20Technical%20Report.md); [CompressARC README](https://github.com/iliao2345/CompressARC) ("Most tasks may take up to 20 minutes to run, on one NVIDIA GeForce RTX 4070 GPU")
- **NVARC (ARC Prize 2025 Kaggle winner, 24.03% on the ARC-AGI-2 private set, about $0.20/task).** [S] — [NVARC paper (PDF in repo)](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf); [NVARC README](https://github.com/1ytic/NVARC); [ARC Prize 2025 report mirror](https://github.com/tiendungchs/PersonalWiki/blob/main/raw/ARC%20Prize%202025%20Technical%20Report.md)
  - The synthetic-data generation (SDG) pipeline used three models: claude-opus-4-20250514, claude-sonnet-4-5-20250929 and gpt-oss-120b. "The majority of generated data is made with gpt-oss-120b." Running gpt-oss-120b "on a single node with 8xH100 GPUs gives us 15k tokens/s generation throughput."
  - Stages:
    - 716 summaries with Opus 4 plus 2×716 with gpt-oss;
    - 91 + 1,000 summaries with Sonnet 4.5;
    - **266,593** mixed summaries (gpt-oss only);
    - **126,901** filtered input-grid programs, at about 70% / 50% acceptance (gpt-oss only);
    - output programs sampled repeatedly per input program (e.g., 20 generated, 15 consistent);
    - **103,253** final puzzles, about 30 I/O pairs each.
  - Fine-tuning data is 3.2M augmented samples (3,255,481). "To do a full fine-tuning of the 4B model we used **4 nodes of 8xH100 for 27 hours**."
  - Test-time: LoRA TTT per puzzle within the Kaggle limit of "12 hours with 4 L4 GPUs for solving 240 tasks".
  - The TRM component was trained "in 24 hours on 8xH100", but "TRM added nothing" on most puzzles; only about 2–3 extra puzzles were solved.
  - Scores: 27.64% on the public leaderboard; a post-deadline Qwen3-4B-Thinking-2507 variant reached 29.72%.
- **Qwen3 (NVARC's base).** It was "pre-trained on 36 trillion tokens". "For smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models". Logits are aligned "with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B)", and this requires "only 1/10 of the GPU hours compared to the four-stage training method". [S] — [Qwen3 Technical Report text](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/qwen3.txt); [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- **gpt-oss-120b** "required 2.1 million H100-hours"; gpt-oss-20b required 210K. [S2, quoting OpenAI's model card] — [Milvus blog](https://github.com/milvus-io/community/blob/master/blog/en/gpt-oss-vs-o4-mini-edge-ready-on-par-performance-dependable-not-mind-blowing.md). Its active parameter count is 5.1B [A†, OpenAI model card, not re-fetched].

**Language models**
- **DeepSeek-V3.** "Requires only 2.788M H800 GPU hours for its full training"; 2.664M of that is pre-training on 14.8T tokens and about 0.1M is post-training. Post-training "distill[s] reasoning capabilities from … one of the DeepSeek R1 series models". Benchmarks: MMLU 88.5, GPQA-Diamond 59.1, AIME-2024 39.2, MATH-500 90.2. [S] — [DeepSeek-V3 README](https://github.com/deepseek-ai/DeepSeek-V3)
- **DeepSeek-R1 (arXiv v2, 4 Jan 2026, Supplementary B.4.4).** [S] — [DeepSeek-R1 paper v2 text](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/deepseek-r1.txt); [R1 README](https://github.com/deepseek-ai/DeepSeek-R1)
  - R1-Zero "employed 64*8 H800 GPUs … approximately 198 hours"; R1 used "the same 64*8 H800 GPUs … roughly 80 hours"; "To create the SFT datasets, we use 5K GPU hours."
  - Table 7: **101K + 5K + 41K = 147K H800-hours = $294K** at $2/GPU-hour.
  - R1 scores: AIME-2024 79.8, MATH-500 97.3, GPQA-Diamond 71.5.
- **R1 distillation set.** About 600k reasoning samples come from rejection sampling ("sample multiple responses and retain only the correct ones") on the RL checkpoint, plus about 200k non-reasoning samples. Table 5 totals **804,745 samples averaging 5,355.3 tokens**. Distillation fine-tunes "the corresponding base model for 2–3 epochs using the 800k data". Bases: Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Llama-3.1-8B and Llama-3.3-70B-Instruct. [S] — [R1 v2 text](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/deepseek-r1.txt)
- **R1-Distill scores (AIME-24 / MATH-500 / GPQA).** 1.5B: 28.9 / 83.9 / 33.8. 7B: 55.5 / 92.8 / 49.1. 32B: 72.6 / 94.3 / 62.1. [S] — [R1 README](https://github.com/deepseek-ai/DeepSeek-R1)
- **Phi-4.**
  - Training: "9.8T tokens, 21 days, 1920 H100-80G GPUs". [S2, HF card mirror] — [phi-4 card mirror](https://github.com/ilayzeidman/open-source-llm-models-wiki/blob/main/raw/phi-4-hf-card.md)
  - Pretraining mix: web 15% (1.3T unique tokens, 1.2 epochs); web rewrites 15% (290B, 5.2 epochs); **synthetic 40% (290B, 13.8 epochs)**; code 20% (820B, 2.4); acquired sources 10% (580B, 1.7). [S2, table reproduced from arXiv 2412.08905] — [paper review mirror](https://github.com/deep-diver/ai-paper-reviewer/blob/main/content/paper-reviews/2412.08905/index.md)
  - The teacher is GPT-4o. Phi-4 beats it on GPQA (56.1 vs 50.6) and MATH (80.4 vs 74.6). — [Phi-4 Technical Report](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/P4TechReport.pdf) (earlier-round notes)
  - Epoch's GPT-4o estimate is **3.8e25 (low confidence)**. [S2] — [xrisk-pause-game compute note citing Epoch](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/compute-requirements-trends.md)
- **BabyLM 2025 (3rd challenge).** [S] — [Findings of the Third BabyLM Challenge, text mirror](https://github.com/juand-r/claude-sandbox/blob/main/explorations/babylm2025/papers/p28.txt); [Simple Diffusion paper text](https://github.com/juand-r/claude-sandbox/blob/main/explorations/babylm2025/papers/p38.txt)
  - Training exposure is capped, counting repeats, at "at most 100M words for the Strict-Small track and at most 1B words for the other tracks".
  - **Strict-track NLP winner: "Simple Diffusion" (Kosmopoulou et al.)**, a masked-diffusion LM with "126.6 M parameters … sequence length of 512. The batch size was set to 512 … 10 epochs, or 7530 training steps."
  - Strict scores (human-likeness / NLP / macro): Simple-Diffusion 12.6 / 58.4 / 35.5; GPT-BERT baselines NLP 57.8–63.0; GPT-2 baseline NLP 55.4. Strict-Small NLP winner: AMLM-Hard-Decay (NLP 58.3).
  - The GPT-BERT baseline is about 120M params, batch 131,072 tokens, 12,330 steps.
  - An independent audit found the macro aggregate unreliable: a mis-trained model "beat every properly-trained model" on it. [S2] — [BabyLM 2025 analysis](https://github.com/juand-r/claude-sandbox/blob/main/explorations/babylm2025/analysis.md)

**Frontier models**
- GPT-5 was "likely trained on less compute than its immediate predecessor, GPT-4.5". A snippet summary gives "approximately 5×10^25 FLOP total, including both pre-training and reinforcement learning … more than twice as much as GPT-4 but less than GPT-4.5". [snip] — [Epoch substack](https://epochai.substack.com/p/why-gpt-5-used-less-training-compute); [Epoch](https://epoch.ai/gradient-updates/why-gpt5-used-less-training-compute-than-gpt45-but-gpt6-probably-wont)
- Coverage of 2026 closed models is poor. In Epoch's database, of models released after 2025-01-01, training-compute estimates exist for only 4/51 OpenAI models, 1/16 Anthropic, 9/47 Google DeepMind and 2/15 xAI. The latest release date in the database was 2026-08-14, and August-2026 closed models (GPT-5.6 Cyber, Grok 4.6, Gemini 3.7 Flash) have no FLOP estimate. Top ECI entries: Claude Fable 5 (162.5), GPT-5.5 Pro (161.7), Claude Opus 5 (161.6). [S2, audit dated 2026-08-31] — [ai-model-world data-source audit](https://github.com/liyupi/ai-model-world/blob/main/docs/data-sources-research.md)

**Biological anchors**
- Carlsmith "thinks it more likely than not that 10^15 FLOP/s is enough", and "it unlikely (<10%) that more than 10^21 FLOP/s is required". The mechanistic method gives "~1e13–1e17 FLOP/s". [snip] — [Coefficient Giving / Open Phil report](https://coefficientgiving.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/); [EA Forum summary](https://forum.effectivealtruism.org/posts/nGQJEYp5X2pCbeweg/new-report-on-how-much-computational-power-it-takes-to-match)
- Cotra's lifetime anchor is a median of 1e24 FLOP (≈1e15 FLOP/s × ~30 years, for ages 0–32). The evolution anchor is a median of about 1e41 FLOP (an average ancestor at nematode-level FLOP/s, with about 1e21 ancestors alive at any time) and carries 10% weight. [snip] — [Epoch: Grokking bio-anchors](https://epoch.ai/blog/grokking-bioanchors); [Alignment Forum mirror](https://www.alignmentforum.org/posts/wgio8E758y9XWsi8j/grokking-forecasting-tai-with-biological-anchors)

### Inferences
Per-row arithmetic. Every non-[S] input is flagged.

1. **AlphaGo Zero**
   - *40-day, 40-block run* [E]: self-play = 29e6 games [S] × 200 moves/game [A, Epoch's own older assumption] × 1,600 sims [S] × 3.41e10 FLOP [S] = **3.16e23**. Training = 6.5e20 [S]. **Total ≈ 3.2e23**, which reproduces Epoch's old 3.41e23. Self-play is **99.8%** of the total.
   - *3-day, 20-block run* [E]: the forward pass is 2 × (14.14M + 20 × 425.85M + 0.63M connections) = 1.71e10 FLOP. Self-play = 4.9e6 × 200 × 1,600 × 1.71e10 = **2.7e22**; training = 7.3e19.
   - Inference per move: 1,600 × 3.41e10 = **5.5e13 FLOP**. A human professional spending about 30 s per move at 1e15 FLOP/s uses 3e16, about 550× more.
2. **AlphaZero** [E]
   - *Chess*: forward pass ≈ 3.15e9 FLOP [E from the 20×256 tower on 8×8]. Self-play = 44e6 [S] × 100–150 plies [A] × 800 [S] × 3.15e9 = **1.1e22–1.7e22**. Hardware check: 5,000 TPUv1 × 92e12 × 9 h = 1.5e22 peak, so the two agree. Superhuman against Stockfish at 4 h ≈ **5e21–7e21**.
   - *Go*: 21e6 × 200 [A] × 800 × 1.71e10 = **5.7e22**, against a hardware peak of 5.6e22 for 34 h. So **~3e22–6e22**.
3. **KataGo** [E]: 26.5 × 19 d × 24 h = 12,084 V100-hours × 125e12 [A†] = 5.4e21 peak. At 10–40% utilisation [A] that is **5e20–2e21, about 1e21**. Epoch's 2.32e19 counts gradient updates only. ELF OpenGo: 74 GPU-years → 2.9e23 peak, **about 9e22** at 30%. The README's single-consumer-GPU claim works out to 330e12 × 0.3 × 90 d ≈ 7.7e20 [E; speculative].
4. **EfficientZero** [E]: 4 × 7 h × 71e12 [A†] = 7.2e18 peak per game. Utilisation is 5–30% [A], because MCTS runs on CPU with small conv nets. That gives **3.6e17–2.1e18 per game**, or **9e18–5.6e19 for the 26-game suite**. For comparison, a human's 2 h of play at 1e15 FLOP/s is 7.2e18.
5. **DreamerV3 (Minecraft diamonds)** [E]: 1 A100 × 9 d = 2.4e20 peak; at 30–90% that is **7e19–2.2e20**, in line with Epoch's 2.2e20. No borrowed compute and no human data.
6. **Dreamer 4** [E]: 2,541 h × 3,600 × 20 FPS = 1.83e8 frames × 256 tokens = 4.7e10 tokens per epoch.
   - Dynamics: 6 × 1.6e9 × 4.7e10 × 1.5 (shortcut/bootstrap overhead [A]) × E epochs, with E = 5–50 [A], gives 3.4e21–3.4e22.
   - Tokenizer: 6e20–6e21.
   - **Total ≈ 4e21–4e22.** Sanity check: 256 v5p × 459e12 × 40% = 4.1e21 per day, so about 1–10 days on the smallest stated pod.
   - Borrowed: no compute, but 2.5K h of human gameplay.
   - Inference: real time on one H100 (≤ about 1e15 FLOP/s peak).
7. **V-JEPA 2** [E]: 252K × 3,072 × 2,048 tokens per clip [A: 2×16×16 tubelets, 16 frames at 256²] = 1.6e12 token-positions. The EMA-target forward pass alone costs 3.2e21; adding context-encoder forward/backward, the predictor and the 64-frame 384² cooldown gives **~1e22**, consistent with Epoch's 9.06e21. From GPU-time: 60 / 8.4 ≈ 7 GPU-years → 2.4e22 on A100s at 35%, or 7.6e22 if they were H100s (upper bound). **Range: 9e21–2.5e22.**
8. **TRM**
   - Training: 288 H100-h × 989e12 = 1.03e21 peak. At 5–30% MFU [A, tiny 7M model] that is **5e19–3e20**.
   - Inference per test input [E]: 3 × (4 + 1) = 15 net passes × 2 × 7e6 × 916 tokens [A: 900 grid + 16 puzzle-embedding] × 16 supervision steps (max) × 1,000 augmentations = **3.1e15 FLOP**, about 3 brain-seconds.
   - ARC Prize's $1.76/task at $2–3 per H100-h is 0.6–0.9 H100-h per task. That is far above the 3e15 algorithmic cost, so the priced figure is dominated by overhead or by the transductive training on evaluation demonstrations. This interpretation is unverified.
9. **HRM** [E]: 192 GPU-h, of unstated type. Peak is 2.2e20 (A100) to 6.8e20 (H100); at 10–30% that is **2e19–2e20**.
10. **CompressARC** [E]: 1,200 s × 29e12 [A†] = 3.5e16 peak per puzzle; for 400 evaluation puzzles, **≤1.4e19 in total** (a peak-based upper bound). Pretraining = 0.
11. **NVARC** [E]
    - Own compute:
      - full fine-tune: 864 H100-h → 3.1e21 peak → **0.9–1.5e21** at 30–50%;
      - SDG: about 3.0e6 generations (266,593 + ~215k input-program attempts + 126,901 × ~20 output programs) × 1k–4k tokens [A] = 3e9–1.2e10 tokens. At the reported 15k tok/s per node that is **450–1,800 H100-h**, i.e. model FLOP 2 × 5.1e9 × tokens × 1.5 (prefill [A]) = **5e19–2e20** (hardware-time peak bound 1.6e21–6.4e21);
      - TRM part: 192 H100-h → about 2e20;
      - TTT: 4 L4 × 121e12 [A†] × 43,200 s = 2.1e19 peak for 240 tasks, i.e. **8.7e16 peak per task** (2.6–4.4e16 at 30–50%).
      - **Own total ≈ 1.5e21–8e21.**
    - Borrowed:
      - Qwen3-4B pretraining 6 × 4.0e9 × 36e12 = **8.64e23** (matches Epoch);
      - the Qwen3 distillation teacher's pretraining, **4.75e24** (235B-A22B) or **7.08e24** (32B);
      - gpt-oss-120b pretraining, **4.94e24** (Epoch). As a check, 2.1M H100-h × 989e12 × 3,600 = 7.5e24 peak, implying about 66% utilisation against BF16 peak. That is plausible only if gpt-oss was trained in lower precision, and I have not verified that.
      - Claude Opus 4 / Sonnet 4.5 are undisclosed, and their use was small.
    - **Full-attribution total ≈ 1.1e25–1.3e25; borrowed share ≥ 99.9%.** Counting only the base model, it is ≈ 8.7e23 with own share 0.2–0.9%.
12. **DeepSeek-V3** [E]: 6 × 37e9 × 14.8e12 = **3.29e24** (Epoch: 3.3e24). Hardware: 2.788M × 3,600 × 989e12 = 9.9e24 BF16 peak, an implied **33% MFU** (17% of the FP8 peak). Borrowed: a small internal R1-preview distillation, inside DeepSeek's own pipeline and undisclosed.
13. **DeepSeek-R1** [E]
    - RL and SFT-data compute: 147K H800-h × 3,600 × 989e12 = 5.2e23 peak. At 10–30% [A, rollout-dominated] that is **5e22–1.6e23**. Epoch's R1 minus V3 is 3.5e24 − 3.3e24 = 2e23, consistent with the upper end.
    - **Total ≈ 3.5e24**; the V3 base is **~94%** of it.
    - Inference per AIME problem: about 1e4 tokens [A] × 2 × 37e9 = **7.4e14 FLOP**.
14. **R1-Distill-Qwen-1.5B / 7B / 32B** [E]
    - Kept distillation tokens = 804,745 × 5,355 = 4.3e9.
    - Teacher generation: 2–8× oversampling [A] × 1.2 prefill [A] × 2 × 37e9 = **7.7e20–3.1e21**. This is consistent with the 5K H800-h "SFT data creation" line (1.8e22 peak; 5e20–2.7e21 at 3–15% decode utilisation).
    - Student SFT at 2.5 epochs: 1.5B 9.7e19; 7B 4.9e20; 32B 2.1e21.
    - Upstream: V3 base 3.3e24, R1 RL about 1.5e23, data generation about 1e21. Student bases (Epoch): 1.75e23 / 8.64e23 / 3.51e24.
    - **Full-pipeline totals: 3.6e24 / 4.3e24 / 7.0e24.** Own-only (SFT) share is 0.003% / 0.01% / 0.03%, so borrowed is ≥99.97%.
    - The teacher chain alone is 95% / 80% / 50% of the total. The rest is the Qwen base-model pretraining, itself borrowed from Alibaba.
15. **Phi-4** [E]
    - Own: 6 × 14e9 × 9.8e12 = 8.2e23. Epoch gives 9.32e23; the hardware peak is 1,920 × 21 d × 989e12 = 3.45e24, an implied MFU of 27%.
    - Teacher inference: 580B unique model-generated tokens [S: 290B synthetic + 290B rewrites] × a generated-to-kept ratio of 1.5–5× [A] × 2 (prompt plus output [A]) × 2 × N_active. With N_active = 1e11–3e11 [A; GPT-4o's size is undisclosed], that is **3.5e23–3.5e24, central ~1.4e24**. So teacher inference is **27–79% of (own + teacher inference)**, comparable to Phi-4's own pretraining.
    - **Full attribution** (GPT-4o at 3.8e25 [S2]): about **4.0e25; borrowed ≈ 98%**.
16. **BabyLM 2025 Strict winner (Simple Diffusion)** [E]: 512 × 512 × 7,530 = 1.97e9 token-positions (an upper bound, since sequences are unpacked). FLOP = 6 × 126.6e6 × 1.97e9 = **1.5e18**. The GPT-BERT baseline is 6 × 120e6 × 1.62e9 = **1.2e18**. No borrowed compute.
17. **GPT-4** = **2.1e25** [S Epoch]. **Grok 4** = **5e26** [S Epoch], the largest Epoch-documented run. For a **2026 frontier model** I assume **5e26–3e27** [A]: Grok 4 × 1–6, consistent with about 4–5×/yr frontier growth. Closed 2026 models are undisclosed and unestimated by Epoch.
18. **Human brain (lifetime anchor)** [E]: 30 years = 9.47e8 s.
    - At 1e13 / 1e15 / 1e17 FLOP/s: **9.5e21 / 9.5e23 / 9.5e25**, i.e. the 1e22–1e26 range, central about 1e24. The <10% tail at 1e21 FLOP/s gives about 1e30.
    - To age 13 (about the BabyLM 1e8-word budget): **4.1e23**. To age 4: 1.3e23.
    - Inference: about 1e15 FLOP/s at 20 W.
19. **Evolution anchor**: about **1e41** [snip; Cotra].

### Gaps
- **Dreamer 4** training duration and step count were not in the text I retrieved. Its ledger value, 4e21–4e22, is built on assumptions (epochs and loss overhead).
- **GPT-4o** active parameters and **Phi-4's teacher sampling ratio** are undisclosed, so Phi-4's teacher-inference share (27–79%) is assumption-driven.
- **NVARC**: tokens per gpt-oss generation are not reported, so I assumed 1k–4k. Claude usage has no FLOP estimate. It is unclear which Qwen3-4B checkpoint (base, instruct or 2507-thinking) was used in the final Kaggle run; this changes how much Qwen3 teacher distillation is inherited.
- **Utilisation** for tiny models (TRM, HRM, CompressARC, EfficientZero) is unknown, so the ranges are 3–10× wide. Algorithmic FLOP counts from configs would narrow them. I computed one for TRM inference only.
- I did not find a primary statement of **gpt-oss-120b's active parameters or training precision** this session.
- **Qwen2.5-Math** bases were themselves built with synthetic data from larger Qwen models; this second-order borrowing was not quantified.

---

## Q2. Consolidated ledger: capability, generality and borrowed-compute share

### Takeaway
Narrow systems with an exact simulator or verifier reach superhuman or human-level performance at **1e18–1e22 FLOP**, i.e. 1e-6 to 1e-2 of the brain-lifetime anchor. Every system with broad, LLM-class language competence sits at **≥1e24 FLOP full-pipeline**. The cheap-looking ones (Phi-4 at 9e23 own, the R1-distills at 1e20 own, NVARC at ~5e21 own) import 95–99.99% of their compute from teachers and base models. No row combines "general" with "≤1e23 full-pipeline".

### Cited Findings
- All compute inputs, capability scores and sources are listed under Q1 (Epoch export mirrors; primary READMEs and paper texts).
- ARC-AGI dissections from the earlier round: HRM/TRM-style models train on evaluation tasks' demonstration pairs with per-task ID embeddings and collapse to 0% without them. That makes them transductive, per-benchmark learners. — [arXiv 2512.11847](https://arxiv.org/abs/2512.11847); [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)
- BabyLM organisers: models trained on ≤100M words "fall short of child-level and LLM competence". — [arXiv 2504.08165](https://arxiv.org/abs/2504.08165)

### Inferences
**The ledger** (FLOP; "own" = system-specific compute; "full" = full attribution of upstream compute; BLA = multiple of the 1e24 brain-lifetime anchor). Most own and full figures are [E], with inputs as in Q1.

| System (date) | Own (pretrain + RL/post) | Borrowed upstream | Full pipeline | Inference / task | Capability | Narrow / general | Borrowed share (full) | × BLA |
|---|---|---|---|---|---|---|---|---|
| BabyLM-25 Strict winner, Simple Diffusion 126.6M (2025) | 1.5e18 | 0 | **1.5e18** | ~2.5e8 per token (2 × 126.6M) | NLP 58.4 (below GPT-BERT baselines' 63.0); far below child/LLM | Language, weak | 0% | 1.5e-6 |
| EfficientZero (2021) | 4e17–2e18 per game (1e19–6e19 for 26 games) | 0 | **~1e18 per game** | MCTS per step (small) | Atari-100k mean 194% / median 109% HNS | Narrow (Atari) | 0% | 1e-6 |
| CompressARC 76K (2025) | 0 pretrain; ≤3.5e16 per puzzle test-time | 0 | **≤1.4e19** (400 puzzles) | ≤3.5e16 | ARC-AGI-1 20% (eval), 4% ARC-AGI-2 | Narrow (per-puzzle MDL) | 0% | ≤1e-5 |
| HRM 27M (2025) | 2e19–2e20 | 0 | **~1e20** | small | ARC-AGI-1 32% verified (40.3% claimed) | Narrow, transductive | 0% | 1e-4 |
| TRM 7M (2025) | 5e19–3e20 | 0 | **~1e20–3e20** | ~3e15 per test input (ARC price $1.76) | ARC-AGI-1 40% verified; ARC-AGI-2 6.2% | Narrow, transductive | 0% | 1e-4 |
| DreamerV3 (2023/2025) | 7e19–2.2e20 | 0 | **~2e20** | 1 GPU real-time | Minecraft diamonds from scratch (all seeds) | Narrow task, general algorithm (150+ tasks, fixed hyperparameters) | 0% | 2e-4 |
| KataGo (2019) | ~1e21 (5e20–2e21) incl. self-play | 0 | **~1e21** | ~1e13–5e13 per move | Superhuman Go (beats ELF final) | Narrow | 0% | 1e-3 |
| NVARC (2025) | 1.5e21–8e21 (FT + SDG + TRM + TTT) | Qwen3-4B 8.6e23 + Qwen3 teacher 4.8e24–7.1e24 + gpt-oss-120b 4.9e24 (+ Claude) | **1.1e25–1.3e25** | ~3e16–9e16 | ARC-AGI-2 24.03% (private, Kaggle) | Narrow system on a general base | ≥99.9% | ~11 |
| Dreamer 4 2B (2025) | 4e21–4e22 [A-heavy] | 0 (2.5K h human video) | **~1e22** | real time on 1 H100 | Offline diamonds 0.7%, iron pickaxe 29% | Narrow task, general-ish world model | 0% compute | 1e-2 |
| V-JEPA 2 1B (2025) | 9e21–2.5e22 | 0 | **~1e22** | — | SSv2 77.3%; zero-shot robot pick-and-place | Perception / world model (no language) | 0% | 1e-2 |
| AlphaZero chess / Go (2017) | chess 1.1–1.7e22; Go 3–6e22 | 0 | **~1e22 / ~5e22** | 800 × fwd per move | Beat Stockfish (chess) at ~5e21–7e21 | Narrow (one net per game) | 0% | 1e-2 |
| AlphaGo Zero (2017) | 3-day 2.7e22; 40-day 3.2e23 | 0 | **2.7e22 / 3.2e23** | 5.5e13 per move | Superhuman Go | Narrow | 0% | 0.03 / 0.3 |
| Phi-4 14B (Dec 2024) | 9.3e23 | GPT-4o inference 3.5e23–3.5e24; GPT-4o pretrain 3.8e25 (low-confidence) | **2e24 (inference-only) / 4e25 (full)** | 2.8e10 per token | GPQA 56.1, MATH 80.4 (beats GPT-4o) | General (STEM-skewed) | 27–79% (inference-only) / 98% (full) | 2–40 |
| DeepSeek-V3 (Dec 2024) | 3.3e24 | small internal R1-preview distillation | **3.3e24** | 7.4e10 per token | MMLU 88.5, GPQA 59.1 | General | ~0% external | 3.3 |
| DeepSeek-R1 (Jan 2025) | RL 5e22–1.6e23 | V3 base 3.3e24 (in-house) | **3.5e24** | ~7e14 per AIME problem | AIME-24 79.8, GPQA 71.5 | General | 94% (base) | 3.5 |
| R1-Distill-Qwen-1.5B / 7B / 32B (Jan 2025) | SFT 1e20 / 5e20 / 2e21 | Qwen base 1.75e23 / 8.6e23 / 3.5e24 + R1 chain 3.45e24 | **3.6e24 / 4.3e24 / 7.0e24** | 2N per token (3e9–6.5e10) | AIME-24 28.9 / 55.5 / 72.6 | General (math-skewed) | ≥99.97% | 3.6–7 |
| GPT-4 (Mar 2023) | 2.1e25 (Epoch) | — | **2.1e25** | undisclosed | MMLU ~86 † | General | — | 21 |
| Grok 4 (2025) | 5e26 (Epoch) | — | **5e26** | undisclosed | frontier-2025 | General | — | 500 |
| 2026 frontier (e.g., GPT-5.5 / Claude Opus 5 class) | 5e26–3e27 [A] | — | **5e26–3e27 [A]** | undisclosed | ARC-AGI-1 98.5% (Opus 5.5, 2026) | General | — | 500–3,000 |
| Human brain, lifetime to 30 y | 1e22–1e26 (central 1e24) | evolution prior | **~1e24** | ~1e15 FLOP/s × task time (ARC task, 2–5 min ≈ 1e17–3e17) | Human general intelligence | General | — | 1 |
| Evolution anchor | 1e41 | — | **1e41** | — | Discovered the brain's algorithm and prior | — | — | 1e17 |

- **Borrowed-compute accounting changes rankings by 1–5 OOM.** Counting own compute only, NVARC (~5e21) looks like DreamerV3 × 20, and R1-Distill-1.5B (1e20) looks like TRM. Under full attribution both sit at 4e24–1e25, i.e. **frontier-2024 scale**.
- The one "LLM-class and not borrowed" row is DeepSeek-V3/R1 at 3.3–3.5e24, about **3–4× the brain-lifetime anchor**.
- The borrowed share is **not** a small correction for data-generation inference. Inference-only borrowing is 1–80% (R1-distill data generation about 1e21; Phi-4 teacher inference about 1e24). Pretraining of the base model and teacher is what dominates.
- **Narrow vs general is the sharpest divide in the table.** Every row ≤1e22 FLOP either (i) exploits an exact simulator and verifier (Go, chess, Atari, Minecraft) or (ii) fits the target benchmark transductively (HRM, TRM, CompressARC). The only exception is the BabyLM model, which is general in modality but weak.

### Gaps
- There is no public "full-pipeline" accounting for any Gemini/GPT/Claude 2026 model, so the 2026 frontier row is an assumption range.
- I found no public estimate of GPT-4o's active parameters. The Phi-4 borrowed share therefore spans 27–79% (inference-only).
- Inference per task for frontier LLMs on ARC is known only in dollars ($0.16–$0.41/task for Opus 5.5 per earlier notes). Converting to FLOP needs undisclosed parameter counts.

---

## Q3. Re-verification of load-bearing figures and discrepancies with the earlier notes

### Takeaway
Most load-bearing numbers survive, but **five need correction or qualification**:
1. Gundlach et al.'s abstract accounts for **6,930×** (not the full 22,000×), and never states "91%". The 91% figure is a body-text claim reported by snippets with an inconsistent compute scale.
2. **Epoch's AlphaGo Zero and AlphaZero numbers now exclude self-play.** 3.4e23 is no longer Epoch's figure.
3. Densing law is **3.3 months** in the arXiv text; 3.5 months is only in secondary or later versions.
4. The R1 RL cluster was **512 H800s**, not 648, with 147K GPU-hours.
5. NVARC's synthetic-data teacher was **gpt-oss-120b plus Claude**, not GPT-4o.

Ho et al. (8 months, CI 5–14), Carlsmith (1e15; <10% above 1e21) and Cotra (1e24; 1e41) are confirmed as quoted.

### Cited Findings
- **Gundlach et al., "On the Origin of Algorithmic Progress in AI"** (Hans Gundlach, Alex Fogelson, Jayson Lynch, Ana Trisovic, Jonathan Rosenfeld, Anmol Sandhu, Neil Thompson; arXiv 2511.21622, Nov 2025). Verbatim v1 abstract [S, three independent arXiv-digest mirrors]:

  > "Algorithms have been estimated to increase AI training FLOP efficiency by a factor of 22,000 between 2012 and 2023 [Ho et al., 2024]. Running small-scale ablation experiments on key innovations from this time period, we are able to account for less than 10x of these gains. Surveying the broader literature, we estimate that additional innovations not included in our ablations account for less than 10x, yielding a total under 100x. … we conduct scaling experiments between LSTMs and Transformers, finding exponent differences in their compute-optimal scaling law while finding little scaling difference for many other innovations. … Using experimental extrapolation and literature estimates, we account for 6,930x efficiency gains over the same time period, with the scale-dependent LSTM-to-Transformer transition accounting for the majority of gains. Our results indicate that algorithmic progress for small models has been far slower than previously assumed, and that measures of algorithmic efficiency are strongly reference-dependent."

  Sources: [arxiv-ai-mailing digest](https://github.com/2shin0/arxiv-ai-mailing/blob/main/ALL/2025-11-28.md); [Awesome-arXiv-Daily-Reporter](https://github.com/CSQianDong/Awesome-arXiv-Daily-Reporter/blob/main/27-Nov-2025/AI/README.md); [arXiv 2511.21622](https://arxiv.org/abs/2511.21622)
- **Gundlach body text** [snip]:
  - LSTM→Transformer and Kaplan→Chinchilla "together account for 91% of total efficiency gains when extrapolating to the 2025 compute frontier". Small scale-invariant innovations give "<10× … less than 10% of total improvements extrapolated to the 2025 compute frontier (2 × 10²³ FLOPs)".
  - The LSTM→Transformer gain is "about 6x" at ~1e15 FLOP, "about 28x" at ~1e17, and "about a 91x gain" at the compute where the Transformer was introduced (2017).
  - Kaplan→Chinchilla is roughly equivalent at 1e20 and gives "~10x or more" at frontier scales.
  - A second snippet phrases the frontier as "(10^23 FLOPs)".
  - — [arXiv HTML 2511.21622](https://arxiv.org/html/2511.21622) (snippet)
- **Ho et al. 2024, "Algorithmic progress in language models"** (arXiv 2403.05812v1) [S, verbatim from an abstract mirror]: "Using a dataset of over 200 language model evaluations on Wikitext and Penn Treebank spanning 2012–2023, we find that the compute required to reach a set performance threshold has halved approximately every 8 months, with a 95% confidence interval of around 5 to 14 months … the increase in compute made an even larger contribution to overall performance improvements over this time period." — [cn-chat-arxiv mirror](https://github.com/qhduan/cn-chat-arxiv/blob/master/papers/24/03/2403.05812.json); [arXiv 2403.05812](https://arxiv.org/abs/2403.05812)
- **Carlsmith 2020** [snip]: "more likely than not that 10^15 FLOP/s is enough"; "unlikely (<10%) that more than 10^21 FLOP/s is required"; mechanistic "~1e13–1e17 FLOP/s". — [Coefficient Giving](https://coefficientgiving.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/)
- **Cotra 2020** [snip]: lifetime median 1e24 (1e15 FLOP/s × ~30 yr); evolution median ~1e41 at 10% weight. — [Epoch: Grokking bio-anchors](https://epoch.ai/blog/grokking-bioanchors)
- **Epoch frontier compute** [S, dataset mirrors]:
  - GPT-4 2.1e25; Grok 3 3.5e26; Grok 4 5.0e26; GPT-4.5 3.8e26; Claude 3.5 Sonnet 2.7e25 — [3a.csv](https://github.com/soeunpark98/ai-carbon-visualization/blob/main/public/data/part3/3a.csv); [models.csv](https://github.com/suveer-dhawan/ai-dispatch/blob/main/data/models.csv)
  - A secondary note attributes different values to Epoch: GPT-4.5 6.4e25, Grok-3 4.6e26, Claude 3.5 Sonnet 3.6e25, GPT-4o 3.8e25 — [xrisk-pause-game note](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/compute-requirements-trends.md)
  - GPT-5 ≈ 5e25 including RL [snip] — [Epoch substack](https://epochai.substack.com/p/why-gpt-5-used-less-training-compute)
- **Epoch "consumer GPU gap"** (15 Aug 2025) [S2, quoted verbatim in a compiled note] — [AIE-talk lag chart notes](https://github.com/Dylancouzon/AIE-talk/blob/main/research/H-lag-chart.md); original [Epoch data insight](https://epoch.ai/data-insights/consumer-gpu-model-gap)
  - "Using a single top-of-the-line gaming GPU like NVIDIA's RTX 5090 (under $2500), anyone can locally run models matching the absolute frontier of LLM performance from just 6 to 12 months ago."
  - Consumer-runnable ceilings: ≤28B parameters (RTX 4090) and ≤40B (RTX 5090).
  - Average lags by benchmark: AA Intelligence Index 6.3 months; MMLU-Pro 7.3; GPQA-Diamond 7.4; LM Arena Elo 12.4.
  - Elo slopes: +125 per year for consumer-runnable open models against +80 per year for the frontier.
  - Open–closed ECI lag: 3.5 months in the Oct-2025 reading (Jan 2023–Oct 2025); **about 4 months (8 ECI points) for Jan–May 2026**.
- **Densing law** (arXiv 2412.04315v2, 6 Dec 2024) [S, full text]:
  - Body: "LLMs doubles approximately every 3.3 months" (A ≈ 0.0073). Abstract: "approximately every three months".
  - A worked example shows GPT-3.5-level inference price falling 266.7× (secondary compilation: cost halves ~every 2.6 months).
  - "With the release of ChatGPT, the growth rate of LLM density increased by 50%."
  - — [Densing law text mirror](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/densing-law.txt); [arXiv 2412.04315](https://arxiv.org/abs/2412.04315)
- **DeepSeek-R1 cost**: 64 × 8 H800s; R1-Zero 198 h (101K GPU-h); R1 80 h (41K); SFT data 5K; total 147K GPU-h = $294K [S]. — [R1 v2 text](https://github.com/bojieli/ai-infra-book/blob/main/references/token-cost/2026-09-07/text/deepseek-r1.txt)
- **NVARC teacher models**: claude-opus-4-20250514, claude-sonnet-4-5-20250929 and gpt-oss-120b, with the majority from gpt-oss-120b [S]. — [NVARC paper](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf)

### Inferences
Discrepancy log against the first-round notes:

| # | Earlier note said | Re-verified | Verdict |
|---|---|---|---|
| 1 | Gundlach: small-scale gains <100× ✔; "91% from Transformer + Chinchilla at the 2025 frontier" | The abstract confirms <10× (ablations) + <10× (literature) → <100×. It **does not mention 91%**. It says the paper accounts for only **6,930× of the 22,000×**, with LSTM→Transformer "the majority". The 91% is a body-text claim (snippet) with an inconsistent scale label ("2025 frontier (2×10²³ FLOPs)", while the actual 2025 frontier is about 5e26 per Epoch). Separately, a "~91×" LSTM→Transformer gain at 2017 scale exists. The two "91"s may be conflated in secondary summaries. | **Qualify**. Quote "<100× at small scale" and "6,930× accounted, majority LSTM→Transformer" as verified. Quote "91%" only as body-text (unverified), and note that the 22,000× is Ho et al.'s figure, only about 1/3 of which Gundlach can reconstruct. |
| 2 | Ho et al.: 8-month halving, CI 5–14 | Verbatim confirmed; the dataset is over 200 evaluations, 2012–2023. The Shapley split (60–95% compute vs 5–40% algorithms) is not in the abstract. | **Confirmed**; Shapley split unverified. |
| 3 | Carlsmith: 1e15 central, 1e13–1e17 mechanistic, span to 1e21 | Confirmed; add "<10% that >1e21 is needed". | **Confirmed**. |
| 4 | Cotra: 1e24 lifetime, 1e41 evolution | Confirmed; 10% weight on evolution. | **Confirmed**. |
| 5 | Epoch: GPT-4 ≈ 2e25 | 2.1e25. | **Confirmed**. |
| 6 | Grok 4 "~5e26 (unverified recollection)" | 5.0e26 in the Epoch export mirror. | **Now verified** (mirror). |
| 7 | Grok 3 first >1e26 | Epoch mirror 3.5e26; a secondary note says 4.6e26. | Confirmed >1e26; the exact value conflicts. |
| 8 | GPT-5 < GPT-4.5 | Confirmed qualitatively; ~5e25 is snippet-only. GPT-4.5 is 3.8e26 in the mirror vs 6.4e25 in a secondary note. | **Conflict** on GPT-4.5's magnitude. |
| 9 | 2026 frontier 1e26–1e27 (consultancy source) | Epoch has **no estimates** for most 2026 closed models (4/51 OpenAI, 1/16 Anthropic since 2025). The largest documented run is still Grok 4 at 5e26. | Keep as assumption only. |
| 10 | AlphaGo Zero "Epoch ~3.4e23" | Epoch now lists **6.5e20** (training only, self-play excluded explicitly). My recomputation including self-play is 3.2e23. | **Correct the attribution**: 3.4e23 is the old Epoch number and the correct full-pipeline number. Epoch's current number excludes 99.8% of the cost. The same applies to AlphaZero (1.06e20), KataGo (2.3e19) and MuZero (4.8e19). |
| 11 | KataGo ~27 V100 × 19 d "from memory" | 19 days, max 28 (average 26–27) V100s, 4.2M games, 241M samples. | **Confirmed**. |
| 12 | Densing law 3.5 months (Nature MI) | arXiv body 3.3 months; abstract "about three months". | **Minor discrepancy**; cite 3.3 (arXiv), with ~3.5 in later or secondary versions. |
| 13 | Consumer GPU gap 6–12 months, ~9-month lag | Confirmed 6–12 months; per-benchmark lags 6.3–12.4 months; ≤28B/≤40B ceilings. The "~9-month" figure was not found in the quotes (likely a midpoint or an X post). | **Confirmed**. |
| 14 | R1 RL: 512 vs 648 GPUs, $294K | 512 H800s (64 × 8); 101K + 5K + 41K = 147K GPU-h; $294K. | **Resolved**: 512. |
| 15 | NVARC seeds "partly with GPT-4o" (NVIDIA blog snippet) | The paper lists Opus 4, Sonnet 4.5 and gpt-oss-120b (majority). GPT-4o is not mentioned. | **Correct** to gpt-oss-120b + Claude. |
| 16 | TRM "~4 H100 × ~2 days, <$500" (theory notes) vs "~3 days" (small-models notes) | The README says ~3 days on 4 H100 = 288 H100-h, i.e. ~$600–2,300 at $2–8/h. <$500 requires under $1.7/H100-h. | **Correct** to ~3 days, ~$0.6–2.3K. |
| 17 | Phi-4 "~400B synthetic tokens" | The pretraining table shows 290B synthetic + 290B web-rewrite unique tokens (40% + 15% of 9.8T), with 13.8 epochs on synthetic. The 400B figure likely refers to the total synthetic corpus before selection. | **Qualify**. |
| 18 | V-JEPA 2 GPU-hours "not stated" | The paper: naive 64f×384² training "roughly 60 GPU-years"; progressive resolution gives 8.4× less, i.e. ~7 GPU-years. Epoch estimates 9.06e21 FLOP. | **Filled**. |
| 19 | DreamerV3: V100 vs A100 conflict; "30M steps / 17 days" | Nature version: single A100, 100M steps for Minecraft, "1 GPU for 9 days". | **Resolved** to A100. The 30M/17-day figure is probably from the 2023 preprint (unverified). |
| 20 | EfficientZero 194.3% / 109.0% | Confirmed in the text; the paper's Table 1 text says 1.904 / 1.160. | Minor internal inconsistency in the paper. |

### Gaps
- I could not read Gundlach et al.'s full body text, so the "91%" sentence and its compute-scale label (2e23 vs 1e23 vs a 2025 frontier of about 5e26) remain unverified. The exact v2 abstract, if one exists, was also not checked.
- The GPT-5 FLOP estimate and Epoch's GPT-4.5 figure (3.8e26 vs 6.4e25) could not be resolved from primary Epoch pages, which are blocked.
- The Nature Machine Intelligence version of the densing law (3.5 months?) was not accessible.
- Epoch's newer software-progress revisions from 2025–2026 were not accessible.

---

## Q4. Derived insights: lowest full-pipeline compute per capability tier, the trend, and where the brain anchor sits

### Takeaway
The lowest demonstrated full-pipeline compute by tier (see the table below):
- **Superhuman narrow game**: about **1e21** (KataGo 2019), 1e-3 of the brain anchor.
- **Human-level Atari-100k**: about **1e18 per game** (EfficientZero 2021), roughly the brain's FLOP over the same 2 h of play.
- **ARC-AGI-1 at about 40%**: about **1e20** (TRM/HRM 2025), but only via transductive, benchmark-specific training.
- **ARC-AGI-1 at ≥80%**: only frontier-LLM pipelines, **≥5e26** full-pipeline.
- **GPT-3.5-level language**: about **1e23** own compute without a teacher (DCLM-7B 2024). With a teacher it is about 7.5e22 own.
- **GPT-4-level language**: about **3.3e24** without borrowing (DeepSeek-V3). About 9e23 own compute with borrowing (Phi-4, Qwen3-4B), but ≥5e24–4e25 full-pipeline.

The brain-lifetime anchor (about 1e24; range 1e22–1e26) sits **above every narrow-superhuman result by 2–6 OOM, and at or below every general-language result**. Full-pipeline compute for general capability has fallen about 6× in 21 months (GPT-4 to V3). "Own" compute has fallen about 20× in the same period, but only by importing teachers.

### Cited Findings
- KataGo surpassed ELF's final (superhuman) model in 19 days on under 30 GPUs, a 50× reduction [S]. — [KataGo paper](https://github.com/FoAKTEE/az/blob/main/ref-paper/arxiv-1902.10565/src/Accelerating_Self_Play_Learning_In_Go_2020.tex)
- EfficientZero reached super-human mean and median on Atari 100k with 2 h of game experience [S]. — [EfficientZero text](https://github.com/qpwo/rules/blob/main/writings/arxiv.org__pdf__2111.00210.pdf.txt)
- DCLM-7B reached 64% MMLU on 2.6T tokens, near Llama 3 8B's 66% with up to 6.6× less compute (earlier-round notes). — [DataComp-LM, arXiv 2406.11794](https://arxiv.org/abs/2406.11794)
- AI Index 2025: the smallest model scoring over 60% on MMLU shrank from PaLM 540B (2022) to Phi-3-mini 3.8B (2024), a 142× parameter reduction (earlier-round notes). Epoch puts phi-3-mini at 7.52e22 FLOP [S]. — [AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report); [Epoch mirror](https://github.com/soeunpark98/ai-carbon-visualization/blob/main/public/data/part3/3a.csv)
- ARC-AGI-1 results (earlier-round notes, from ARC Prize):
  - Berman's Grok-4-based evolutionary program search: 79.6% ($8.42/task);
  - o3-preview (low): 75.7% (Dec 2024);
  - Claude Opus 5.5: 98.5% at $0.16/task (2026);
  - Greenblatt (GPT-4o, about 8k programs/task): about 42% semi-private (2024);
  - ARChitects 2024: 53.5% private (Kaggle, small LLM).
  - — [small-models notes, ARC table](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- DeepSeek-V3 MMLU 88.5 / GPQA 59.1 [S] — [README](https://github.com/deepseek-ai/DeepSeek-V3). GPT-4 is 2.1e25 [S Epoch] with MMLU 86.4 (†; GPT-4 Technical Report, not re-fetched) — [arXiv 2303.08774](https://arxiv.org/abs/2303.08774)
- Qwen3-4B "can rival" Qwen2.5-72B-Instruct (vendor claim, earlier-round notes). — [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)

### Inferences
**Lowest full-pipeline compute by capability tier** [E, from Q1–Q2]:

| Tier | Lowest full-pipeline (who, when) | Earlier reference point(s) | Trend | Position vs brain anchor (1e24) |
|---|---|---|---|---|
| Superhuman narrow game (Go) | **~1e21**: KataGo, 2019 (0% borrowed) | AGZ 3-day 2.7e22 (2017); ELF ~9e22 (2018); AGZ 40-day 3.2e23 | ~25× in ~1.5 yr (2017 to 2019); KataGo's author suggests about 1e21 on one consumer GPU is possible | 1e-3 × (1e-5 to 0.1 across the Carlsmith range) |
| Superhuman chess | ~5e21–7e21: AlphaZero at the point it passed Stockfish (2017) | — | — | ~5e-3 × |
| Human-level Atari (100k, 26 games) | **~1e18 per game, ~3e19 suite**: EfficientZero, 2021 | DQN reached a similar mean with 500× more data (compute not sourced) | Sample efficiency 500×; compute trend not measured | Per game ≈ 0.05–0.3× the brain's 7.2e18 FLOP over the same 2 h of play |
| ARC-AGI-1 ≈ 20% | **≤1.4e19**: CompressARC, 2025 (no pretraining) | — | — | ≤1e-5 × |
| ARC-AGI-1 ≈ 40% (verified) | **~1e20–3e20**: TRM (40%) / HRM (32%), 2025, transductive | Greenblatt ~42% on GPT-4o (~4e25 full, 2024) | **~5 OOM drop in one year**, but by switching to a narrow per-benchmark learner | 1e-4 × |
| ARC-AGI-1 ≈ 50–65% | ARChitects 53.5% (2024; small LLM base, borrowed ~1e23–1e24 [A]); LoopViT 65.8% / URM 53.8% (self-reported, unverified; ~1e20 scale [A]) | — | — | ~1e-4 to 1 × |
| ARC-AGI-1 ≥ 80% | **≥5e26**: Berman on Grok 4 (79.6%, 2025); Opus 5.5 98.5% (2026, undisclosed; likely ≥1e26) | o3-preview 75.7% (Dec 2024, undisclosed) | Inference $/task fell 2–4 OOM (earlier notes); **full-pipeline did not fall**, it tracks frontier pretraining | ≥500 × |
| GPT-3.5-level language (MMLU 60–70) | **~1e23** own, no teacher: DCLM-7B, 6 × 7e9 × 2.6e12 = 1.1e23 (2024). **7.5e22** own: phi-3-mini (2024, synthetic teacher data, full pipeline much higher) | GPT-3 175B 3.1e23 (2020, MMLU ~44 †) | ~3× below GPT-3's compute (1.1e23 vs 3.1e23) at a much higher score | 0.1 × |
| GPT-4-level language (MMLU ≥85) | **3.3e24**: DeepSeek-V3, Dec 2024 (no external borrowing) | GPT-4 2.1e25 (Mar 2023) | **~6× in 21 months** full-pipeline (≈2.8×/yr, matching Ho et al.'s 8-month halving). Own compute: Phi-4 9.3e23 (Dec 2024), Qwen3-4B 8.6e23 (Apr 2025, vendor "rivals 72B"), ~20× lower; full-pipeline ≥5e24 | 3.3 × (V3); ~1 × own-only (Phi-4) |
| Broad human-level generality (not reached at low compute) | — (frontier ≥5e26; no ≤1e23 general system) | — | — | Brain does it at ~1e24 |

1. **The brain anchor is not "low" relative to narrow AI.** Narrow superhuman play (Go at 1e21), human-level Atari (1e18 per game) and 40% ARC-AGI-1 (1e20) all sit **3–6 OOM below** the lifetime anchor. In the Atari case, compute over the same 2-hour experience window is at parity with the brain or below it. Narrow compute efficiency is therefore already better than the brain's. The brain's advantage lies elsewhere:
   - **generality** from the same budget;
   - **priors** from about 1e41 FLOP of evolution;
   - **transfer**: a 13-year-old reaches language competence on about 1e8 words with about 4e23 FLOP.
2. **The BabyLM contrast is the cleanest brain-vs-AI experiment.** At the same word budget (≤1e9 words of exposure, vs about 1e8 words for a child), the winning model uses **1.5e18 FLOP**, about **3e-6 of the brain's 4.1e23 FLOP to age 13**. It lands far below child and LLM competence. The brain spends about **4e15 FLOP per word heard** (4.1e23 / 1e8, though that compute also serves vision, motor control and so on). A 126M model spends 7.6e8 FLOP per token and DeepSeek-V3 2.2e11. Nobody has tested a "brain-scale compute on child-scale data" regime, and standard 6ND scaling cannot absorb it: 1e23 FLOP on 1.3e9 token-exposures would need about 1e13 parameters. **This is the largest untested cell in the ledger.** It is where a sample-efficient learner, such as world-model or JEPA-style with heavy per-token computation or recurrence, would have to prove itself.
3. **General language capability has a hard floor near 1e24 today, and borrowing hides but does not lower it.** The cheapest non-borrowed GPT-4-class model (V3/R1 at 3.3–3.5e24) is within **3–4×** of the lifetime anchor. Every cheaper-looking general model (Phi-4, R1-distills, Qwen3-4B, NVARC's base) recovers **≥4e24** under full attribution. The full-pipeline trend for general capability is about 3× per year (GPT-4 to V3: 2.1e25 to 3.3e24 in 21 months). At that rate, full-pipeline GPT-4-level at ≤1e23 (brain-to-age-4 scale) would take about **3–4 more years** [E: 33× at 2.8×/yr ≈ 3.4 yr]. Gundlach et al. warn that the scale-dependent part of past gains will not repeat at small scale, so this extrapolation is optimistic.
4. **The "catch-up" is inference-side and teacher-side, not pipeline-side.** Inference cost per task fell 2–4 OOM for ARC-AGI-1 and the consumer-GPU lag is 6–12 months. Yet full-pipeline compute for the *first* system at each general tier tracks frontier pretraining: ≥5e26 for ARC-AGI-1 ≥80%, and the 2026 frontier at an assumed 5e26–3e27. Evolution-to-brain is the natural analogue: discovery cost (1e41) vastly exceeds per-instance cost (1e24).
5. **Where the brain anchor sits overall.** On the ledger's log axis (1e18 to 1e41), the lifetime anchor at 1e24 lies:
   - ~6 OOM above the cheapest narrow human-level result (EfficientZero per game);
   - ~3 OOM above superhuman Go (KataGo);
   - ~4 OOM above 40% ARC-AGI-1 (TRM);
   - about equal to or 3× below the cheapest non-borrowed GPT-4-class model;
   - ~20× below GPT-4 (2.1e25) and ~500× below Grok 4 (5e26);
   - ~17 OOM below the evolution anchor.

   On inference, one H100 (~1e15 peak) runs Dreamer 4's world model in real time and roughly equals Carlsmith's central brain estimate. AlphaGo Zero's 5.5e13 FLOP per move is ~550× less than a professional's 30 s of thought.

### Gaps
- No verified low-compute (≤1e23 full-pipeline) system reaches ARC-AGI-1 ≥80% or GPT-4-class breadth, so those tiers have no "lowest" point below frontier scale.
- The DQN/Rainbow/BBF compute for human-level Atari was not sourced, so the Atari compute trend (as opposed to the data trend) is unmeasured here.
- The ARChitects 2024 base model and its pretraining FLOP were not verified, so the 50–65% ARC-AGI-1 tier's full-pipeline cost is an assumption.
- There are no public FLOP numbers for the 2026 frontier (GPT-5.5, Claude Opus 5 / Fable 5, Gemini 3.x), so the ≥80% ARC-AGI-1 tier's full-pipeline cost is bounded only by the Grok 4 proxy.
