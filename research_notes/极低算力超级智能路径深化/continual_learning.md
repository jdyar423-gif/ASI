# Continual / Lifelong Learning as the Core Mechanism of Low-Compute ASI: State of the Art to September 2026

Scope: this file goes deeper than the first-round notes (`skeptics_and_forecasts.md`, `small_models_program_synthesis.md`, report §"排名第一" and "七个信号"). It does not repeat the headline figures already recorded there: Voyager 3.3×/15.3×, Stitch 1000–10000×, LILO +33/+20/+2, Pang's 50× fewer LLM calls, the basic ARC TTT results, and the one-line description of Nested Learning.

Evidence tags:
- **[PR]**: peer-reviewed venue.
- **[PP]**: arXiv preprint, not yet peer-reviewed.
- **[SR]**: self-reported by a company or blog, not independently checked.
- **[REPL]**: independently re-implemented or replicated.
- **(snippet)**: read only through a search-engine snippet. arxiv.org, openreview.net, nature.com, cursor.com, keenagi.com, oaklab.ai, heise.de, betakit.com, the-decoder.com and news.ycombinator.com were all blocked for direct fetch in this session. GitHub READMEs were read directly.

All dates are publication or announcement dates.

---

## Q1. Loss of plasticity and catastrophic forgetting: how bad at scale, and is it solved?

### Takeaway
Loss of plasticity is real and reproducible. On the canonical benchmark, without intervention, accuracy falls from 89% to 77% over 2,000 tasks. It is "solved" only in small networks, by injecting non-gradient diversity (continual backprop) or by combining layer norm with weight decay. The first scaling study on GPT-style LMs (June 2026, 5M–314M non-embedding parameters) finds that plasticity loss still occurs, and that its onset grows only **sublinearly** with model size. Scale delays the problem but does not cure it. No method has been shown to keep plasticity and retention together in a >1B-parameter network over a long non-stationary stream.

### Cited Findings
**Canonical result (peer-reviewed, reproducible code):**
- **Dohare et al., Nature 632:768–774 (Aug 2024) [PR].** Standard deep-learning methods "gradually lose plasticity in continual-learning settings until they learn no better than a shallow network."
  - Plasticity "is maintained indefinitely only by algorithms that continually inject diversity into the network," such as continual backprop. Continual backprop randomly reinitializes a small fraction of less-used units.
  - The authors conclude: "methods based on gradient descent are not enough—that sustained deep learning requires a random, non-gradient component to maintain variability and plasticity."
  - The experiments cover task-incremental ImageNet, class-incremental CIFAR and PPO RL. The code runs on a standard computer with 8 GB+ RAM.
  - Sources: [repo README](https://github.com/shibhansh/loss-of-plasticity); [Nature](https://www.nature.com/articles/s41586-024-07711-7).
- **Magnitude on Continual ImageNet (Dohare et al.):** binary-classification accuracy dropped from **89% on an early task to 77% on the 2,000th task**, "about the level of a linear network." This is a 12-point drop. — [TechXplore](https://techxplore.com/news/2024-08-team-solution-ai-problem.html) (snippet)
- **Follow-up, G-CBP (Springer *Evolving Systems*, 2026) [PR].** On the 1,000-task Continual ImageNet, gradient-coupled continual backprop raises average accuracy from **77.97% to 79.63%** over fixed-rate CBP (3 learning-rate regimes, 30 runs). The gain is modest, and the absolute level stays far below the ~89% seen on early tasks. — [Springer](https://link.springer.com/article/10.1007/s12530-026-09873-3) (snippet)

**Mechanisms and remedies:**
- **Lyle et al., "Disentangling the Causes of Plasticity Loss" (CoLLAs 2024 / PMLR v274) [PR].**
  - Plasticity loss decomposes into several independent mechanisms. Intervening on several at once gives "highly robust learning algorithms."
  - **Layer norm plus weight decay** is "highly effective at maintaining plasticity" on synthetic non-stationary tasks and on ALE RL.
  - Taken alone, shrink-and-perturb, weight decay, or resetting only the optimizer state "do not improve plasticity." Layer norm does help.
  - Sources: [arXiv 2402.18762](https://arxiv.org/abs/2402.18762); [PMLR](https://proceedings.mlr.press/v274/lyle25a.html) (snippet)
- **Remedy families, from the 2024–2026 literature (snippets):**
  - Periodic resets of dormant neurons: shrink-and-perturb, CBP, ReDo. — [Lyle et al.](https://arxiv.org/pdf/2402.18762)
  - Plasticity injection. — same source.
  - Activation-function fixes: CReLU, smooth-leaky activations, deep Fourier features. — [Plastic Learning with Deep Fourier Features](https://arxiv.org/pdf/2410.20634)
  - Churn reduction to prevent rank collapse. — [OpenReview EkoFXfSauv](https://openreview.net/forum?id=EkoFXfSauv)
  - Proposed mechanisms: spectral or Hessian-rank collapse. — [arXiv 2509.22335](https://arxiv.org/pdf/2509.22335)
  - A standardized benchmark, "Plasticine" (Apr 2025). — [arXiv 2504.17490](https://arxiv.org/pdf/2504.17490)
  - A dedicated survey (Nov 2024). — [arXiv 2411.04832](https://arxiv.org/pdf/2411.04832)

**At LLM scale (the key 2026 result):**
- **Hernandez-Garcia et al., "Can Scale Save Us From Plasticity Loss in Large Language Models?" (June 2026) [PP].** J. F. Hernandez-Garcia is a co-author of the Nature paper.
  - GPT-style models with >100M non-embedding parameters "can lose the ability to efficiently adapt to new data even when trained on real, large-scale natural-language datasets."
  - Plasticity loss appears across **5M to 314M non-embedding parameters**.
  - Its onset "follows a predictable scaling law, growing sublinearly with model size." Scaling "will delay plasticity loss for a substantial number of tokens, but cannot fully defeat it."
  - Source: [arXiv 2606.24752](https://arxiv.org/abs/2606.24752) (snippet)

**Theory (2026):**
- **"Predicting Plasticity in Deep Continual Learning: A Theoretical Perspective" (May 2026) [PP].**
  - Counterexamples show that widely used diagnostics (representation rank, NTK rank) "can fail to predict the loss of trainability."
  - The paper proposes a new metric, **"optimization readiness"** (gradient strength × gradient reliability), which provably lower-bounds one-step optimization gain.
  - The metric ranks checkpoints by trainability better than earlier diagnostics, but it is validated only on toy settings (Slowly-Changing Regression, Permuted MNIST).
  - Source: [arXiv 2605.09044](https://arxiv.org/abs/2605.09044) (snippet)

**A mitigating 2026 finding:**
- **Liu & Mou, "Do Neural Networks Lose Plasticity in a Gradually Changing World?" (Feb 2026) [PP].**
  - Loss of plasticity "is an artifact of abrupt task changes in the environment and can be largely mitigated if the world changes gradually."
  - Under interpolated or gradual shift, networks "preserve plasticity magnitudes longer."
  - Source: [arXiv 2602.09234](https://arxiv.org/html/2602.09234v1) (snippet)

**Catastrophic forgetting in LLM fine-tuning (magnitudes):**
- Sequentially fine-tuning on new facts cuts held-out NaturalQuestions F1 by **89% with full fine-tuning** and **71% with LoRA**, at matched new-knowledge acquisition. — [Lin et al., arXiv 2510.15103](https://arxiv.org/abs/2510.15103) (snippet; details in Q4)
- On-policy RL forgets much less than SFT at similar new-task performance. The amount of forgetting is predicted by the forward KL between the fine-tuned and base policy, measured on the new task. — [RL's Razor, ICLR 2026](https://mlanthology.org/iclr/2026/shenfeld2026iclr-rl/) [PR] (snippet)

### Inferences
- Two different failures are often conflated.
  - **Forgetting** (loss of old skills) is largely manageable at LLM scale with sparse or KL-constrained updates (Q4).
  - **Plasticity loss** (loss of the *ability to learn*) is the more fundamental threat to a lifelong learner. Its only robust fixes (CBP, LN+WD, resets) are validated on networks orders of magnitude smaller than frontier models and on streams of ≤ a few thousand tasks.
- The sublinear-scaling result cuts against the low-compute thesis in a specific way. A compact agent (≤1B parameters) that must learn for a "lifetime" of 10¹²+ tokens or frames is in exactly the regime where plasticity loss bites earliest. The compact learner therefore needs an explicit diversity-injection or normalization mechanism; it cannot rely on over-parameterization.
- Liu & Mou suggest that real-world streams, which change gradually, may be friendlier than benchmark streams with abrupt task switches. That would make benchmark results pessimistic. It is one preprint and is not yet replicated.
- Status: **not solved at scale.** Solved in small networks, as a proof of concept.

### Gaps
- No study reports plasticity-loss onset for ≥1B-parameter models, or for any model beyond ~10¹¹ tokens of continual training.
- I could not retrieve the exact token counts or loss curves in Hernandez-Garcia et al. 2026, only the snippet summary.
- No head-to-head of CBP versus LN+WD versus resets at LLM scale was found.
- I could not verify whether Dohare's CBP results have been reproduced by an independent group at larger scale. G-CBP is an extension, not a replication.

---

## Q2. Streaming / online deep RL: sample and compute efficiency without replay buffers

### Takeaway
Elsayed, Vasan & Mahmood (Oct 2024) showed that "stream-x" algorithms can learn stably one sample at a time, with no replay buffer, no target network and no batches. They reach roughly **batch-RL sample efficiency**: Stream Q(λ) matches DQN on MinAtar, and Stream AC beats PPO and SAC on several MuJoCo tasks. The implementation is about 150 lines. The 2026 successors add online representation learning (SPR), recurrence (RTRL) and batch-to-streaming hybrids. Streaming delivers constant memory and cheap per-step compute, and it removes the "stream barrier." It does **not** yet deliver better-than-batch sample efficiency, which is the property a low-compute ASI would need.

### Cited Findings
- **Elsayed, Vasan & Mahmood, "Streaming Deep Reinforcement Learning Finally Works" (arXiv Oct 2024; NeurIPS 2024 workshop version) [PR-workshop / PP].**
  - Defines the **"stream barrier"**: earlier streaming deep RL was unstable and failed to learn. Batch methods run in streaming mode (PPO1, SAC1, DQN1: buffer and batch size of 1) "struggle."
  - Stream-x (Stream AC(λ), Stream Q(λ), Stream SARSA(λ), stream TD) uses layer norm before each activation, eligibility traces, observation and reward scaling, and the **ObGD** (overshooting-bounded gradient descent) optimizer.
  - It uses "a single set of hyperparameters" and should "work without any hyperparameter tuning."
  - The minimal implementation is **around 150 lines of code**.
  - Sources: [repo README](https://github.com/mohmdelsayed/streaming-drl); [arXiv 2410.14606](https://arxiv.org/abs/2410.14606); [ML Anthology, NeurIPS-W 2024](https://mlanthology.org/neuripsw/2024/elsayed2024neuripsw-deep/)
- **Quantitative claims (snippet from the paper):**
  - Stream AC with λ=0.8 "outperforms PPO and SAC in all environments" tested and is "more sample efficient than PPO in Humanoid-v4, HumanoidStandup-v4, and Ant-v4."
  - Stream Q(0.8) "is as sample efficient as DQN in MinAtar."
  - Stream-x gives "the best model-free performance in DM Control Dog environments."
  - Source: [arXiv 2410.14606](https://arxiv.org/pdf/2410.14606) (snippet)
- **Successors (2025–2026):**
  - **QRC(λ)** (Elelimy et al., 2025) is another streaming method. — [arXiv 2605.24709 related-work](https://arxiv.org/pdf/2605.24709) (snippet)
  - **"Squeezing More from the Stream" (Nilaksh, Clavaud, Reymond, Rivest, Chandar; Feb 2026) [PP].**
    - The problem it addresses: streaming agents are "notoriously sample-inefficient" because "value-based losses alone struggle to extract meaningful representations from transient data."
    - The fix: add Self-Predictive Representations, with orthogonal gradient updates to handle correlated samples.
    - It "systematically outperforms existing streaming baselines" on Atari, MinAtar and Octax, and trains "on just a few CPU cores."
    - Sources: [arXiv 2602.09396](https://arxiv.org/pdf/2602.09396) (snippet); [code](https://github.com/chandar-lab/stream-rep-rl)
  - **Streaming RL under partial observability with RTRL (May 2026) [PP]** extends stream-x to recurrent agents "while preserving the no-replay, constant-memory streaming property." — [arXiv 2605.24709](https://arxiv.org/pdf/2605.24709) (snippet)
  - **"Towards Batch-to-Streaming Deep RL for Continuous Control" (Mar 2026) [PP]** concludes that streaming DRL is "most valuable not as a replacement for batch methods, but as a complement to them in practical deployment scenarios." — [arXiv 2603.08588](https://arxiv.org/html/2603.08588) (snippet)
- **Real-world use, Keen Technologies' physical Atari (details in Q3).**
  - A game-independent agent "learns under real-time constraints to reliably surpass standard benchmark performance within **five hours (1 million frames)**" on multiple games.
  - The setup is a real Atari 2600+, a 60 fps camera and servo joystick actuation, running on "a gaming laptop or workstation."
  - Source: [Keen physical_atari README](https://github.com/Keen-Technologies/physical_atari)
- **Step-size adaptation**, the Alberta line's complement to streaming.
  - Adam and RMSProp cannot tell a noisy feature (which needs a small step) from a drifting one (which needs a large step).
  - IDBD (Sutton 1992) and its successors optimize per-weight step sizes online, and "on simple problems IDBD is able to consistently improve step-size vectors, where RMSProp and Adam do not."
  - Source: [Degris et al., "Step-size Optimization for Continual Learning," arXiv 2401.17401](https://arxiv.org/abs/2401.17401) (snippet)

### Inferences
- Streaming RL's compute profile suits a compact lifelong agent: O(1) memory, one update per step, CPU-trainable, a single hyperparameter set. Measured sample efficiency, however, is **parity** with batch methods (roughly 1×) at best. It is not the 10–100× advantage that model-based methods (Dreamer, EfficientZero) show with replay.
- The 2026 successors all try to bring back what replay provided: representation learning (SPR), memory (RTRL) and hybrids. The field has not found a way for streaming learners to extract more per sample than batch learners.
- For the ASI thesis, streaming is necessary infrastructure, since experience cannot all be stored forever. It is not a source of compounding.

### Gaps
- No published FLOPs-per-frame or wall-clock comparison of stream-x against DQN or PPO at equal performance was retrievable.
- No streaming-RL result on long *sequences* of different tasks (continual RL across games) was found. All stream-x results are single-task.
- Atari-57 median human-normalized score for stream-x was not retrievable.

---

## Q3. Alberta Plan, OaK, Keen Technologies and Oak Lab: what is specified or demonstrated as of 2026?

### Takeaway
OaK is a **specification without an integrated demonstration**. Sutton presented it at RLC 2025 (Aug 2025) and as a NeurIPS 2025 invited talk (Dec 2025): all components learn continually, each weight has a meta-learned step size, and abstractions grow through an FC-STOMP loop. In 2026 the program split.
- **Keen Technologies (Carmack)** published a real-time **physical Atari** platform. Its agent surpasses benchmark performance in 1M frames (about 5 hours) on several games. It has no public continual or multi-game results.
- **Sutton and Khurram Javed left Keen and founded Oak Lab** (Toronto, announced about July 2026). Its stated goal is "a trillion-parameter agent that learns and plans in real-time with 20 watts." Its first public technical artifact is **NetworkIDBD** on a NoisyMNIST stream: a 10,000-unit, single-hidden-layer network trained on 100k online samples. Oak Lab has not published the equations or code.

### Cited Findings
**OaK specification (2025):**
- OaK is "a model-based RL architecture with three special features":
  1. "all of its components learn continually";
  2. "each learned weight has a dedicated step-size parameter that is meta-learned using online cross-validation";
  3. "abstractions in state and time are continually created in a five-step progression: Feature Construction, posing a SubTask based on the feature, learning an Option to solve the subtask, learning a Model of the option, and Planning using the option's model (FC-STOMP)."
  - Sources: [Amii video page, RLC 2025](https://www.amii.ca/videos/oak-architecture-rich-sutton-rlc2025); [NeurIPS 2025 invited talk](https://neurips.cc/virtual/2025/invited-talk/109601) (snippets)
- Sutton framed the RLC-2025 and AGI-2025 talk as a response to an AI industry that "to an extent… has lost its way." — [Sutton on X, Aug 2025](https://x.com/RichardSSutton/status/1957501548214513897) (snippet)

**Keen Technologies, Physical Atari (read directly from GitHub):**
- Paper: "Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots," by **Khurram Javed, Joseph Modayil, Gloria Kennickell, Richard S. Sutton, John Carmack**. It is an RLC paper; the code repo was updated June 2026. — [physical-atari-rlc](https://github.com/Keen-Technologies/physical-atari-rlc)
- The platform has a Raspberry Pi "Devbox" running `PhysicalALE`, a camera, and a "Robotroller" with 3 Dynamixel servos at 1 Mbps. — same repo.
- Three stated contributions: a physical platform; "a game-independent RL algorithm… [that] learns under real-time constraints to reliably surpass standard benchmark performance within five hours (1 million frames) on multiple Atari games"; and evidence on "the limitations of our simulators." — [physical_atari README](https://github.com/Keen-Technologies/physical_atari)
- The recommended games are Ms. Pac-Man, Centipede, Up 'n Down and Krull. Q*Bert, Battle Zone, Atlantis and Defender are "less tested."
- Score and lives detection from camera video is "the most brittle part" and needs per-game supervised classifiers.
- "Both trained policies and reinforcement learning algorithms can degrade significantly when exposed to these real-world conditions, even if they perform well in simulation." — same README
- The repo contains no multi-game sequential or continual results. The Keen GitHub org lists only these two repos. — [Keen-Technologies org](https://github.com/Keen-Technologies)

**Carmack, Upper Bound talk notes (May 2025):**
- "Sequential [multi-task learning] is much harder than parallel."
- When a trained agent moves to a second game, it "quickly forgets the first game."
- He cites **GATO showing negative transfer**: learning a new game became harder after learning others in parallel. The immediate challenge is "to just avoid getting worse."
- He also notes that "watching state-of-the-art agents learn new games, they play poorly," whereas after weeks mastering dozens of games an agent "should pick up new games more effectively."
- Sources: [slides/notes mirror](https://www.slideshare.net/slideshow/john-carmack-s-notes-from-his-upper-bound-2025-talk/279574708); [Carmack on X](https://x.com/ID_AA_Carmack/status/1925710474366034326) (snippets)

**Oak Lab (2026):**
- Sutton "left Keen Technologies alongside Khurram Javed to launch Oak Lab," based in Toronto. It has not disclosed funding and describes itself as "small, focused team." Coverage dated around 15 July 2026. — [TechTimes, 15 Jul 2026](https://www.techtimes.com/articles/320598/20260715/turing-award-winner-sutton-launches-oak-lab-calls-current-ai-fundamentally-broken.htm); [The Next Web](https://thenextweb.com/news/richard-sutton-oak-lab-reinforcement-learning); [BetaKit](https://betakit.com/ai-pioneer-richard-sutton-founds-new-research-lab/) (snippets)
- Stated goal: **"a trillion-parameter agent that learns and plans in real-time with 20 watts of power."** Sutton calls current deep-learning methods "weak and inefficient," needing "not more tweaks, but fundamentally new ideas and a thorough reworking." — [Oak Lab mission page](https://oaklab.ai/mission); [KuCoin news flash](https://www.kucoin.com/news/flash/69-year-old-reinforcement-learning-pioneer-richard-sutton-launches-oak-lab-to-build-human-level-ai) (snippets)
- Sutton and Javed "promised to publish technical papers in the coming weeks and months." Sutton published a new continual-learning algorithm, **NetworkIDBD**. — [MLQ](https://mlq.ai/news/turing-award-winner-rich-sutton-launches-oak-lab-to-build-continuously-learning-ai-agents/) (snippet)

**NetworkIDBD experiment (read directly from an independent reproduction README, Aug 2026) [REPL-attempt]:**
- The Oak Lab post "Learning from experience instead of curated datasets" describes the setup:
  - Input: a 64×64 frame with an MNIST digit on only 10% of samples and Bernoulli(0.01) background noise.
  - Targets: ±1 for odd/even digits and 0 when there is no digit, with Gaussian target noise of variance 5.
  - Network: **one 10,000-unit ReLU hidden layer**.
  - Data: **100,000 online samples**, each used once.
- NetworkIDBD "extends Sutton's Incremental Delta-Bar-Delta to a neural network so it can withhold credit from unpredictable targets and unused inputs."
- "Oak has not published NetworkIDBD equations or code."
- The independent reproduction used an Autostep-style candidate:
  - It cut clean MSE by **8.1–10.6%** relative to tuned SGD at 100k samples, and by **9.4%** at 1M samples.
  - Digit-sign accuracy rose from 0.9451 to 0.9501.
  - It "does **not** reproduce the strongly center-selective credit assignment shown by Oak."
  - Source: [CarsonBurke/NetworkIDBD](https://github.com/CarsonBurke/NetworkIDBD); Oak post: [oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets) (blocked; described via the README)

### Inferences
- About four years after the Alberta Plan (2022), the public Alberta/OaK line consists of **component algorithms** tested on small problems: streaming TD and AC, step-size meta-learning, CBP, and NetworkIDBD on a single-layer network. There is also one real-world single-task RL platform. **No integrated OaK agent (FC-STOMP loop) has been demonstrated publicly**, and no continual multi-task result exists on Physical Atari.
- The stated Oak Lab target makes the gap concrete: 10¹² parameters, real-time learning, 20 W. The brain runs about 10¹⁴ synapses at about 20 W. A 20 W digital agent with 10¹² parameters updating in real time would need roughly 10³–10⁴× better learning-energy efficiency than today's GPUs. This is my order-of-magnitude reasoning, not a sourced figure.
- The Keen/Oak split in 2026 suggests there is still no consensus, even among the program's founders, on the shortest path. Keen emphasizes a real-time robotics substrate; Oak Lab emphasizes new credit-assignment and step-size algorithms.
- Carmack's own diagnosis is that today's agents show negative transfer and forget sequentially. This is the direct opposite of the "compounding" the ASI thesis requires, and he frames the near-term goal as merely "avoid getting worse."

### Gaps
- I could not access the RLC Physical Atari paper's actual scores or sim-to-real gap numbers (keenagi.com blocked). I also could not confirm whether the RLC venue is 2025 or 2026.
- I could not fetch the Oak Lab post or mission page directly. The NetworkIDBD equations are unpublished.
- No talk transcript for NeurIPS 2025 OaK was retrieved. It is unknown whether Sutton gave a timeline there.
- I found no public Keen results on continual multi-game RL, despite Carmack's 2025 notes framing it as the core problem.

---

## Q4. Continual learning for LLMs and foundation models: measured forgetting vs gain, and production systems

### Takeaway
2025–2026 produced a toolkit for writing new information into LLM weights with little forgetting.
- **Sparse memory-layer fine-tuning** cuts forgetting on NaturalQuestions to −11%, against −89% for full FT and −71% for LoRA. An independent 0.5B re-implementation partly replicated it.
- **On-policy RL forgets less than SFT.**
- **Knowledge editing** scales to about 1M edits.
- **Self-editing** (SEAL: SQuAD no-context 33.5%→47.0%).
- **Test-time-memorizing architectures** (Titans, ATLAS, Hope, TTT-E2E).

All of these are demonstrated at ≤7B scale and over short edit sequences, or they are *context compression* rather than lifelong accumulation. The strongest at-scale "continual" result is economic, not cognitive: time-continual pretraining with replay matches periodic retraining at **2.6× less compute**. The only verified production weight-updating loop is Cursor's about-5-hour real-time RL cycle. It gains a few percent per cycle and is gated by offline regression evals.

### Cited Findings
**Weight-level knowledge incorporation:**
- **SEAL (Zweiger, Pari, Guo, Akyürek, Kim, Agrawal; MIT; NeurIPS 2025) [PR].**
  - The model is trained with RL to generate its own "self-edits" (fine-tuning data and update directives).
  - Knowledge incorporation: SQuAD no-passage-in-context rises **33.5% → 47.0%**, and self-generated data beats GPT-4.1-generated synthetic data.
  - Few-shot, on a *simplified subset* of ARC: **72.5%** success, against **0%** for ICL and **20%** for TTT with untrained self-edits.
  - Forgetting: "performance on earlier tasks gradually declines as the number of edits increases… still susceptible to catastrophic forgetting," though "without complete collapse."
  - "All experiments can be run with 2 A100/H100 GPUs."
  - Sources: [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/file/6b41e04c41726e2a60e456d0a2b961ab-Paper-Conference.pdf) (snippet); [repo README](https://github.com/Continual-Intelligence/SEAL)
- **SCoL, "Self-Consolidating Language Models" (May 2026) [PP].**
  - Training: an LLM generates textual instructions naming *which of its own layers* to update, trained by meta-RL with a reward for acquisition minus forgetting.
  - It beats in-context, summarization, batch and sequential-TTT baselines on SQuAD incorporation and LongBench v2.
  - The learned update locations are sparse and align with high-Fisher layers.
  - It transfers from short to longer streams.
  - Source: [arXiv 2605.07076](https://arxiv.org/abs/2605.07076) (snippet; no numbers retrieved)
- **Sparse memory fine-tuning (Jessy Lin et al., Meta, Oct 2025) [PP].**
  - Method: update only memory-layer slots that are highly activated by the new knowledge relative to pretraining usage (TF-IDF ranking).
  - Held-out **NaturalQuestions F1 drops 11%**, against **89% for full FT** and **71% for LoRA**, "with the same level of new knowledge acquisition."
  - Source: [arXiv 2510.15103](https://arxiv.org/abs/2510.15103) (snippet)
- **Independent re-implementation, Gupta et al. (May–June 2026) [PP][REPL-partial].**
  - On Qwen-2.5-0.5B-Instruct and MedMCQA, SMF gains **+2.5 pp** while keeping forgetting metrics "within roughly 1 point of the base model."
  - LoRA and full FT "achieve larger gains but with clear drift."
  - This shows that the low forgetting comes paired with **lower acquisition**.
  - Sources: [arXiv 2605.03229](https://arxiv.org/abs/2605.03229) (snippet); [Improving SMF, arXiv 2604.05248](https://arxiv.org/abs/2604.05248)
- **RL's Razor (Shenfeld et al.; ICLR 2026) [PR].** On-policy RL "is implicitly biased towards KL-minimal solutions," whereas SFT "can converge to distributions arbitrarily far from the base model." Forgetting is predicted by new-task KL. This is validated on LLMs and robotic foundation models. — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259) (snippet)
- **Lifelong knowledge editing.** UltraEdit (May 2025) reports keeping "high accuracy across **1 million edits**" on UltraEditBench (2M+ Wikidata pairs), whereas "previous methods typically degrade after **10,000 to 50,000** edits." Under lifelong editing, AlphaEdit and fine-tuning "significantly degrade the general ability" of the model. UltraEdit is ">7× faster" and uses "<1/3 the VRAM," and can edit a 7B model on a 24 GB GPU. — [arXiv 2505.14679](https://arxiv.org/abs/2505.14679) (snippet; self-reported on the authors' own benchmark)

**Architectures that learn at test time:**
- **Titans (Behrouz, Zhong, Mirrokni; Jan 2025) [PP].** A deep neural memory updated by gradient steps on "surprise" at test time. It scales to >2M-token contexts on needle-in-a-haystack, where Mamba2 and TTT degrade. — [arXiv 2501.00663](https://arxiv.org/abs/2501.00663) (snippet)
- **ATLAS (May 2025) [PP].** On BABILong, Titans' performance "drops in 10M" context while ATLAS keeps "+80% accuracy in 10M context length." — [arXiv 2505.23735](https://arxiv.org/abs/2505.23735) (snippet)
- **Nested Learning / Hope (NeurIPS 2025; full arXiv Dec 2025) [PR].**
  - Scale: models of 760M and 1.3B parameters, trained on 30B and 100B tokens.
  - At 1.3B/100B: Hope **15.11 WikiText perplexity / 57.23% average accuracy**, against Titans 15.60 / 56.82% and Transformer++ 18.53 / 52.25%.
  - Another secondary source reports 14.39 / 58.04% for the same configuration, so the figures conflict across write-ups.
  - Class-incremental text classification (CLINC, Banking) beats EWC. BABILong holds up to 10M tokens.
  - Sources: [arXiv 2512.24695](https://arxiv.org/abs/2512.24695); [Google Research blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) (snippets)
- **"Language Models Need Sleep" (Behrouz, Hashemi, Javanmard, Mirrokni; Google; June 2026) [PP].**
  - Sleep has two stages: "Knowledge Seeding," which distills a smaller self's memories *upward* into a larger network, and "Dreaming," in which RL generates a synthetic rehearsal curriculum.
  - It "consistently outperformed" ICL and other CL methods on sequential acquisition of new languages and classes.
  - Source: [arXiv 2606.03979](https://arxiv.org/abs/2606.03979) (snippet; no numbers retrieved)
- **TTT-E2E (Dec 2025) [PP].** Frames long context "as a problem in continual learning": a sliding-window Transformer keeps learning by next-token prediction on its context, with a meta-learned initialization.
  - For **3B models trained on 164B tokens** it "scales with context length in the same way as Transformer with full attention," which Mamba 2 and Gated DeltaNet do not.
  - Books loss at 128K is 2.67, against 2.70 for full attention.
  - It runs **2.7× faster than full attention at 128K**, with constant latency.
  - Sources: [repo README](https://github.com/test-time-training/e2e); [arXiv 2512.23675](https://arxiv.org/abs/2512.23675) (snippet)

**Amortizing inference with offline "sleep" compute:**
- **Sleep-time compute (Lin, Snell et al.; Letta and UC Berkeley; Apr 2025) [PP].**
  - Pre-computing over a context before queries arrive **reduces test-time compute about 5×** at equal accuracy on Stateful GSM-Symbolic and Stateful AIME.
  - Scaling sleep-time compute raises accuracy by up to **13% and 18%** respectively.
  - Amortizing across related queries **cuts average cost per query by 2.5×**.
  - Sources: [arXiv 2504.13171](https://arxiv.org/abs/2504.13171) (snippet); [repo](https://github.com/letta-ai/sleep-time-compute)

**Continual pretraining at web scale:**
- **TiC-LM (Apple; Apr 2025) [PP].** Covers 114 Common Crawl dumps, "orders of magnitude larger" than earlier continual-LM benchmarks. "Autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (**2.6×**)." Replay is "crucial" on generic web data but less so in specific domains. — [arXiv 2504.02107](https://arxiv.org/abs/2504.02107); [Apple ML Research](https://machinelearning.apple.com/research/tic-lm-web-scale) (snippets)

**Production systems that update weights in deployment (2026):**
- **Cursor, "Improving Composer through real-time RL" (Mar 2026) [SR].**
  - Each cycle collects "billions of tokens from user interactions with the current checkpoint," turns them into reward signals, "adjust[s] all model weights," checks for regressions on evaluation suites including CursorBench, then deploys.
  - The whole process "takes about five hours," so Cursor can ship an improved checkpoint "multiple times in a single day."
  - A/B results: agent edits persisted **+2.28%**, dissatisfied follow-ups **−3.13%**, latency **−10.3%**.
  - Sources: [Cursor blog](https://cursor.com/blog/real-time-rl-for-composer) (snippet); [Composer 2 technical report, arXiv 2603.24477](https://arxiv.org/html/2603.24477v2)
- Industry context: "Frontier-scale weight updates remain offline because every update needs evaluation, safety review, and rollback." Most deployed "online" learning is mini-batch adapter updates, not streaming SGD. — [FutureAGI blog, 2026](https://futureagi.com/blog/real-time-learning-in-large-language-models-llms/) (snippet)
  - **Caution:** the same blog claims "2026 frontier-scale evidence" of 15–32% capability degradation from continual fine-tuning of GPT-5.1, Claude Opus 4.5 and others. I could not trace a primary source, and closed models cannot be fine-tuned this way publicly. **Treat as unreliable.**
- **Field-level framing (Aug 2026 survey) [PP].** Continual learning is shifting "from parameter-centric learning toward system-level adaptation": on-policy learning, TTT at inference, and "external harness components such as memory, skill libraries." — [Continual Learning in Transition, arXiv 2608.06216](https://arxiv.org/abs/2608.06216) (snippet); see also [Harness Continual Learning, arXiv 2608.19013](https://arxiv.org/pdf/2608.19013)

### Inferences
- **Forgetting–acquisition trade-off, quantified.** The methods that forget least (SMF, KL-constrained RL, sparse edits) also acquire less per update, as the 0.5B replication shows (+2.5 pp against larger LoRA/FT gains). There is a Pareto frontier. No method dominates on both axes, so "no forgetting" today means "slower learning."
- **The "continual" architectures are mostly context compressors.** Titans, ATLAS, Hope and TTT-E2E write the *current context* into fast weights and are evaluated on single long documents (≤10M tokens). They show that test-time gradient learning is cheap and scalable within an episode. They do **not** show cross-episode accumulation over months, which is what the ASI thesis needs. Hope's CMS is the closest attempt, and its evidence is 1.3B/100B-token language modeling plus small class-incremental sets.
- **The measured compounding at LLM scale is 2–5×, and it is amortization, not acceleration.** TiC-LM saves 2.6× against retraining. Sleep-time compute saves about 5× at test time and 2.5× per query. Cursor gains a few percent per 5-hour cycle. None shows the marginal cost of *new capability* falling over successive updates.
- **Production reality.** The one production continual-weight loop runs on a 5-hour cycle, is gated by offline eval suites, and is driven by RL rather than SFT, which fits RL's Razor. The bottleneck is **evaluation and verification cost per update**, not the update itself.

### Gaps
- None of the forgetting results cover more than about 10³ sequential updates at ≥7B scale, except knowledge editing, where facts are narrow triples.
- SEAL's per-self-edit compute and its forgetting curve over the number of edits were not retrievable as numbers.
- There is no independent replication of Hope/Nested Learning or of the "Sleep" paper. The numbers conflict across secondary sources.
- No production system with weight updates *per user or per interaction* was found. Cursor's is fleet-level and batched.
- Whether frontier labs (OpenAI, Anthropic, Google) update weights continuously in deployment is undisclosed. I found no primary source.

---

## Q5. Lifelong skill and program-library accumulation: does marginal compute per task decline over thousands of tasks?

### Takeaway
**No. As of September 2026 there is no demonstration of declining marginal compute per new task over ≥10³ diverse tasks without forgetting or library bloat.**
- The largest lifelong-skill benchmarks of 2026 have **166 tasks** (SkillFlow) and **202 skills** (skill-shadowing study).
- Gains are model-dependent: +8.4 pp for Claude Opus 4.6, about 0 or negative for weaker models.
- Libraries **degrade performance by up to 21 points** as they grow to 202 skills. About 68% of that drop comes from wrong-skill selection.
- Under a **token-matched budget**, a plain agent matches or beats AWM, ASI and ReasoningBank on WebArena.

The best evidence for real amortization is **programmatic** skills that execute deterministically (SpeedRunner, Aug 2026; DreamCoder in narrow DSLs). It appears in narrow or embodied domains only.

### Cited Findings
**Classic program-library evidence (beyond the first-round notes):**
- **DreamCoder (PLDI 2021 [PR]; Phil. Trans. R. Soc. A 2023 [PR]).**
  - Across 8 domains it "always solved the most held-out tasks and generally solved them in the least time (**mean 54.1s; median 15.0s**)."
  - It reached "nearly 100%" of held-out tasks in LOGO graphics and tower building.
  - Training to convergence "typically takes around a day using moderate computational resources."
  - Task sets are on the order of 10²–10³ per domain, inside hand-designed DSLs.
  - Sources: [arXiv 2006.08381](https://arxiv.org/pdf/2006.08381); [Royal Society](https://royalsocietypublishing.org/rsta/article/381/2251/20220050/112456/DreamCoder-growing-generalizable-interpretable) (snippets)
- **HOUDINI (NeurIPS 2018) [PR]** framed lifelong learning as type-directed program synthesis over a growing library of neural functions, applied to sequences of related tasks. Its task sequences were short. — [arXiv 1804.00218](https://arxiv.org/pdf/1804.00218) (snippet)
- A 2025–2026 search for library learning over thousands of tasks with falling cumulative search cost returned only DreamCoder, LILO and HOUDINI-era work. I found **no 2025–2026 result** at that scale. — [search result set: LILO](https://proceedings.iclr.cc/paper_files/paper/2024/file/819cebb05f993840e8a52d7564c5c282-Paper-Conference.pdf), [Stitch](https://arxiv.org/pdf/2211.16605)

**Agent workflow and skill memory:**
- **Agent Workflow Memory (Wang, Mao, Fried, Neubig; ICML 2025) [PR].**
  - Relative success improves by **+24.6% on Mind2Web** and **+51.1% on WebArena**, with fewer steps.
  - Online AWM beats baselines by **8.9–14.0 absolute points** as the train–test gap widens.
  - Scope: 1,000+ tasks across 200+ domains in total, but not one long sequential stream measured for compounding.
  - Sources: [arXiv 2409.07429](https://arxiv.org/abs/2409.07429); [ICML poster](https://icml.cc/virtual/2025/poster/45496) (snippets)
- **Budget-matched re-evaluation (June 2026) [PP]. This is the key negative result.**
  - Setup: AWM, ASI (Agent Skill Induction) and ReasoningBank compared with a token-matched vanilla actor that spends the same budget on more steps. Three WebArena domains; three models (Gemini 3 Flash, GPT-5.4-mini, Qwen 3.6-27B).
  - "The vanilla baseline matched or surpassed all three augmentation methods in aggregate success rate while often using fewer total tokens."
  - Gains "often vanish against a budget-matched actor."
  - "Run-to-run variance materially affects outcomes."
  - Source: [arXiv 2606.15017](https://arxiv.org/abs/2606.15017) (snippet)
- **"More Skills, Worse Agents? Skill Shadowing" (Song & Wei, Databricks; Agent Skills '26 workshop) [PR-workshop].**
  - Performance falls "by up to **21%**" when scaling from a small helpful set to a **202-skill library**.
  - The decomposition gives about **68% to skill shadowing** (wrong-skill selection, which grows with library size) and about 30% to context overhead, which is "indistinguishable from zero" statistically.
  - Source: [arXiv 2605.24050](https://arxiv.org/abs/2605.24050) (snippet)
- **Retrieval at scale (secondary).** "Tool selection accuracy drops from above 90% with fewer than 30 candidates to **13.6% with 11,100**." A meta-analysis of **47,150 public Skills** found a median of about 1.5k tokens, mostly markdown documentation. Curated Skills raise success by **+16.2 pp across 84 tasks**. — surfaced together from [SkillsBench, arXiv 2602.12670](https://www.emergentmind.com/papers/2602.12670) and [Agent Skill Evaluation and Evolution survey, arXiv 2606.11435](https://arxiv.org/html/2606.11435v1) (snippet; **exact attribution of each number to paper uncertain**)
- **SkillFlow (Apr 2026) [PP].**
  - Size: **166 tasks, 20 workflow families**, 8–9 tasks per family sharing an execution flow, across 5 domains.
  - Lifelong skill evolution lifts **Claude Opus 4.6 from 62.65% to 71.08% (+8.43 pp)**.
  - **Kimi K2.5 gains +0.60 pp** despite a 66.87% skill-usage rate. **Qwen-Coder-Next regresses** relative to vanilla, at 44.58% completion.
  - "Most current models fail to achieve stable self-evolution through iterative skill updates."
  - Source: [arXiv 2604.17308](https://arxiv.org/abs/2604.17308) (snippet)
- **SkillLearnBench (Apr 2026) [PP]:** 20 tasks across 15 sub-domains for continual skill-generation methods. — [arXiv 2604.20087](https://arxiv.org/html/2604.20087v1) (snippet)
- **SpeedRunner, "Better, Faster, Stronger: Programmatic Skill Learning Best Reduces Agent Cost" (JHU, Aug 2026) [PP].**
  - Claim: skills represented as **programs** give the best cost reduction, because "the routine is reasoned about once and then invoked cheaply, whereas prose skills must be reread and re-followed on every use."
  - SpeedRunner, a coding agent that refactors skills from trajectories, "consistently achieves the frontier in learning and cost reduction" across **three embodied environments**, and is robust to distribution shift.
  - Source: [arXiv 2608.11338](https://arxiv.org/abs/2608.11338) (snippet; no numbers retrieved)
- Related mitigations for bloat: skill compression and graph compression ("SkillZip," Aug 2026), cost-aware skill rewriting, and "less-is-more" effects from compressing skill content. — [SkillZip, arXiv 2608.05604](https://arxiv.org/pdf/2608.05604); [What Should a Skill Remember?, arXiv 2606.09421](https://arxiv.org/html/2606.09421) (titles or snippets only)

### Inferences
- **The library curve is concave, then declining, at around 10² items.** The evidence points to an early benefit with a handful of relevant skills, then saturation and degradation by about 200 skills, driven by retrieval and selection errors. Nobody has shown the opposite at 10³–10⁴ items, which the ASI thesis requires. DreamCoder avoided the problem by compressing (MDL) rather than accumulating; LLM skill libraries mostly accumulate prose.
- **Measured compounding factor for LLM-agent skill memory: about 1.0× under budget-matched evaluation (June 2026).** Measured without budget matching, it is +8 to +51% relative success, but only for the strongest models. Much of the apparent gain from skill and memory modules is extra test-time compute in disguise.
- **Program-form skills are the exception that fits the thesis.** Executing code costs about zero LLM tokens per reuse, so amortization is real (SpeedRunner, Pang on ARC in the earlier notes, DreamCoder). This matches the first round's conclusion that "library reuse plus a few proposals plus execution checks" is the most compute-efficient pattern. The open question is still whether it scales beyond narrow or embodied domains and past about 10² programs without shadowing.
- Update to the earlier report's signal #3 (test-time learning or library compounding on 10³–10⁴ tasks): **still not met as of Sep 2026.** Two things are new: first measurements of *negative* scaling (−21 pts at 202 skills), and the finding that budget-matched gains vanish.

### Gaps
- No benchmark measures marginal FLOPs per new task along a sequence of ≥1,000 heterogeneous tasks. SkillFlow's 166 is the largest lifelong-skill benchmark found.
- SpeedRunner's numeric cost reductions and environment names could not be retrieved.
- There is no recent Voyager-style long-horizon study (thousands of Minecraft tasks) measuring library size against retrieval accuracy against cost.
- There is no measurement of library "forgetting," meaning skills broken by later refactoring, other than SkillFlow's qualitative note on unstable self-evolution.

---

## Q6. Complementary learning systems (fast episodic plus slow consolidation) and generative replay: measured efficiency

### Takeaway
CLS-inspired designs are everywhere in 2026 ML: dual-layer agent memory, "sleep" consolidation, Hope's multi-frequency CMS, and sparse memory layers as a fast store. Quantitative efficiency evidence, however, is thin. The measured numbers are:
- replay-based continual pretraining saves **2.6×** compute against retraining (TiC-LM);
- sleep-time pre-computation saves about **5×** test-time compute (and 2.5× per query);
- a sparse fast store limits forgetting to −11% against −89% (SMF).

I found no 2025–2026 result showing generative replay (as opposed to stored-data replay) matching stored replay at lower total compute at scale.

### Cited Findings
- **Replay is necessary at web scale.** In TiC-LM, fixed-ratio replay of old data is "crucial to avoid forgetting on generic web data but less so on specific domains." With autoregressive schedules it matches retraining at 2.6× less compute. — [arXiv 2504.02107](https://arxiv.org/abs/2504.02107) (snippet)
- **Brain-inspired generative replay (van de Ven et al.).** To cut generative replay's overhead, the generator was *integrated into the main model* through generative feedback or backward connections. Follow-ups in 2025–2026 include a "Brain Generative Replay" paper (ICANN 2025), "Insights from Brain-Inspired Replay" (Sep 2025) and "Prototype Latent World Model Replay for Class-Incremental Learning" (Jun 2026). — [Semantic Scholar, BI-R](https://www.semanticscholar.org/paper/Brain-inspired-replay-for-continual-learning-with-Ven-Siegelmann/7bfb4ef17eabec1acd266958bdb08622eebfbb05); [Springer ICANN 2025](https://link.springer.com/chapter/10.1007/978-3-032-04558-4_15); [arXiv 2509.00047](https://www.arxiv.org/pdf/2509.00047); [arXiv 2606.29465](https://arxiv.org/pdf/2606.29465) (snippets; no efficiency numbers retrieved)
- **LLM-scale CLS analogues (2026, all [PP] or workshop, qualitative only):**
  - **MIRROR (ICLR 2026):** "fast encoding of experience paired with slow reconstructive consolidation." — [ICLR 2026](https://iclr.cc/virtual/2026/10021270)
  - **Dual-Layer Agentic Memory:** "fast, selective write routing with slow parametric consolidation." — [arXiv 2608.22215](https://arxiv.org/html/2608.22215v2)
  - **EVAF:** a test–retest protocol for *selective* parametric consolidation. — [arXiv 2606.29916](https://arxiv.org/pdf/2606.29916)
  - **Memini (ICML 2026 workshop):** Benna–Fusi fast/slow synaptic variables on a memory graph. (snippets)
- **Self-generated replay:** a May 2026 paper studies "Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay." — [arXiv 2605.26097](https://arxiv.org/pdf/2605.26097) (title only)
- **"Sleep" consolidation in LLMs:**
  - Google's "Language Models Need Sleep" (Jun 2026): upward distillation plus RL "dreaming" curriculum. — [arXiv 2606.03979](https://arxiv.org/abs/2606.03979)
  - Letta's sleep-time compute gives the ~5× figure above. — [arXiv 2504.13171](https://arxiv.org/abs/2504.13171)
  - Karpathy's "maybe it's a LoRA" sleep phase is in the earlier notes.
- **Fast store as a separate parameter set:** sparse memory layers act as a hippocampus-like fast store with a TF-IDF write gate (−11% versus −89% forgetting). — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103) (snippet)

### Inferences
- The ML version of CLS that works today is **"store raw data plus replay"** (TiC-LM) and **"separate sparse fast parameters"** (SMF). Generative replay, where a model dreams its own past, remains mostly a small-benchmark technique (class-incremental CIFAR-scale) without published compute-efficiency wins at scale.
- For a low-compute lifelong agent, raw-data replay is cheap in FLOPs but costly in storage and runs against the streaming premise (Q2). Generative replay would fix storage but adds generator compute. No 2025–2026 source quantifies that trade.
- The 2026 wave of "sleep" papers (Google, Letta, SCoL) shows that the idea has become mainstream research. Every result is a preprint at ≤ a few-billion-parameter scale.

### Gaps
- I found no numeric comparison of generative against stored replay in FLOPs per retained unit of accuracy, in 2025–2026.
- No efficiency numbers were retrieved for MIRROR, Memini, EVAF or Dual-Layer Memory.
- There is no evidence on how CLS-style consolidation behaves over ≥10³ consolidation cycles, meaning whether errors in consolidated memory compound.

---

## Q7. Bottom line: the demonstrated "compounding factor" today, and the 2–3 hardest open problems

### Takeaway
The demonstrated compounding factor of continual learning, as of September 2026, is **about 1–5× and is amortization (not re-paying for what was already learned), not acceleration (each new skill getting cheaper to learn).**
- Continual pretraining saves 2.6× against retraining.
- Sleep-time pre-compute saves about 5× test-time compute.
- Programmatic skill or library reuse saves up to about 50× on narrow benchmarks (Pang, in the earlier notes).
- LLM prose-skill memory gives about 1× under budget matching.
- Deep RL shows forgetting or *negative* transfer across sequential tasks.

**No system has shown marginal cost per new task falling across ≥10³ diverse tasks while retaining old skills.** The core mechanism of the #1-ranked low-compute-ASI direction is therefore **still undemonstrated**. Its components (streaming learning, plasticity maintenance, low-forgetting writes, program libraries) each work in isolation, at small scale or over short horizons.

### Cited Findings
Summary of measured "compounding" (sources as in Q1–Q6):

| Mechanism | Best measured factor | Scale / horizon | Evidence | Date |
|---|---|---|---|---|
| Time-continual pretraining with replay vs periodic retraining | 2.6× less compute at equal loss | Web-scale, 114 CC dumps | [PP] [TiC-LM](https://arxiv.org/abs/2504.02107) | Apr 2025 |
| Sleep-time pre-computation | ~5× less test-time compute; 2.5× lower cost per query; +13%/+18% accuracy | Stateful GSM/AIME | [PP] [Lin et al.](https://arxiv.org/abs/2504.13171) | Apr 2025 |
| Sparse memory FT vs full FT / LoRA | Forgetting −11% vs −89% / −71% (NQ F1) at equal acquisition | Memory-layer LM (size not retrieved); 0.5B replication | [PP][REPL-partial] [Lin et al.](https://arxiv.org/abs/2510.15103), [Gupta et al.](https://arxiv.org/abs/2605.03229) | Oct 2025 / 2026 |
| Lifelong knowledge editing | 1M edits vs prior collapse at 10k–50k | 7B | [PP, self-benchmark] [UltraEdit](https://arxiv.org/abs/2505.14679) | May 2025 |
| Self-editing (SEAL) | SQuAD no-context 33.5→47.0%; forgetting grows with the number of edits | 2 GPUs | [PR] [SEAL](https://papers.neurips.cc/paper_files/paper/2025/file/6b41e04c41726e2a60e456d0a2b961ab-Paper-Conference.pdf) | Jun 2025 |
| Production real-time RL | +2.28% persisted edits, −3.13% dissatisfaction per ~5-hour cycle | Fleet-level, billions of tokens per cycle | [SR] [Cursor](https://cursor.com/blog/real-time-rl-for-composer) | Mar 2026 |
| Continual backprop vs backprop | Backprop decays from 89% to 77% over 2,000 tasks; CBP "appears to maintain plasticity indefinitely" (exact CBP accuracy not retrieved) | Small conv nets, binary ImageNet tasks | [PR] [Dohare et al.](https://github.com/shibhansh/loss-of-plasticity) | Aug 2024 |
| Streaming RL (stream-x) | ≈1× batch sample efficiency, with no buffer | Single tasks (MuJoCo, DMC, MinAtar, Atari) | [PR-workshop] [Elsayed et al.](https://arxiv.org/abs/2410.14606) | Oct 2024 |
| LLM skill or workflow memory | +8.4 pp (Opus 4.6); about 0 or negative for weaker models; ≈1× under token-matched budget; −21 pts at 202 skills | 166 tasks; 202 skills; WebArena | [PP] [SkillFlow](https://arxiv.org/abs/2604.17308), [budget study](https://arxiv.org/abs/2606.15017), [shadowing](https://arxiv.org/abs/2605.24050) | Apr–Jun 2026 |
| Sequential multi-game deep RL | Forgetting; GATO-style negative transfer | Atari | Carmack notes [SR] [slides](https://www.slideshare.net/slideshow/john-carmack-s-notes-from-his-upper-bound-2025-talk/279574708) | May 2025 |
| Plasticity at LLM scale | Loss persists at 5M–314M; onset delay grows sublinearly with size | GPT-style LMs | [PP] [Hernandez-Garcia et al.](https://arxiv.org/abs/2606.24752) | Jun 2026 |

### Inferences
**The three hardest open problems, ranked by how directly they block the low-compute-ASI mechanism:**

1. **Sustained plasticity plus retention in one compact network over a lifetime-length, non-stationary stream (the stability–plasticity problem at scale).**
   - Evidence: plasticity loss persists and scales only sublinearly with size (2026). Robust fixes (CBP, LN+WD) are shown only on small networks. Low-forgetting writes (SMF, KL-constrained RL) buy retention at the cost of acquisition speed. Diagnostics are unreliable ("optimization readiness" is a 2026 toy-scale proposal).
   - Oak Lab's founding bet (NetworkIDBD; per-weight meta-learned step sizes) is an explicit wager that this needs new *credit-assignment* algorithms, not new architectures. Its public evidence is one 10k-unit network on 100k samples.
   - A compact learner is hit hardest, because it cannot rely on over-parameterization to postpone plasticity loss.

2. **Positive forward transfer, meaning a falling marginal cost per new task.**
   - Every measurement over more than ~10² tasks or skills shows saturation, retrieval or shadowing failure, or negative transfer: −21 pts at 202 skills, ≈1× under budget matching, and GATO/Carmack negative transfer in RL.
   - The only clear amortization comes from *executable program* libraries in narrow domains and from caching or pre-compute (sleep-time).
   - The field also lacks the metric: no benchmark reports FLOPs per new task along a 10³–10⁴-task stream. The research agenda item from the earlier report ("在数千个任务的序列上，每个新任务的边际算力能否持续下降") has no benchmark to run on yet.

3. **Deciding what to consolidate, and verifying it cheaply (the consolidation and evaluation bottleneck).**
   - Every working weight-update loop gates updates behind expensive evaluation: Cursor runs regression suites every ~5 hours; SEAL's inner loop fine-tunes to score each self-edit; SCoL and Sleep use meta-RL with forgetting penalties.
   - Self-generated consolidation (SEAL, Sleep, generative replay) still shows cumulative forgetting as edits accumulate. It also inherits the verifier problem that the earlier report found limits self-improvement (collapse after 2–3 rounds).
   - For a low-compute agent, the per-update evaluation cost may dominate the update cost itself. It could erase the amortization gains unless verification is itself cheap: executable programs, or grounded environment feedback.

**Implications for the earlier report's #1 ranking** (my inference, not a source claim):
- The components are more mature than in 2024. Streaming RL works, forgetting can be reduced by about 8×, test-time weight learning scales to 10M-token contexts, and program skills amortize.
- The *compounding* claim, however, rests on no demonstration. Over the last 12 months the evidence shifted slightly against naive accumulation (library shadowing, budget-matched null results) and slightly toward **program-form, compressed** libraries and **RL-style (KL-minimal) weight updates**.
- A realistic near-term configuration therefore looks like "moderate pretrained prior + KL-constrained RL updates + sparse fast memory + compressed executable skill library + periodic sleep consolidation." That matches the report's fallback scenario of mid-scale pretraining followed by continual learning, which costs one to two more orders of magnitude, more than the pure compact-learner scenario.

**Falsifiable 2027 checkpoints to add to the signals table:**
- (a) Plasticity-loss scaling measured at ≥1B parameters, with a remedy that removes it.
- (b) A public benchmark of ≥1,000 sequential heterogeneous tasks reporting marginal FLOPs per task, forgetting, and library size.
- (c) Oak Lab or Keen publishing an integrated agent (FC-STOMP or equivalent) with positive transfer across ≥10 sequential Atari games, physical or emulated.

### Gaps
- No source reports an end-to-end FLOP ledger for any continual learner that could be compared with a retrain-from-scratch baseline over a long horizon. TiC-LM is the only one, and only for pretraining loss.
- Keen's and Oak Lab's internal results are unpublished. Carmack's "signs of life by ~2030" target (earlier notes) has no public intermediate milestone beyond Physical Atari.
- Frontier-lab continual-learning efforts (a widely discussed 2026 theme) have no primary technical disclosures that I could retrieve.
- Several 2026 numbers are secondary-source snippets because of blocked primary sites: SkillsBench attribution, Hope perplexities, Oak Lab goals. They should be re-verified before any headline use.
