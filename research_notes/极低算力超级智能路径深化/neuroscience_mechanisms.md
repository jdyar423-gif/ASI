# Neuroscience mechanisms behind the brain's sample- and energy-efficiency, and which transfer to AI (status as of 26 Sept 2026)

Method note for the report writer:
- **Access was very limited.** This session's egress proxy blocked nature.com, science.org, arxiv.org, PubMed/PMC, bioRxiv, e11.bio, emergentmind, gwern, substack and most university sites. The shared web-search budget ran out partway through.
- **Where the facts come from.** Most facts are search-engine extracts of primary sources (marked **(snippet)**). The rest come from GitHub-hosted material I read in full: the State of Brain Emulation Report 2025 data repository (cloned), Foresight talk transcripts, READMEs, and a Dwarkesh transcript saved in the session scratchpad. Those are marked **(read)**.
- **†** marks classic, widely reproduced findings that I cite from background knowledge with their DOI but did not re-fetch in this session. Spot-check them before quoting exact numbers.
- **Evidence tags:** [established] means replicated, textbook-level neuroscience. [strong] means causal or multi-lab evidence but still debated in details. [hypothesis] means a theory with partial support. [ML-measured] means a gain measured in an ML system.
- **Scope.** General brain-compute and energy anchors (1e15 FLOP/s, 1e24 FLOP lifetime, 20 W, <1e8 words, Sandberg's J/spike) and the JEPA/Dreamer/BabyLM/neuromorphic results are in `极低算力超级智能路径/theory_limits.md` and `brain_inspired_alternatives.md`. They are only cross-referenced here, not repeated.

---

## 1. Hippocampal one-shot/episodic learning, replay and systems consolidation (CLS, SWRs, BTSP) — and their ML implementations

### Takeaway
The hippocampus has the best-evidenced "sample-efficiency engine" in the brain:
- **One-shot writes.** BTSP creates a new place field from one dendritic plateau event, and its synaptic changes are 8–16x larger than STDP's.
- **Tagging of what to keep.** Awake sharp-wave ripples tag which experiences get kept (Science 2024).
- **Offline replay.** Replay during sleep is causally needed for, and can boost, memory. It also appears to *compose* new maps (Nat Neurosci 2025).
- **Selective consolidation.** Transfer to cortex is selective, and 2023 theory says it should happen only when it helps generalization.

ML versions give real but regime-limited gains:
- Episodic control learns much faster early. NEC scored 16.7% median human-normalized at 1M Atari frames, but prioritized replay overtakes it by 40M frames.
- Experience replay is foundational to deep RL.
- Retrieval memory gives about 25x parameter efficiency (RETRO).
- Generative replay mitigates forgetting.

Overall this is a **high-evidence, high-transferability** mechanism class. The missing piece is a working "consolidate only what generalizes" controller at scale.

### Cited Findings
**Complementary learning systems (CLS) theory**
- Original CLS (1995) [established†]: a fast-learning hippocampus stores episodes with sparse, pattern-separated codes. A slow-learning neocortex extracts structure through interleaved replay, which avoids catastrophic interference — [McClelland, McNaughton & O'Reilly 1995, Psych. Rev.†](https://doi.org/10.1037/0033-295X.102.3.419)
- "CLS updated" (2016) [hypothesis/review†] adds three points. Replay can be *prioritized* by reward and novelty. Neocortex can learn fast when new information fits an existing schema. CLS is explicitly linked to DQN's experience replay — [Kumaran, Hassabis & McClelland 2016, TICS†](https://doi.org/10.1016/j.tics.2016.05.004)
- Schema-accelerated consolidation [strong†]: rats that already had a spatial schema learned new paired associates in one trial, and these became hippocampus-independent within about 48 h rather than weeks — [Tse et al. 2007, Science†](https://doi.org/10.1126/science.1135935)
- **Generalization-regulated consolidation (Sun, Advani, Spruston, Saxe & Fitzgerald, Nat Neurosci, 20 July 2023)** [hypothesis, formal model]: in a neural-network formalization of systems consolidation, *unregulated* transfer of hippocampal memories to neocortex causes overfitting and harms generalization. They resolve this by proposing that memories consolidate only when doing so aids generalization, which explains why only a subset of memories consolidate — [Nat Neurosci](https://www.nature.com/articles/s41593-023-01382-9); [Janelia news](https://www.janelia.org/news/new-theory-better-explains-how-the-brain-stores-memories) (snippet)

**Sharp-wave ripples (SWRs) and replay**
- Causal necessity [strong†]: selectively disrupting SWRs during post-learning sleep impairs spatial memory — [Girardeau et al. 2009, Nat Neurosci†](https://doi.org/10.1038/nn.2384)
- Causal sufficiency and enhancement [strong†]: optogenetically *prolonging* SWRs improved memory in a maze task — [Fernández-Ruiz et al. 2019, Science†](https://doi.org/10.1126/science.aax0758)
- **Experience selection by awake SWRs (Yang, Sun, Huszár, Hainmueller, Kiselev & Buzsáki, Science, 28 Mar 2024)** [strong]:
  - Successive maze traversals were tracked by continuously drifting neuron populations.
  - When brain state shifted during reward consumption, SWRs occurred on some trials, and their content mostly decoded *that* trial.
  - Later sleep SWRs replayed mostly the waking-tagged trials.
  - Interpretation: awake SWRs act as a *tagging/selection* mechanism for consolidation, with reward favouring the state shift.
  — [Science](https://www.science.org/doi/10.1126/science.adk8261); [Buzsáki lab](https://buzsakilab.com/wp/2024/03/28/selection-of-experience-for-memory-by-hippocampal-sharp-wave-ripples/) (snippet)
- **Compositional replay (Bakermans, Warren, Whittington & Behrens, Nat Neurosci 28:1061–1072, 2025)** [hypothesis with data tests]:
  - If state spaces are built compositionally from reusable primitives, hippocampal responses can be read as compositional memories that *bind* the primitives.
  - This lets agents behave optimally in new environments "with no new learning".
  - Replay is predicted to build and consolidate these memories. In two datasets, replay from newly discovered landmarks induced and strengthened new remote firing fields.
  — [Nat Neurosci](https://www.nature.com/articles/s41593-025-01908-3); [SWC blog](https://www.sainsburywellcome.org/web/blog/imagining-future-exploring-role-hippocampal-replay) (snippet)
- In humans (MEG), replay has been reported to construct compositional solutions in hippocampal–prefrontal circuits [strong-ish†] — [Schwartenbeck et al. 2023, Cell†](https://doi.org/10.1016/j.cell.2023.09.004)
- A 2024 Nat Neurosci paper reports that inhibitory plasticity supports replay *generalization* in hippocampus. I have the title only — [Nat Neurosci s41593-024-01745-w](https://www.nature.com/articles/s41593-024-01745-w)

**Behavioral timescale synaptic plasticity (BTSP)**
- Discovery (Bittner, Milstein, Grienberger, Romani & Magee, Science 2017) [established†]: one dendritic plateau potential in CA1 creates a new place field in a *single trial*. It potentiates inputs active within a window of **seconds** before or after the plateau, far longer than STDP's tens of milliseconds — [Science†](https://doi.org/10.1126/science.aan3846)
- Current synthesis (J Neurosci review, Nov 2025) [established/strong] (snippet):
  - BTSP is "strong, bidirectional", acting on synaptic weights "over many seconds".
  - It is triggered by plateau potentials tied to somatic bursts, and so "capable of producing new place cells in one trial".
  - Models fitted to in vitro and in vivo data put **maximum potentiation 8–16x higher than STDP**.
  — [J Neurosci 45(46):e1332252025](https://www.jneurosci.org/content/45/46/e1332252025); [PMC12614050](https://pmc.ncbi.nlm.nih.gov/articles/PMC12614050/)
- Reach beyond space (2024) [strong]: BTSP in hippocampus creates *non-spatial* representations during learning and is modulated by entorhinal inputs. I have the title only — [PMC11383060](https://pmc.ncbi.nlm.nih.gov/articles/PMC11383060/). An earlier finding that entorhinal cortex supplies an instructive signal directing CA1 changes is from [Grienberger & Magee 2022, Nature†](https://doi.org/10.1038/s41586-022-05378-6)
- The field has matured into a Nature Neuroscience review, "BTSP: properties, elements and functions" (2026). I have the title only — [Nat Neurosci s41593-026-02214-2](https://www.nature.com/articles/s41593-026-02214-2)
- **BTSP as an ML primitive (Wu & Maass, Nat Commun 16:342, 2 Jan 2025)** [ML-measured, model]:
  - The rule "differs in five essential aspects" from earlier plasticity rules.
  - One-shot BTSP builds a **high-capacity content-addressable memory with binary synapses**, without needing high-resolution weights.
  - It reproduces the human-memory "repulsion effect", in which similar items are pulled apart.
  — [Nat Commun](https://www.nature.com/articles/s41467-024-55563-6) (snippet)
  - Follow-ups: BTSP combined with hyperdimensional computing — [ResearchGate](https://www.researchgate.net/publication/397330134_Behavioral_Time_Scale_Synaptic_Plasticity_BTSP_endows_Hyperdimensional_Computing_with_brain-like_information_retrieval_flexibility); a CCN 2025 abstract on BTSP plus replay — [CCN 2025](https://2025.ccneuro.org/abstract_pdf/Cho_2025_Behavioral_timescale_synaptic_plasticity_replay_emergent.pdf). Titles only.

**ML implementations and measured gains**
- Model-free episodic control (Blundell et al. 2016): a hippocampus-inspired table of the best returns seen, used to act. On Atari it outperformed DQN and A3C variants "in the initial phase of learning"; the authors argued standard deep RL needs "thousands of hours of in-game time" — [arXiv 1606.04460](https://arxiv.org/pdf/1606.04460); [Blundell PDF](https://www.gatsby.ucl.ac.uk/~ucgtcbl/papers/BluUriPriLiRudLeiRaeWieHas2016a.pdf) (snippet)
- **Neural Episodic Control (Pritzel et al., ICML 2017)** [ML-measured] (snippet):
  - Mechanism: a differentiable neural dictionary with slow keys and fast-updating values.
  - Median human-normalized Atari score: **16.7% at 1M frames** and **36.0% at 4M frames**.
  - It met or beat human performance on about 25% of games within 10M frames.
  - Prioritized replay eventually overtakes it: **89.0% vs NEC's 83.3% at 40M frames**.
  — [PMLR](https://proceedings.mlr.press/v70/pritzel17a/pritzel17a.pdf); [review summary](https://liner.com/review/neural-episodic-control)
- Experience replay itself is a hippocampal-replay analogue and was essential to DQN's stability [established†] — [Mnih et al. 2015, Nature†](https://doi.org/10.1038/nature14236). The 2016 CLS update ties the two together explicitly — [Kumaran et al. 2016†](https://doi.org/10.1016/j.tics.2016.05.004)
- Brain-inspired generative replay (van de Ven, Siegelmann & Tolias, Nat Commun 2020) [ML-measured]:
  - Method: replay *internal/hidden* representations generated by the network's own context-modulated feedback connections.
  - Result: state of the art on class-incremental split CIFAR-100, and "replaying just a few or low-quality samples can already be sufficient".
  - Exact accuracies not retrieved.
  — [Nat Commun](https://www.nature.com/articles/s41467-020-17866-2); [GitHub](https://github.com/GMvandeVen/brain-inspired-replay) (snippet)
- Sleep-like unsupervised replay reduced catastrophic forgetting in ANNs [ML-measured†] — [Tadros et al. 2022, Nat Commun†](https://doi.org/10.1038/s41467-022-34938-7)
- **Retrieval as an episodic store (RETRO, DeepMind, Dec 2021)** [ML-measured] (snippet):
  - A 7.5B model retrieving from a database of trillions of tokens performs comparably to GPT-3 (175B) and Jurassic-1 (178B), with **25x fewer parameters**.
  - The gain is roughly constant from 150M to 7B parameters and improves at test time with a larger database.
  — [arXiv 2112.04426](https://arxiv.org/pdf/2112.04426); [Synced](https://syncedreview.com/2021/12/13/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-164/)
  - Caveat: a follow-up attributes much of RETRO's perplexity gain to lexical overlap between the retrieval database and test data. This is my recollection of its abstract, not re-verified — [arXiv 2302.12128†](https://arxiv.org/pdf/2302.12128)
- Hippocampus-inspired LLM memory, 2024–2025 [ML-measured, magnitudes not retrieved]:
  - HippoRAG (NeurIPS 2024) and HippoRAG 2 (ICML 2025) claim better multi-hop retrieval and "sense-making" than standard RAG, with cheaper offline indexing than GraphRAG, RAPTOR and LightRAG — [GitHub](https://github.com/OSU-NLP-Group/HippoRAG) (read)
  - EM-LLM (ICLR 2025): segments the token stream into episodic "events" using Bayesian surprise, then retrieves by similarity and temporal contiguity. It outperforms InfLLM and RAG on LongBench and ∞-Bench and retrieves across **10M tokens** — [GitHub](https://github.com/em-llm/EM-LLM-model) (read)
- Continual-learning memory architectures (Titans, Nested Learning/Hope) are in `brain_inspired_alternatives.md` §7 and are not repeated here.

### Inferences
- **Why this matters for efficiency.** Three mechanisms combine:
  - **BTSP:** one experience yields a stored, retrievable trace immediately.
  - **SWR tagging:** chooses which traces are worth rehearsing, weighted by reward.
  - **Offline replay:** reuses each tagged experience many times, "free" in terms of new data.

  Together these turn a single sample into many effective gradient steps for the slow system. That is a plausible source of an order-of-magnitude-level data advantage, though no study quantifies the multiplier in animals.
- **The ML record matches the theory.** Nonparametric or episodic memory wins big *early* and in low-data regimes (NEC 16.7% median at 1M frames) but is overtaken asymptotically (PR 89% vs 83.3% at 40M frames). This is exactly what CLS predicts for a system with no well-tuned fast-to-slow consolidation.
- **Retrieval saves capacity, not total compute.** Retrieval memory is the largest measured parameter-efficiency gain (25x), but it shifts cost to a large external store and may partly reflect data overlap.
- **The biggest unexploited idea is "consolidate only what generalizes" (go-CLS)** together with reward- or salience-tagged replay (Yang/Buzsáki). Current LLM pipelines either never consolidate (RAG, context) or consolidate everything (fine-tuning), which the go-CLS model predicts will overfit.
- **Compositional replay (Bakermans 2025) is the most ASI-relevant new result.** It suggests replay is not just rehearsal but *constructive search* over recombined primitives, giving zero-new-learning generalization. This parallels test-time search and refinement loops (see `theory_limits.md` Q3/Q6), but it has not been implemented as such in ML at scale.

### Gaps
- No quantitative estimate of the effective "data multiplier" of sleep replay in animals (replays per experience × efficacy).
- Exact accuracy numbers for BI-R (van de Ven 2020), Wu & Maass capacity figures, and HippoRAG/EM-LLM percentage gains were not retrievable (paper pages blocked).
- No large-scale (≥1B-parameter) ML test of generalization-gated consolidation (go-CLS) or SWR-style tagged replay was found.
- BTSP in neocortex or in humans: not established in sources I could access.

---

## 2. Cortical predictive learning, world models, cognitive maps and factorized/compositional representations

### Takeaway
- **Predictive processing** is a well-supported *description* of cortex [strong but contested in its specific "error-unit" form]. The most solid quantitative link is that better next-word predictors are better models of cortical language responses.
- **Cognitive maps** (grid cells, place cells) plus the Behrens-lab idea of **factorizing abstract structure from sensory content** and binding them in hippocampus (the Tolman-Eichenbaum Machine, TEM) explain rapid transfer of structural knowledge to new environments [hypothesis with growing support].
  - TEM is formally close to a transformer with recurrent positional encodings (TEM-t), which is a more sample-efficient learner than TEM.
- **Measured ML gains** from these ideas come mainly from predictive objectives already covered elsewhere (JEPA over 10x, Dreamer about 100x data; see `brain_inspired_alternatives.md` §1) and from meta-learned compositionality at small scale (MLC). No large-scale gain from explicit grid/structure factorization has been demonstrated.

### Cited Findings
**Predictive processing in cortex**
- Predictive processing is proposed as a "canonical cortical computation" [hypothesis/review†]. Evidence includes layer-specific mismatch (prediction-error-like) responses to violated sensorimotor expectations in mouse V1 — [Keller & Mrsic-Flogel 2018, Neuron†](https://doi.org/10.1016/j.neuron.2018.10.003)
- Across dozens of language models, next-word-prediction performance predicts how well a model's activations explain human language-network responses (fMRI/ECoG) [strong†] — [Schrimpf et al. 2021, PNAS†](https://doi.org/10.1073/pnas.2105646118)
- Adding *long-range, multi-timescale* forecasts to GPT-2 activations improves their mapping onto brain responses to speech. The forecast horizon lengthens up the cortical hierarchy, i.e., a "predictive coding hierarchy" [strong†] — [Caucheteux, Gramfort & King 2023, Nat Hum Behav†](https://doi.org/10.1038/s41562-022-01516-2)

**Cognitive maps and the Tolman-Eichenbaum Machine**
- **TEM (Whittington, Muller, Mark, Chen, Barry, Burgess & Behrens, Cell, Nov 2020)** [hypothesis, model]:
  - Entorhinal cells represent abstract structure ("where", position-like codes g).
  - Lateral entorhinal cortex carries sensory content ("what", x).
  - Hippocampus binds the two in fast conjunctive memories (p).
  - Trained as a sequence predictor across many environments sharing a structure, TEM develops grid, band, border and object-vector cells, and unifies spatial and relational memory.
  — [Cell](https://www.sciencedirect.com/science/article/pii/S009286742031388X); [PubMed](https://pubmed.ncbi.nlm.nih.gov/33181068/) (snippet)
- **TEM ↔ transformers (Whittington, Warren & Behrens, ICLR 2022)** [ML-measured, small scale]:
  - A transformer with *recurrent* position encodings learns grid and band cells in its position encodings when trained on tasks needing abstract spatial knowledge.
  - The authors show it is closely related to TEM.
  - Their Figure 4 reports **"TEM-t is a more efficient learner than TEM, both in sample efficiency and time per gradient step"**, and can store and retrieve many more memories. The measure is zero-shot accuracy on never-taken links to visited states.
  — [arXiv 2112.04035](https://arxiv.org/html/2112.04035v2); full text read via a [GitHub mirror](https://github.com/tiendungchs/PersonalWiki/blob/main/raw/t-TEM.md) (read)
- Compositional construction by replay: see Bakermans et al. 2025 in section 1 — [Nat Neurosci](https://www.nature.com/articles/s41593-025-01908-3)
- Meta-learning for compositionality (MLC, Lake & Baroni, Nature 623:115–121, 2023) [ML-measured]: a standard seq2seq transformer trained over a stream of few-shot compositional tasks reached human-like systematic generalization and modelled human few-shot errors — [GitHub](https://github.com/brendenlake/MLC) (read); [Nature](https://www.nature.com/articles/s41586-023-06668-3)
- Latent predictive world models (I-JEPA over 10x cheaper than MAE; V-JEPA 2-AC with 62 h of robot video; Dreamer 4 with about 100x less data than VPT) are documented in `brain_inspired_alternatives.md` §1 and not repeated.

### Inferences
- **Evidence strength splits in two.** That cortex *learns by prediction*, i.e., self-supervised, is strongly supported both behaviourally and through the LLM-brain alignment literature. That cortex implements *explicit error units* (classical predictive coding) remains contested. For AI, only the first claim matters, and it is already the dominant paradigm (next-token prediction, JEPA). This mechanism is therefore mostly *already transferred*. The remaining brain-derived increments are:
  - (a) predict in *latent* space, not pixels or tokens (JEPA);
  - (b) predict at *multiple horizons* (Caucheteux);
  - (c) use actions and interaction as part of the predicted stream (world models).
- **Factorized structure ⟂ content with fast binding is the key brain idea for *compositional* sample efficiency.** Once a structural "map" is learned, a new environment or task costs only fast hippocampal binding (one-shot) rather than slow relearning. In ML this corresponds to:
  - abstract positional or relational encodings;
  - in-context learning with attention as the binding memory;
  - meta-learning over task families (MLC).

  The TEM-t equivalence suggests transformers already implement part of it. The untested frontier is *explicitly learned, reusable structural codes* (e.g., learned relational position codes shared across domains) at LLM scale.
- **Measured gains exist but are small-scale.** TEM-t is more sample-efficient than TEM, and MLC achieves human-level systematic generalization on a toy grammar. There is no evidence yet of a large multiplier on broad benchmarks.

### Gaps
- I could not verify 2023–2026 Allen Institute OpenScope and other results that directly test (and in some reports contest) expectation-driven prediction errors in mouse cortex. The web-search budget was exhausted before this was checked.
- No quantitative ML result was found showing grid-like or TEM-style factorized codes improving sample efficiency in large-scale language or vision models.
- The exact sample-efficiency ratio of TEM-t vs TEM (figure only) was not extracted numerically.

---

## 3. Credit assignment beyond backprop: dendrites, bursts, neuromodulators, three-factor rules

### Takeaway
How cortex assigns credit across layers remains **unknown** [hypothesis].
- **Candidates proposed so far:**
  - burst-multiplexed apical-dendrite signals (burstprop);
  - prospective configuration (infer target activity first, then change weights);
  - eligibility traces plus neuromodulatory third factors (e-prop).

  Each can approximate gradient descent. Only prospective configuration claims *better* sample efficiency than backprop, and only at small scale in online, continual and non-stationary settings.
- **Neuromodulation is better evidenced:**
  - Dopamine as a TD reward-prediction error is established, with 2020–2025 refinements: distributional codes and heterogeneous discount timescales (Nature 2025).
  - Challenges remain: dopamine as an *adaptive learning rate* (Nature 2023) or as *retrospective causal* learning (Science 2022). The latter is partly rebutted by a 2025 study.
- **ML verdict:**
  - TD learning and distributional RL are proven ML wins.
  - Local learning rules mostly buy **biological plausibility and on-chip/online learnability**, not sample efficiency on GPUs.

### Cited Findings
**Local credit-assignment rules**
- Review [established framing†]: exact backprop is implausible in cortex, but feedback connections could convey error-like signals that approximate gradients — [Lillicrap, Santoro, Marris, Akerman & Hinton 2020, Nat Rev Neurosci†](https://doi.org/10.1038/s41583-020-0277-3)
- **Burstprop (Payeur, Guerguiev, Zenke, Richards & Naud, Nat Neurosci, May 2021)** [hypothesis + ML demo]:
  - Rule: a presynaptic event followed by a postsynaptic *burst* induces LTP; otherwise LTD.
  - Bursting is controlled by top-down feedback onto apical dendrites, so bursts multiplex an error-like signal with feedforward events.
  - It "approximate[s] gradient descent" and was trained on MNIST, CIFAR-10 and ImageNet.
  - Accuracy numbers were not retrievable; the code README gives none.
  — [Nat Neurosci](https://www.nature.com/articles/s41593-021-00857-x); [Zenke lab](https://zenkelab.org/2021/05/paper-burst-dependent-synaptic-plasticity-can-coordinate-learning-in-hierarchical-circuits/); [code](https://github.com/jordan-g/Burstprop) (snippet/read)
- A single-phase successor, BurstCCN (Greedy et al., NeurIPS 2022), learns in cortico-cortical-style networks without separate phases — [arXiv 2206.11769](https://arxiv.org/html/2206.11769v2) (snippet)
- **Prospective configuration (Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz, Nat Neurosci 27:348–358, published 3 Jan 2024)** [ML-measured, small scale]:
  - Method: the network first relaxes (as energy-based or predictive-coding networks do) to the activity pattern that *should* follow learning, then changes weights to consolidate it.
  - Claims: "more efficient in learning" than backprop and better at explaining animal and human learning data.
  - Settings reported, per a secondary summary of the paper:
    - online (batch size 1) learning;
    - continual learning with alternating FashionMNIST tasks;
    - concept drift;
    - limited-data learning;
    - CNNs on CIFAR-10;
    - RL on three classic-control tasks (higher reward per episode).
  - The mechanism is reduced interference between weight updates ("target alignment").
  — [Nat Neurosci](https://www.nature.com/articles/s41593-023-01514-1); [Oxford news](https://www.ox.ac.uk/news/2024-01-03-study-shows-way-brain-learns-different-way-artificial-intelligence-systems-learn); [code](https://github.com/YuhangSong/Prospective-Configuration); [secondary summary (Chinese)](https://github.com/lao-xiang-1/study) (snippet/read; effect sizes not retrieved)
- e-prop (Bellec et al., Nat Commun 2020) [ML-measured†]: eligibility traces combined with online, top-down "learning signals" (three-factor rule) train recurrent spiking networks close to BPTT performance on several benchmarks — [Nat Commun†](https://doi.org/10.1038/s41467-020-17236-y)
- Predictive-coding networks now match backprop at ResNet-18/CIFAR-10 scale but need iterative inference, which is more compute per example on GPUs. See `brain_inspired_alternatives.md` §2.
- Evolved Hebbian rules (Najarro & Risi, NeurIPS 2020): networks with *random* initial weights and evolved per-synapse Hebbian rules self-organize during each episode to solve RL tasks such as CarRacing — [GitHub](https://github.com/enajx/HebbianMetaLearning) (read)

**Neuromodulators**
- Dopamine as TD reward-prediction error [established†] — [Schultz, Dayan & Montague 1997, Science†](https://doi.org/10.1126/science.275.5306.1593). Dopamine neurons carry a *distributional* code for value, as in distributional RL [strong†] — [Dabney et al. 2020, Nature†](https://doi.org/10.1038/s41586-019-1924-6)
- **Multi-timescale RL (Masset et al., Nature 642:682–690, June 2025)** [strong]:
  - Mouse dopamine neurons encode RPEs with a **diversity of discount time constants**.
  - Agents that learn at many timescales "possess distinct computational benefits".
  - Per the press summary, multi-discount agents were "more efficient at solving complex learning tasks" than single-discount agents.
  — [Nature](https://www.nature.com/articles/s41586-025-08929-9); [Harvard MCB](https://www.mcb.harvard.edu/department/news/understanding-dopamine-neurons-on-multiple-timescales/) (snippet)
- **Dopamine as learning rate (Coddington, Lindo & Dudman, Nature, 18 Jan 2023)** [strong, contested]: calibrated manipulations of mesolimbic dopamine gave effects "inconsistent with value learning" but predicted by a model in which dopamine sets an **adaptive learning rate** for policy learning, not an error signal — [Nature](https://www.nature.com/articles/s41586-022-05614-z) (snippet)
- **Retrospective causal learning (Jeong … Namboodiri, Science, 23 Dec 2022)** [contested]: the ANCCR model proposes that animals search backward for causes when a meaningful event occurs, and that mesolimbic dopamine conveys causal associations rather than RPE — [Science](https://www.science.org/doi/10.1126/science.abq6740) (snippet)
  - Rebuttal (Nat Neurosci 2025): behaviour and dopamine track *prospective*, not retrospective, contingency, which TD models explain — [Nat Neurosci s41593-025-01915-4](https://www.nature.com/articles/s41593-025-01915-4) (snippet)
  - Also 2026: "Duration between rewards controls the rate of behavioral and dopaminergic learning" (title only) — [Nat Neurosci s41593-026-02206-2](https://www.nature.com/articles/s41593-026-02206-2)
- Acetylcholine and noradrenaline [hypothesis†]: ACh is proposed to signal *expected* uncertainty and NE *unexpected* uncertainty. Together they regulate how much new evidence overrides priors, effectively a learning-rate and attention control — [Yu & Dayan 2005, Neuron†](https://doi.org/10.1016/j.neuron.2005.04.026)
- ML follow-up on "artificial dopamine": TD learning with distributed, local error signals (title only) — [arXiv 2411.03604](https://arxiv.org/pdf/2411.03604)

### Inferences
- **Do these mechanisms improve sample efficiency, or only plausibility?**
  - **Burstprop, BurstCCN, e-prop and PC:** plausibility and locality. At best they match backprop, usually slightly below and at small scale. Their efficiency case is for neuromorphic or analog on-chip learning, where locality removes data movement (see `brain_inspired_alternatives.md` §5–6). They do *not* cut the number of samples.
  - **Prospective configuration:** the one credible "better than backprop per sample" claim. The wins come in regimes that matter for a lifelong learner (batch size 1, non-stationary, continual). Evidence is small networks and toy RL, not replicated at scale, and it costs extra inference iterations per update.
  - **Neuromodulatory ideas that already transferred and paid off:** TD errors (the basis of deep RL) and distributional value (distributional RL). Multi-timescale discounting (Masset 2025) and uncertainty- or dopamine-gated learning rates (Coddington 2023; Yu & Dayan) are cheap to add. They are relevant to *sample* efficiency because they tell the learner *how much* to update per sample. Measured ML gains exist mostly in RL, not in LLM pretraining.
- **The brain's likely edge is *when and how much* to learn, not *how* to compute gradients.** Dopamine and ACh/NE gate learning-rate, plasticity and replay tagging (SWRs during reward). This "meta-control of plasticity" is more promising for low-compute AI than replacing backprop.

### Gaps
- Numerical accuracy of burstprop on ImageNet, and effect sizes for prospective configuration, were not retrievable (paper pages blocked).
- No large-scale (≥100M-parameter) test of prospective configuration or any local rule beating backprop in samples-to-accuracy was found.
- I could not check 2025 in-vivo work on compartment-specific plasticity rules in cortical dendrites (search budget exhausted).
- No quantitative ML result was found for ACh/NE-style uncertainty-gated learning rates improving LLM or vision data efficiency.

---

## 4. Sparsity and energy: sparse coding, event-driven computation, spike vs FLOP — and whether sparsity multiplies *learning* efficiency

### Takeaway
- **The brain's energy efficiency rests on:**
  - very sparse activity: under ~1% of cortical neurons strongly active at once, with an energetic optimum of ~1–4%;
  - event-driven signalling;
  - minimised communication: communication costs ~35x more ATP than computation, and cortical "computation" uses only ~0.1–0.2 W.

  These facts are well established.
- **ML has already reproduced two forms of sparsity:**
  - *emergent* activation sparsity: 3% of T5 MLP units and 6.3% of ViT-B16 units are non-zero per input;
  - *engineered* conditional computation (MoE): ~7x compute saving at matched quality, with the gap widening with scale.
- **What sparsity multiplies:** compute and energy per token, and possibly capacity against interference (continual learning). It is *not* a proven multiplier of samples-to-competence.

### Cited Findings
- **Energy budget and sparse activity (Lennie, Curr Biol, Mar 2003)** [established] (snippet):
  - Spikes are expensive: 10 spikes in 200 ms cost more than 100x maintaining the resting potential.
  - This limits concurrently active neurons to "possibly … fewer than 1%".
  - The energetically optimal active fraction is ~1–4% for populations representing 100–1,000 entities (about 3% as a point estimate).
  — [Curr Biol](https://www.sciencedirect.com/science/article/pii/S0960982203001350); [PDF](https://www2.bcs.rochester.edu/sites/plennie/pdfs/Lennie03a.pdf)
- Earlier energy budget [established†]: the brain's signalling energy is dominated by action potentials and postsynaptic currents — [Attwell & Laughlin 2001, J Cereb Blood Flow Metab†](https://doi.org/10.1097/00004647-200110000-00001)
- **Communication vs computation (Levy & Calvert, PNAS, Apr 2021)** [strong] (snippet):
  - An ATP audit of human cortex finds **communication** (axonal spikes, transmitter release) uses **35x more energy than computation**.
  - Cortical computation is allotted **~0.1 W** of ATP; a companion preprint title says "less than 0.2 watts".
  — [PNAS](https://www.pnas.org/doi/abs/10.1073/pnas.2008173118); [arXiv 2102.06273](https://arxiv.org/abs/2102.06273)
- Sparse coding as a learning principle [established†]: optimizing for sparse codes of natural images yields V1-like receptive fields — [Olshausen & Field 1996, Nature†](https://doi.org/10.1038/381607a0)
- **Emergent sparsity in transformers ("Lazy Neuron", Li et al., ICLR 2023)** [ML-measured] (snippet):
  - Per input, only **3.0% (T5-Base)** and **6.3% (ViT-B16)** of MLP activations are non-zero.
  - Larger and wider models are *sparser*.
  - This holds for NLP and vision, on training and evaluation data, at all layers.
  — [arXiv 2210.06313](https://arxiv.org/abs/2210.06313); [ICLR](https://iclr.cc/virtual/2023/poster/11756)
- **MoE compute leverage (Tian et al., "Towards Greater Leverage", arXiv 2507.17702; ICLR 2026)** [ML-measured] (snippet):
  - Scope: over 300 models up to 28B parameters.
  - "Efficiency leverage" follows power laws in the expert *activation ratio* and total compute, with an optimal granularity range.
  - On an identical 1T-token dataset, Ling-mini-beta (**0.85B active**) matched a **6.1B dense** model using **over 7x less compute**.
  - The MoE–dense gap widens with compute.
  — [arXiv](https://arxiv.org/abs/2507.17702); [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/32b640528f5b67975562210f00c131ed-Paper-Conference.pdf)
- Sparsity and continual learning [ML-measured†]: sparse, top-k, sparse-distributed-memory-style MLPs reduce catastrophic forgetting in class-incremental settings — [Bricken et al. 2023, "Sparse Distributed Memory is a Continual Learner", ICLR†](https://arxiv.org/abs/2303.11934)
- Energy per spike and per synaptic event (Sandberg) and GPU J/FLOP figures are in `theory_limits.md` Q2. Spiking LLMs and neuromorphic chips (SpikingBrain, Loihi 2, NorthPole) are in `brain_inspired_alternatives.md` §5.

### Inferences
- **Energy per spike vs per FLOP.** From the cross-referenced figures (~1e-14 J per synaptic event vs ~5–7e-13 J per GPU MAC), a synaptic event is ~50–70x cheaper than a GPU MAC. The brain *also* performs far fewer events than a dense network would, because only ~1–4% of units fire at a time. Levy & Calvert imply that at ~0.1 W, "compute" is nearly free and the budget is spent moving information. **For silicon, the lesson is to minimise data movement and activate few units**, which is what near-memory chips and MoE do.
- **Sparsity is a big *compute* multiplier but not a demonstrated *sample* multiplier.** MoE gives ~3–7x+ at fixed data, with no evidence of reaching a given capability with fewer tokens beyond what extra total parameters give. Its plausible sample-efficiency role is indirect: sparse codes reduce interference, enabling one-shot writes without overwriting. This supports the hippocampal/CLS mechanisms in section 1 (pattern separation), rather than being a standalone learning-efficiency lever.
- **The brain's 1–4% activity is a similar order to the lazy-neuron sparsity** (3–6%). The remaining silicon gap is exploiting that sparsity in hardware (event-driven, near-memory), not discovering it.

### Gaps
- No study was found that measures samples-to-accuracy (not FLOPs) benefits of activation sparsity or MoE at matched total parameters.
- No reliable 2025–2026 estimate was found of the actual fraction of human cortical neurons active per second across tasks; the ~1% figure is Lennie's energetic bound.

---

## 5. Connectomics and whole-brain emulation as a shortcut to an efficient learning algorithm

### Takeaway
**Where connectomics stands (2024–2025):**
- A whole-fly connectome: 139,255 neurons, over 50M synapses, Nature Oct 2024.
- A 1 mm³ mouse-cortex connectome with co-registered function: 200k cells, 523M synapses, ~75k functionally imaged neurons, Nature Apr 2025.
- A 1 mm³ human sample.

**What connectome models can do:**
- Connectome-based models of the fly *predict* activity: 91% accuracy for whole-brain LIF sensorimotor predictions; agreement with 26 studies for the visual system when combined with task optimization.
- The connectome **alone is generally insufficient**. It must be paired with recordings, learned single-neuron parameters, or molecular annotations.

**Cost, compute and timelines:**
- A whole mouse connectome costs $6–21B with today's EM and proofreading. E11 Bio/PRISM-style expansion microscopy with barcoded self-proofreading aims for ~$10–100M (company target).
- Human-scale work is still $100M–$100B+ and 2040s-scale.
- *Running* an emulation is cheap by AI standards: 3e16–1e19 FLOP/s real-time for a human brain at LIF level, i.e., ~30–14,000 H100s.

**Verdict:** reverse-engineering is **not a near-term shortcut** for 2026–2030 low-compute ASI. Its realistic value is as a source of *constraints and priors*: cell types, wiring rules, reward-circuit ("steering subsystem") architecture. It does not supply ready-made learned weights or plasticity rules, which need molecular data.

### Cited Findings
**FlyWire whole-fly connectome (Dorkenwald et al. + FlyWire Consortium, Nature, 2 Oct 2024)** [established] (snippet)
- Scale: **139,255 proofread neurons** and **over 50 million** synaptic connections, from serial-section TEM of a female *D. melanogaster* brain.
- Effort: about **20 person-years** of proofreading, much of it by citizen scientists; over 100 labs; about a dozen Nature papers.
- Sources: [Nature collection](https://www.nature.com/collections/hgcfafejia); [Princeton](https://www.princeton.edu/news/2024/10/02/mapping-entire-fly-brain-step-toward-understanding-diseases-human-brain); [BrainFacts](https://www.brainfacts.org/neuroscience-in-society/supporting-research/2024/researchers-create-first-adult-fruit-fly-brain-connectome-110724)

**Whole-brain LIF model (Shiu et al., Nature 634:210–219, Oct 2024)** [strong]
- Method: a leaky integrate-and-fire model of the whole fly brain, using connectome weights and predicted neurotransmitter identity (~139k neurons, ~3.7M connections).
- Results: predicted circuits for feeding initiation and antennal grooming, with "91% prediction accuracy against experimental *Drosophila* data" as cited by the Eon Systems README.
- The model ran on a laptop (Berkeley News headline).
- Sources: [PubMed](https://pubmed.ncbi.nlm.nih.gov/39358519/); [Eon Systems fly-brain repo](https://github.com/eonsystemspbc/fly-brain) (read); [fruit-fly-lab repo](https://github.com/syn-ack-ai/fruit-fly-lab); [Berkeley News](https://news.berkeley.edu/2024/10/02/researchers-simulate-an-entire-fly-brain-on-a-laptop-is-a-human-brain-next/) (snippet)

**Connectome-constrained deep mechanistic networks (Lappalainen et al., Nature, Sept 2024)** [strong] (snippet)
- Method: experimentally determined connectivity for **64 cell types** of the fly optic-lobe motion pathway; unknown neuron and synapse parameters were fit by deep learning so the network detects motion.
- Result: predictions agreed with measurements across **26 studies**.
- Key caveat: "connectome datasets alone are generally not sufficient to predict neural activity", but pairing connectivity with recordings gives accurate predictions for *unrecorded* neurons.
- Sources: [Nature](https://www.nature.com/articles/s41586-024-07939-3); [Janelia](https://www.janelia.org/news/researchers-combine-the-power-of-ai-and-the-connectome-to-predict-brain-cell-activity)
- A 2025 Nat Neurosci paper, "Prediction of neural activity in connectome-constrained recurrent networks", addresses the same identifiability question (title only) — [Nat Neurosci](https://www.nature.com/articles/s41593-025-02080-4)

**MICrONS (seven papers, Nature 640, 10 Apr 2025)** [established] (snippet)
- Content: a mouse visual-cortex volume of ~1 mm³ with **over 200,000 cells, 523 million synapses and 4 km of axons**, plus functional imaging of **~75,000 neurons** during visual stimulation.
- Firsts: the first EM reconstruction spanning multiple functional areas; the largest multimodal connectomics dataset as of 2025.
- A "like-to-like" connectivity rule was reported from these data.
- Sources: [ScienceDaily](https://www.sciencedaily.com/releases/2025/04/250409114838.htm); [MICrONS Explorer](https://www.microns-explorer.org/cortical-mm3); [Nature issue 8058](https://www.nature.com/nature/volumes/640/issues/8058); like-to-like note from the [SoBE data repo](https://github.com/MxSchons-GmbH/sobe-2025-data-repository) (read)

**Human 1 mm³ (H01)** [established†]
- ~1.4 PB of EM data, ~57,000 cells and ~150M synapses from human temporal cortex — [Shapson-Coe et al. 2024, Science†](https://doi.org/10.1126/science.adk4858)
- The 2025 SoBE talk confirms "in humans, a single cubic millimeter has recently been mapped using EM" — [Zanichelli transcript](https://github.com/kanzure/diyhpluswiki/blob/master/transcripts/foresight-institute/niccolo-zanichelli-state-of-brain-emulation-2025.mdwn) (read)

**E11 Bio PRISM** [company claim]
- Method: protein barcoding of individual neurons, expansion microscopy imaged on light-sheet microscopes, and AI "self-proofreading". It targets manual proofreading, which is ">95% of expenses".
- Cost claim: "brain mapping at 100x lower cost", with an entire mouse connectome in ~5 years for ~$100M, versus a Wellcome-Trust-style estimate of ~$10B and ~15 years.
- Dates: announced Dec 2024; further results publicised Oct 2025 (publication venue not confirmed).
- Sources: [E11 PRISM](https://www.e11.bio/blog/prism); [E11 roadmap](https://www.e11.bio/blog/roadmap); [MedicalXpress, Oct 2025](https://medicalxpress.com/news/2025-10-protein-barcodes-brain-circuits-scale.html); [X post summarising E11](https://x.com/kimmonismus/status/1864023550278906262) (snippet)
- Marblestone (Convergent Research, which incubated E11), 30 Dec 2025 (read):
  - The Wellcome Trust report put the first mouse connectome at "several billion dollars".
  - E11 and the field aim for "low tens of millions of dollars" per mouse connectome.
  - A human brain is ~1,000x bigger, so it is "still billions" if scaled naively.
  - A focused programme (whole mouse, a human "Steering Subsystem", several mammals) could cost "hundreds of millions to low billions".
  - His own AI timelines are "10-year-ish".
  — [Dwarkesh Podcast transcript](https://www.dwarkesh.com/p/adam-marblestone)

**State of Brain Emulation Report 2025 (Zanichelli, Schons, Shiu, Freeman & Arkhipov; arXiv 2510.15745, Oct 2025; Zenodo v1 2026)**
- Framing:
  - Three capabilities: neural dynamics recording, connectomics, and computational neuroscience.
  - The field is now collecting enough data to emulate sub-million-neuron organisms such as larval zebrafish.
  - Fewer than **500 people** worldwide work directly on brain emulation.
  - The report is 175 pages with 41 expert reviewers.
  — [arXiv](https://arxiv.org/abs/2510.15745); [LessWrong launch post](https://www.lesswrong.com/posts/qgttsmkESTwjjEE4E/the-state-of-brain-emulation-report-2025-launched) (snippet)
- **Real-time compute, from the report's data repository (read)** — [SoBE data repo](https://github.com/MxSchons-GmbH/sobe-2025-data-repository), `data/compute/computational-demands-organisms.tsv`; the repo notes that QC was "ongoing":
  - Assumptions: point-neuron LIF at 10 Hz and 1e4 timesteps/s.

  | Organism | Time-stepped (FLOP/s) | Event-driven (FLOP/s) | Storage |
  |---|---|---|---|
  | Fly brain | 2.7e12–4.8e12 | — | — |
  | Mouse brain | 6.75e15–1.1e16 | 2.7e13–3.1e14 | 1.1–2.2 TB |
  | Human brain | 8.5e18–1.39e19 | 3.4e16–8.7e16 | 1.4–2.7 PB |

  - The summary table gives 0.195 PFLOPS (fly), 10 PFLOPS (mouse) and 2,000 PFLOPS (human) — [same repo](https://github.com/MxSchons-GmbH/sobe-2025-data-repository), `compute-requirements.tsv` (read)
  - A search extract of the report says "a single H100 GPU can simulate approximately 500,000 to 1 million neurons" — [arXiv PDF](https://arxiv.org/pdf/2510.15745) (snippet)
- **Connectome cost per neuron, from `costs/neuron-reconstruction-estimates.tsv` (read):**

  | Project / estimate | Year | Cost per neuron |
  |---|---|---|
  | *C. elegans* (White et al.) | 1986 | $16,556 |
  | Fly (FlyWire) | ~2018–2024 | $214 |
  | Zebrafish | 2021 | $100 |
  | Mouse 10 mm³, NIH BRAIN CONNECTS ($43M for 1.5M neurons) | 2024 | $29 |
  | Whole mouse, Wellcome estimate (10 nm EM) | 2023 | $307 |
  | Whole mouse, 15 nm EM with current proofreading | 2025 | $92 |
  | Illustrative, 1000x less proofreading (EM or ExM) | 2030 | $2.16–7.66 |

  - Proofreading assumption: **5 human-hours per neuron at $50/h**, i.e., $250 per neuron, falling to 0.005 h with 1000x automation (`costs/proofreading.tsv`).
- Search extracts of the report and related coverage (page ambiguous, possibly the Asimov Press "Building Brains on a Computer" article) (snippet):
  - Whole-mouse imaging ≈ $200–300M plus human proofreading ≈ $7–21B.
  - Budget thresholds of ~$1B for a mouse connectome by 2030 and ~$100B for a human.
  - "The first convincing mouse brain emulation … about one billion dollars in the 2030s, and tens of billions for the first human brain emulation model by the late 2040s".
  — [arXiv PDF](https://arxiv.org/pdf/2510.15745); [Asimov Press](https://press.asimov.com/articles/brains)

**Bottlenecks named by SoBE-affiliated speakers (Foresight 2025 talks, read)**
- Zanichelli:
  - Whole-brain optical recording is feasible or near-feasible for worm, fly and larval zebrafish.
  - In mammals, light scattering limits recording depth to **~1.5 mm**, so mouse and human emulation "will likely have to rely primarily on structural data".
  - **Structure-to-function is "the main bottleneck"**.
  - A human-scale effort would need an AI-scaling-law-like "industrial approach".
  — [transcript](https://github.com/kanzure/diyhpluswiki/blob/master/transcripts/foresight-institute/niccolo-zanichelli-state-of-brain-emulation-2025.mdwn)
- Matelsky (UPenn/Kording lab):
  - Recording every human neuron would take electrophysiology "on the order of centuries"; optical recording, with a ~2–3 year doubling time, "probably 40 to 50 years".
  - Perturbations dramatically cut the data needed to infer connectivity. In a *C. elegans* simulation, a few seconds of heavily perturbed recording rivalled days of unperturbed data, "off by an order of magnitude in either direction".
  - Plasticity rules are "implemented in molecules", so "if you're not annotating molecules, you have no hope".
  — [transcript](https://github.com/kanzure/diyhpluswiki/blob/master/transcripts/foresight-institute/jordan-matelsky-scaling-brain-emulation-2025.mdwn)

**What a connectome can reveal (Marblestone, Dec 2025)** [hypothesis] (read)
- "The analogous thing to connectome is like seeing the weights". Its value is as "a huge number of additional constraints" on hypotheses about architecture, learning rules and loss functions.
- Cell-census data already show "more weird and diverse and bespoke cell types in the Steering Subsystem" (hypothalamus and other subcortical reward/instinct circuits) than in cortex.
- Source: [Dwarkesh transcript](https://www.dwarkesh.com/p/adam-marblestone)

### Inferences
- **Compute is not the barrier.** Real-time human emulation at LIF level is 3.4e16–1.4e19 FLOP/s, which is ~34–14,000 H100-equivalents at ~1e15 FLOP/s each (my arithmetic). That is well within 2026 frontier clusters. The barriers are:
  - (i) acquisition cost. Human 8.6e10 neurons × $2–8/neuron (2030 illustrative) ≈ **$190–660B**; today's $92–307/neuron gives ~$8–26T. So ~100–1,000x more cost reduction is needed to reach the ~$1B–$100B range discussed.
  - (ii) the **structure-to-function / molecular-annotation gap**. Fly results show connectivity alone underdetermines dynamics.
  - (iii) plasticity. A static connectome is a snapshot of *learned weights*, not the learning algorithm.
- **As a shortcut to an *efficient learning algorithm*, connectomics is indirect and slow:**
  - Earliest whole-mouse connectome: ~2030 if E11-style costs materialise (company target).
  - Human-scale emulation: 2040s (SoBE extract).
  - The most plausible payoff for AI is **reverse-engineering the innate "steering" and reward architecture** (loss functions, curricula) and wiring motifs as *priors*, plus connectome-constrained models as inductive biases (Lappalainen-style). That is valuable for a 2030s design, but it will not shortcut 2026–2030 efforts.
- **The fly is the proof of concept for "connectome + task optimization → function".** No ML *capability* gain from connectome-derived architectures has yet been shown.

### Gaps
- No peer-reviewed number was found for PRISM's measured error rates or per-neuron cost; the $100M figure is a company target.
- MICrONS project cost (IARPA, ~2016–2025) was not retrieved.
- No published estimate was found of the extra data needed (molecular annotations per synapse) to infer plasticity rules from structure.
- The SoBE data repo carried a "QC ongoing" notice; the compute rows have `confidence: none`.

---

## 6. Infant learning efficiency: active/curious learning, social learning, self-generated curricula, embodiment — and evolved priors

### Takeaway
- **Headcam studies show the raw data a child sees is learnable by generic ML at moderate efficiency:**
  - 61 h (~1% of waking hours) of one child's audio-visual input trained a CLIP-like model to **61.6%** word-referent accuracy vs CLIP's 66.7%, using ~1e4x fewer image-text pairs.
  - 200 h of headcam video reached ~70% of an ImageNet-trained model.
  - ViTs trained "through the eyes" of newborn chicks matched chick-level view-invariance.
- **Developmental data ordering helps:** youngest-first, slow and simple input gave the best downstream results (NeurIPS 2023).
- **Infants actively allocate attention by learning progress.** Social live interaction is required for some learning (phonetics).
- **The largest efficiency factor is probably *evolved priors and reward/loss functions*.**
  - Zador's genomic bottleneck compresses networks by orders of magnitude and improves transfer.
  - Marblestone and Byrnes place the "secret sauce" in innate reward and loss functions.
  - Synthetic "prenatal-like" pre-pretraining (neural cellular automata) gives 1.4–1.6x faster LM convergence.
- **Measured ML transfer is real but modest (1.x–10x). Curricula and multimodality did not help in BabyLM** (see `brain_inspired_alternatives.md` §4).

### Cited Findings
**Headcam and egocentric-video studies**
- **Vong, Wang, Orhan & Lake, Science, 1 Feb 2024** [ML-measured] (snippet):
  - Data: **61 hours** of head-mounted-camera recordings from one child aged 6–25 months, about **1% of waking hours**. That is 600,000 frames paired with 37,500 transcribed utterances, or ~250k word instances.
  - Model: a contrastive vision-language model (CVCL).
  - Result: **61.6%** classification accuracy on 22 concepts, versus **66.7%** for CLIP ViT-L/14, with zero-shot generalization to new referents.
  — [Science](https://www.science.org/doi/10.1126/science.adi1374); [NYU news](https://www.nyu.edu/about/news-publications/news/2024/february/ai-learns-through-the-eyes-and-ears-of-a-child.html); [PsyPost](https://www.psypost.org/ai-autonomously-learns-language-through-the-experience-of-a-single-child-in-groundbreaking-study/)
- **Orhan & Lake, Nat Mach Intell 6:271–283, 2024** [ML-measured] (snippet):
  - Self-supervised models (DINO, MAE and others) trained on **200 hours** of one child's headcam video over two years, with no strong inductive biases.
  - The best embedding models reached **~70% of a high-performance ImageNet-trained model**, learned broad categories and localization, but were less object-centric.
  — [NMI](https://www.nature.com/articles/s42256-024-00802-0); [model menagerie](https://github.com/eminorhan/silicon-menagerie) (read)
- **Curriculum from infant video (Sheybani, Hansaria, Wood, Smith & Tiganj, NeurIPS 2023 spotlight)** [ML-measured]:
  - Pretraining first on the *youngest* infants' egocentric video "provided the strongest learning signal and led to the best learning outcomes" downstream.
  - Tested with generative (VideoMAE), predictive (JEPA) and contrastive (SimCLR) SSL.
  - Attributed to temporal slowness and spatial simplicity of young infants' input.
  — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html); [code](https://github.com/ssheybani/baby-vision-curriculum) (snippet/read)
- **Embodied controlled rearing (Pandey, Wood & Wood, NeurIPS 2023)** [ML-measured]:
  - ViTs trained with a contrastive-through-time loss (successive views within 300 ms) on first-person images from virtual replicas of newborn-chick chambers, each containing a single object.
  - They "solved the same view-invariant object recognition tasks as the chicks".
  - The authors conclude that "ViTs were not more data hungry than newborn visual systems" *given embodied temporal data streams*.
  — [GitHub/abstract](https://github.com/buildingamind/ViT-CoT) (read)
- Related 2025–2026 child-view ML work (titles only):
  - "Temporal Slowness in Central Vision Drives Semantic Object Learning" — [arXiv 2602.04462](https://arxiv.org/pdf/2602.04462)
  - "Human Gaze Boosts Object-Centered Representation Learning" — [arXiv 2501.02966](https://arxiv.org/pdf/2501.02966)
  - "On the robustness of modeling grounded word learning through a child's egocentric input" — [arXiv 2507.14749](https://arxiv.org/pdf/2507.14749)
  - "Continual Visual and Verbal Learning Through a Child's Egocentric Input" (June 2026) — [arXiv 2606.05115](https://arxiv.org/html/2606.05115)

**Active, curious and social learning**
- **Active attention by learning progress (Poli, Serino, Mars & Hunnius, Sci Adv, Sept 2020)** [strong] (snippet):
  - Infants' saccadic latencies, looking times and disengagement are explained by an ideal learner's measures of *surprise*, *predictability* and *learning progress*.
  - Infants "tailor their attention to maximize learning".
  — [Sci Adv](https://www.science.org/doi/10.1126/sciadv.abb5053)
  - Related titles: "Volatility-driven learning in human infants" (Sci Adv) — [link](https://www.science.org/doi/10.1126/sciadv.adu2014); "Guiding Curiosity: How Learning Progress Shapes Young Children's Exploration of New Toys" (Dev Sci, 2026) — [link](https://onlinelibrary.wiley.com/doi/10.1111/desc.70229)
- Social gating of learning [strong†]: 9-month-old American infants learned Mandarin phonetic contrasts from brief *live* interaction with a speaker (a few hours in total), but not from the same material via audio or video recording — [Kuhl, Tsao & Liu 2003, PNAS†](https://doi.org/10.1073/pnas.1532872100)
- Curiosity in ML [ML-measured]: purely curiosity-driven agents (prediction-error intrinsic reward, *no* extrinsic reward) were studied across **54 environments**. Random features sufficed as the prediction space for many benchmarks — [Burda, Edwards, Pathak et al. 2019, GitHub](https://github.com/openai/large-scale-curiosity) (read)
- Review: "Lessons from infant learning for unsupervised machine learning" (NMI 2022; title only) — [NMI](https://www.nature.com/articles/s42256-022-00488-2)

**Evolved priors: genome, reward functions, prenatal "pretraining"**
- "A critique of pure learning" [hypothesis†]: much animal competence is innate and encoded through a "genomic bottleneck", and that bottleneck acts as a regularizer favouring generalizable circuits — [Zador 2019, Nat Commun†](https://doi.org/10.1038/s41467-019-11786-6)
- **Genomic bottleneck, quantified (Shuvaev, Lachi, Koulakov & Zador, PNAS, Sept 2024)** [ML-measured] (snippet):
  - Standard architectures' weight matrices can be compressed through a small "genomic network" by **several orders of magnitude**, with performance at initialization "approach[ing] that of the fully trained network".
  - For *complex* but not simple tasks, the compressed "genome" gives **enhanced transfer learning** to new tasks and datasets.
  — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2409160121); [PubMed](https://pubmed.ncbi.nlm.nih.gov/39264740/)
- **Reward and loss functions as the "secret sauce" (Marblestone on Dwarkesh, 30 Dec 2025)** [hypothesis] (read):
  - "The brain's secret sauce is its reward functions, not its architecture."
  - Evolution "may have built a lot of complexity into the loss functions … many different loss functions for different areas turned on at different stages of development … generating a specific curriculum".
  - He cites Steve Byrnes' split into a Learning Subsystem (cortex, little pre-initialization) and a Steering Subsystem (innate reward and heuristics).
  — [Dwarkesh](https://www.dwarkesh.com/p/adam-marblestone)
- **Synthetic "prenatal" pre-pretraining ("Training Language Models via Neural Cellular Automata", arXiv 2603.10055, Mar 2026; code under GitHub user danihyunlee; I did not verify the author list)** [preprint, ML-measured] (read):
  - Setup: **164M tokens** of neural-cellular-automata trajectories before natural-language pretraining (4–13B tokens), versus from scratch, C4 or Dyck pre-pretraining at matched budget.
  - Final perplexity: **4–6% better** (OpenWebText 14.66 → 13.82) and **1.6x faster convergence**.
  - Downstream: GSM8K 3.82% → 4.36%; HumanEval 6.75% → 7.49%; BigBench-Lite 20.91% → 26.51%.
  - Against **10x more C4 data (1.6B tokens)**, NCA still converged 1.4x faster with 5% better perplexity.
  — [project page/code](https://github.com/danihyunlee/nca-pre-pretraining); [arXiv 2603.10055](https://arxiv.org/abs/2603.10055)
- Formal-language pre-pretraining (Hu et al. 2025, "Between Circuits and Chomsky") also "increase[s] pre-training token efficiency" for Pythia-160M. Exact savings not retrieved — [GitHub](https://github.com/michahu/pre-pretraining) (read)

### Inferences
- **Decomposing the child's advantage (my synthesis, not established):**
  - (a) **Priors.** An evolved architecture plus a curriculum of innate loss and reward functions, plus prenatal self-generated activity. These are probably the largest factor but the least understood.
  - (b) **Data quality and ordering.** Slow, simple, egocentric, temporally coherent input curated by the infant's own attention and learning-progress-driven curiosity.
  - (c) **Social gating.** Live social interaction as a teaching signal that determines what is learned (Kuhl).
  - (d) **Fast memory.** Hippocampal one-shot binding (section 1).
- **What ML has transferred, and how much it bought:**
  - (b) gives modest gains: Sheybani youngest-first curriculum; Pandey time-contrastive learning makes ViTs chick-efficient in impoverished environments.
  - (a) gives modest but real gains via synthetic structured pre-pretraining (NCA 1.4–1.6x) and genome-style compression.
  - CVCL's ~1e4x fewer image-text pairs for ~92% of CLIP accuracy (61.6/66.7) holds only on a narrow 22-concept test. CLIP's ~400M-pair training set is †, and the ratio is my arithmetic.
- **Embodiment's effect on data efficiency is real but narrow.** Embodied, temporally coherent streams make generic learners as data-efficient as newborn chicks for *object invariance*. There is no evidence yet for general cognition.
- **For a low-compute ASI, the practical analogue of "evolved priors" is distillation from existing frontier models plus structured synthetic pre-pretraining.** Evolution's ~1e41 FLOP discovery cost (`theory_limits.md` Q6) has, in effect, already been partly paid by today's frontier training runs.

### Gaps
- Numerical downstream gains for Sheybani's curriculum and exact compression factors in Shuvaev 2024 were not retrievable.
- No study was found that quantifies the separate contributions of social interaction, active attention and embodiment to children's language-learning efficiency.
- The BabyView and other large 2024–2026 headcam corpora, and models trained on them, were not checked.

---

## 7. Bottom line: ranking mechanisms by (evidence it matters for brain efficiency) × (evidence of ML gain when transplanted)

### Takeaway
Top-ranked mechanisms, in order:
- (1) evolved priors and reward/loss-function curricula, whose ML analogue is pretraining, distillation and structured pre-pretraining;
- (2) predictive latent world-model learning;
- (3) hippocampal fast episodic memory with one-shot writes and retrieval;
- (4) tagged/prioritized replay with generalization-gated consolidation;
- (5) sparse conditional computation.

Factorized structure codes, learning-progress-driven active learning and multi-timescale neuromodulated learning rates are mid-ranked. Local backprop alternatives and connectome emulation rank lowest *for a 2026–2030 low-compute design*.

The measured ML gains of the top mechanisms are each ~2–25x, sometimes ~100x in-domain. Stacked, they plausibly cover ~2–4 of the ~5 orders of magnitude of the language-data gap. This is speculative; no system has combined them.

### Cited Findings
The scores below synthesise sections 1–6 and the cross-referenced prior notes. Every numeric entry is sourced above or in `brain_inspired_alternatives.md` / `theory_limits.md`. Scores run 1 (weak) to 3 (strong).

| # | Mechanism | Brain-efficiency evidence | ML transplant evidence (best measured gain) | Product | Include in low-compute ASI design? |
|---|---|---|---|---|---|
| 1 | Evolved priors and innate loss/reward curricula (genomic bottleneck, "steering subsystem", prenatal activity) | 3 (innate behaviour; bottleneck theory; cell-type census) but mechanism poorly known | 2–3: pretraining and distillation are the de facto analogue; NCA pre-pretraining 1.4–1.6x; genome compression by orders of magnitude with better transfer ([PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2409160121); [arXiv 2603.10055](https://arxiv.org/abs/2603.10055)) | 7–9 | **Yes.** Distil from frontier models, add structured synthetic pre-pretraining and hand-designed multi-objective losses |
| 2 | Predictive (self-supervised) latent world models | 2–3 (prediction–brain alignment; error-unit form contested) | 3: I-JEPA over 10x cheaper; Dreamer 4 about 100x less data; V-JEPA 2-AC with 62 h robot data (`brain_inspired_alternatives.md` §1) | 6–9 | **Yes.** Predict in latent space, at multiple horizons, action-conditioned |
| 3 | Hippocampal fast episodic memory (BTSP-like one-shot writes, pattern separation, retrieval) | 3 (BTSP one-trial place fields, 8–16x STDP) | 2–3: NEC 16.7% median at 1M frames; RETRO 25x fewer parameters; EM-LLM retrieval over 10M tokens ([J Neurosci 2025](https://www.jneurosci.org/content/45/46/e1332252025); [PMLR](https://proceedings.mlr.press/v70/pritzel17a/pritzel17a.pdf); [RETRO](https://arxiv.org/pdf/2112.04426)) | 6–9 | **Yes.** Writable episodic store with one-shot binding, as a first-class component |
| 4 | Replay and systems consolidation (SWR tagging, prioritized/generative replay, go-CLS gating) | 3 (causal SWR disruption and prolongation; Science 2024 tagging) | 2: DQN replay essential; BI-R state of the art in class-incremental learning; small-scale only for consolidation gating ([Nat Commun 2020](https://www.nature.com/articles/s41467-020-17866-2); [Nat Neurosci 2023](https://www.nature.com/articles/s41593-023-01382-9)) | 6 | **Yes.** Salience/reward-tagged replay, consolidating only what improves held-out generalization |
| 5 | Sparse, event-driven activation and conditional computation | 3 for *energy* (1–4% active; communication 35x computation) | 3 for *compute* (MoE over 7x; emergent 3–6% sparsity); 1 for *samples* ([arXiv 2507.17702](https://arxiv.org/abs/2507.17702); [Lennie 2003](https://www.sciencedirect.com/science/article/pii/S0960982203001350)) | 6 (compute), 3 (samples) | **Yes, for compute and energy.** Do not count on it for data efficiency |
| 6 | Factorized structure ⟂ content codes (grid/TEM) and compositional binding | 2 (grid cells established; factorization theory supported by 2023–25 replay/composition data) | 1–2: TEM-t more sample-efficient than TEM; MLC human-like systematicity at toy scale ([ICLR 2022](https://arxiv.org/html/2112.04035v2); [MLC](https://github.com/brendenlake/MLC)) | 2–4 | **Yes, as a research bet.** Reusable relational/positional codes plus meta-learning over task families |
| 7 | Active, curiosity/learning-progress-driven data selection; developmental and social curricula | 2–3 (infants tailor attention; live social gating) | 1–2: youngest-first curriculum gains; curiosity RL; BabyLM curricula mostly failed ([Sci Adv 2020](https://www.science.org/doi/10.1126/sciadv.abb5053); [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a9ad92a81748a31ef6f2ef68d775da46-Abstract-Conference.html)) | 2–6 | **Yes, cheaply.** Learning-progress-based data selection; interaction with teacher models |
| 8 | Neuromodulated learning control (TD/distributional/multi-timescale RPE; dopamine as learning rate; ACh/NE uncertainty) | 3 for RPE; 2 for newer roles (contested) | 2: TD and distributional RL foundational; multi-timescale agents "more efficient" in Masset 2025 models ([Nature 2025](https://www.nature.com/articles/s41586-025-08929-9)) | 4–6 | **Yes.** Multi-horizon value heads; uncertainty-gated learning rates and plasticity |
| 9 | Local credit assignment (burstprop, dendritic, PC, e-prop, prospective configuration) | 1–2 (mechanism unknown) | 1: parity with backprop at best; prospective configuration wins only in small online/continual settings ([Nat Neurosci 2024](https://www.nature.com/articles/s41593-023-01514-1)) | 1–2 | **No** for GPU training. Revisit for on-chip continual learning hardware |
| 10 | Connectome/WBE reverse-engineering | n/a (a method, not a mechanism) | 1: fly connectome models predict activity (91%; 26 studies), but no ML capability gain; mouse connectome ~2030+ at $10M–$6B+ ([Shiu 2024](https://pubmed.ncbi.nlm.nih.gov/39358519/); [Lappalainen 2024](https://www.nature.com/articles/s41586-024-07939-3); [SoBE data](https://github.com/MxSchons-GmbH/sobe-2025-data-repository)) | 1 (2026–30); 2–4 (2030s) | **Not for 2026–2030.** Fund as a long-term source of priors (steering circuits, wiring rules) |

### Inferences
- **The recommended low-compute ASI "brain recipe" is an agent with the following parts** (synthesis, not an established result):
  - (i) a large prior obtained cheaply by *distillation* from existing frontier models plus structured synthetic pre-pretraining. This is the evolution analogue; don't re-pay it.
  - (ii) a latent, action-conditioned predictive world model trained self-supervised on interactive streams.
  - (iii) a fast episodic memory with one-shot writes (BTSP-like), pattern-separated sparse keys, and retrieval during both inference and training.
  - (iv) an offline "sleep" phase that replays reward- and surprise-tagged episodes and consolidates into slow weights **only when held-out generalization improves** (go-CLS). It also uses replay constructively to compose new plans (Bakermans 2025), a bridge to test-time search.
  - (v) factorized structural codes reused across tasks.
  - (vi) learning-progress-driven data selection and interaction with teacher models.
  - (vii) multi-timescale value learning and uncertainty-gated plasticity.
  - (viii) MoE-style sparse conditional compute on near-memory hardware.
- **What the brain evidence says to de-prioritise:** replacing backprop with local rules (no sample-efficiency evidence at scale), and waiting for whole-brain emulation (2030s–2040s, $B-scale, structure-to-function unsolved).
- **Magnitude check (speculative):**
  - Measured gains of the included mechanisms are roughly:
    - world models 10–100x (in-domain data);
    - episodic/retrieval 2–25x (early learning, parameters);
    - replay and consolidation are qualitative (forgetting avoided);
    - sparse compute 3–7x (FLOPs);
    - structured pre-pretraining 1.4–1.6x;
    - curricula ~1–2x.
  - They are not independent and have never been combined. A plausible stacked range is 1e2–1e4x in samples or compute versus a plain dense LLM, short of the ~1e5 language-data gap.
  - The residual most likely lies in (i): the evolved prior and loss-function curriculum. That is exactly where neuroscience knowledge is thinnest, and where connectomics of the "steering subsystem" could eventually help.
- **Established vs hypothesis:**
  - Established: BTSP one-shot plasticity; SWR necessity for consolidation; dopamine RPE; sparse, energy-limited cortical activity; FlyWire and MICrONS data.
  - Hypothesis: go-CLS gating; compositional replay; TEM factorization; dopamine-as-learning-rate; "reward functions are the secret sauce"; connectome-to-algorithm shortcuts.

### Gaps
- No experiment was found that *combines* several of these mechanisms in one system and measures total sample or compute efficiency against a dense-LLM baseline at ≥1e21 FLOP.
- No quantitative decomposition of human learning efficiency into prior vs data vs memory vs social factors exists in the sources I found.
- Several 2025–2026 primary results could only be read as search snippets or titles because of proxy blocks and an exhausted search budget. These are the BTSP Nat Neurosci review, compartment-specific dendritic plasticity, and OpenScope predictive-processing tests. The report writer should treat those entries as provisional.
