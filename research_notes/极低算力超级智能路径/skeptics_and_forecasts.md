# Skeptics, Counter-case and the Expert/Forecasting Landscape: Is Low-Compute ASI Plausible, and Which Direction Do Leading Researchers Expect to Deliver It?

*Provenance note: in this session most web domains (dwarkesh.com, epoch.ai, arxiv.org, lesswrong.com, incompleteideas.net and others) were blocked for direct fetch. I read the primary Dwarkesh Podcast transcripts (Sutton 2025-09-26, Karpathy 2025-10-17, Sutskever 2025-11-25, Marblestone 2025-12-30, Dwarkesh's "Thoughts on Sutton" 2025-10-04, Hassabis 2024-02-28, Chollet 2024-06-11) from verbatim GitHub mirrors whose headers give the original URL and date. I extracted the Silver & Sutton "Era of Experience" PDF text directly from DeepMind's storage bucket. I read Byrnes' "Foom & Doom 1" from a verbatim clipping. Where a claim comes only from a search-engine snippet or a secondary synthesis, it is flagged "(secondary)".*

---

## 1. Sutton's "Bitter Lesson" (2019), how it was received, and Sutton's 2025 views: can "general methods that leverage computation" fit with low compute?

### Takeaway
The Bitter Lesson says methods should *scale with* computation. It does not say intelligence needs a lot of computation. By 2025 Sutton himself argued that LLMs are *not* a clean case of the Bitter Lesson, because they depend on finite human data. He expects them to be overtaken by agents that learn continually from their own experience. This view is compatible with far more *efficient* use of compute: Dwarkesh's steelman points out that today's LLMs spend most of their compute in deployment while learning nothing.

### Cited Findings
- **Original claim (13 Mar 2019):** "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin." The two methods it names as scaling arbitrarily are search and learning. The essay closes with "We want AI agents that can discover like we can, not which contain what we have discovered." — [Sutton, The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html). *(These are canonical, widely quoted lines. The page itself was blocked for live fetch in this session.)*
- **How it was used (reception):** Dwarkesh Patel (26 Sep 2025) said the essay is "perhaps the most influential essay in the history of AI" and that "people have used that as a justification for scaling up LLMs because… this is the one scalable way we have found to pour ungodly amounts of compute into learning about the world." — [Dwarkesh Podcast, Sutton episode](https://www.dwarkesh.com/p/richard-sutton)
- **Sutton on whether LLMs follow the Bitter Lesson (26 Sep 2025):** "They are clearly a way of using massive computation, things that will scale with computation up to the limits of the Internet. But they're also a way of putting in lots of human knowledge… Will they reach the limits of the data and be superseded by things that can get more data just from experience rather than from people? … I expect there to be systems that can learn from experience. Which could perform much better and be much more scalable. In which case, it will be another instance of the bitter lesson, that the things that used human knowledge were eventually superseded by things that just trained from experience and computation." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/richard-sutton)
- **Sutton's core critique of LLMs (same episode):** "Reinforcement learning is about understanding your world, whereas large language models are about mimicking people… They're not about figuring out what to do." "To mimic what people say is not really to build a model of the world at all. You're mimicking things that have a model of the world: people." "There's no goal… There's no ground truth. You can't have prior knowledge if you don't have ground truth." He rejects the idea that LLMs serve as a good "prior" for later RL, answering Dwarkesh's question with "No." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/richard-sutton)
- **Sutton's agent architecture ("base common model of the agent with the four parts"):** a policy, a value function learned with TD, perception/state construction, and "the transition model of the world… learned very richly from all the sensation that you receive, not just from the reward." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/richard-sutton)
- **Sutton on generalization as the unsolved core:** "We're not seeing transfer anywhere. Critical to good performance is that you can generalize well from one state to another state. We don't have any methods that are good at that." He also says catastrophic interference "is exactly bad generalization." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/richard-sutton)
- **"Big world hypothesis" argument for on-the-job learning:** "The reason why humans become useful on the job is because they are encountering their particular part of the world. It can't have been anticipated and can't all have been put in in advance. The world is so huge that you can't." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/richard-sutton)
- **"Weak methods have won":** "Learning and search have just won the day." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/richard-sutton)
- **Silver & Sutton, "Welcome to the Era of Experience" (Apr 2025, preprint chapter for MIT Press *Designing an Intelligence*):**
  - "A new generation of agents will acquire superhuman capabilities by learning predominantly from experience."
  - Imitation "has not and likely cannot achieve superhuman intelligence across many important topics," and "The pace of progress driven solely by supervised learning from human data is demonstrably slowing."
  - "Experience will become the dominant medium of improvement and ultimately dwarf the scale of human data."
  - Its example is AlphaProof: it was "initially exposed to around a hundred thousand formal proofs" and then "generated a hundred million more through continual interaction with a formal proving system."
  - It proposes streams of experience, grounded rewards, world models that predict "the consequences of the agent's actions," and "scalable planning methods."
  - It argues that "More efficient mechanisms of thought surely exist, using non-human languages that may… utilise symbolic, distributed, continuous, or differentiable computations."
  - It says RLHF-era LLMs "bypassed core RL concepts" (value functions, exploration, world models, temporal abstraction).
  - [Silver & Sutton, DeepMind PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
- **Dwarkesh's steelman of Sutton (4 Oct 2025), and the reconciliation with compute efficiency:**
  - "The bitter lesson says that you want to come up with techniques which most effectively and scalably leverage compute. Most of the compute spent on an LLM is used on running it in deployment. And yet it's not learning anything during this time… these models are usually trained on the equivalent of 10s of 1000s of years of human experience."
  - His own counter-view: "Imitation learning is continuous with and complementary to RL. Models of humans can give you a prior which facilitates learning 'true' world models."
  - He cites Ilya's "pretraining data = fossil fuels" analogy: you could not have gone from water wheels to solar panels without the cheap intermediary.
  - [Dwarkesh, "Some thoughts on the Sutton interview"](https://www.dwarkesh.com/p/thoughts-on-sutton)
- **Karpathy's response to Sutton (17 Oct 2025):** "We're not building animals. We're building ghosts… because we're not doing training by evolution. We're doing training by imitation of humans." He also says "That's why I call pre-training this crappy evolution. It's the practically possible version with our technology… to get to a starting point where we can do things like reinforcement learning." — [Dwarkesh Podcast, Karpathy](https://www.dwarkesh.com/p/andrej-karpathy)
- **Sutton's institutional bet:** he partnered with John Carmack's Keen Technologies (announced via Amii, 25 Sep 2023) as Chief Scientific Advisor. The goal is "developing a genuine AI prototype by 2030, including establishing, advancing and documenting AGI signs of life." — [secondary compilation citing Amii/X](https://github.com/coco-research/coco/blob/main/systems/superintelligence/engineering/research/john-carmack/notes.md)

### Inferences
- The Bitter Lesson is a claim about *scalability* (the slope of performance against compute) and about *not hand-coding knowledge*. It is not a claim that an absolute amount of compute is required. A low-compute ASI can still be "Bitter-Lesson-compliant" if it is a general learning-plus-search method that simply has a far better constant factor or exponent. Sutton's 2025 view explicitly places the next bitter-lesson winner in experience-driven continual RL, not in bigger human-data pretraining.
- Sutton and Dwarkesh's reconciliation is effectively an efficiency argument across the *whole pipeline*. An agent that keeps learning during deployment turns inference compute into learning compute, which removes the split between one huge pretraining run and serving that learns nothing.
- The Bitter Lesson also cuts *against* the most extreme low-compute hopes. Any method that wins will be used with as much compute as is available, so "low-compute ASI" most plausibly means "ASI-level capability reachable at low compute", not "the frontier will stay low-compute" (see §7, Jevons).

### Gaps
- I could not fetch the Bitter Lesson page live or the Alberta Plan (Sutton, Bowling, Pilarski, 2022). From background knowledge, the Alberta Plan stresses continual learning from ordinary experience and "cognizance of computational considerations", but this was not verified in-session. Sutton's 2025 "OaK" (Options and Knowledge) architecture talks could not be retrieved either.
- I found no published response from Sutton after Dwarkesh's Oct 2025 steelman.

---

## 2. The strongest case that general superintelligence *fundamentally* needs massive compute (the counter-case)

### Takeaway
The strongest counter-arguments are:
1. The human "algorithm" was itself bought with a huge amount of evolutionary search (Cotra's evolution anchor is about 1e41 FLOP), plus innate priors that are costly to rediscover.
2. Historically, most measured algorithmic progress has been *scale-dependent*: it only pays off, and can only be *identified*, at large compute.
3. RL from scratch without pretrained representations has repeatedly "burned a forest" of compute.
4. Neurons may do more computation than assumed.
5. Even if a cheap algorithm exists, economics (Jevons) pushes whoever has it to use more compute. So the *frontier* ASI would still be compute-heavy.

### Cited Findings
- **Evolution as the hidden pre-training bill:** Cotra's biological anchors report has an evolution anchor. It counts the total FLOP performed since the first neurons, assuming ~1e21 ancestors each running at nematode-level FLOP/s, and gives a median of about 1e41 FLOP. She weights it 10%. The report puts the brain at 1e13–1e16 FLOP/s (median 1e15). Her 2020 median for transformative AI was 2050, which she shortened by about 10 years in 2022. — [Epoch, "Grokking bioanchors"](https://epoch.ai/blog/grokking-bioanchors); [Scott Alexander, "What Happened With Bio Anchors?"](https://www.astralcodexten.com/p/what-happened-with-bio-anchors) (secondary summaries of Cotra)
- **Innate priors (Sutskever, 25 Nov 2025):** "Evolution has given us a small amount of the most useful information possible. For things like vision, hearing, and locomotion, I think there's a pretty strong case that evolution has given us a lot." — [Dwarkesh Podcast, Sutskever](https://www.dwarkesh.com/p/ilya-sutskever-2)
- **Neurons may be more computationally rich (Sutskever):** "There may be another blocker though, which is that there is a possibility that the human neurons do more compute than we think. If that is true, and if that plays an important role, then things might be more difficult." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/ilya-sutskever-2)
- **More compute still wins within a paradigm (Sutskever):** "If you want to build the absolutely best system then it helps to have much more compute. Especially if everyone is within the same paradigm, then compute becomes one of the big differentiators." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/ilya-sutskever-2)
- **RL from scratch is compute-hungry without representations (Karpathy, 17 Oct 2025):** "If you're just stumbling your way around and keyboard mashing… your reward is too sparse and you just won't learn. You're going to burn a forest computing, and you're never going to get something off the ground. What you're missing is this power of representation… You have to get the language model first." — [Dwarkesh Podcast, Karpathy](https://www.dwarkesh.com/p/andrej-karpathy)
- **Hassabis (28 Feb 2024):** "Theoretically, I think there's no reason why you couldn't go full AlphaZero-like on it… Having said that, I think by far the quickest way to get to AGI, and the most plausible way, is to use all the knowledge that's existing in the world right now… the final AGI system will have these large multimodal models as part of the overall solution, but they probably won't be enough on their own." — [Dwarkesh Podcast, Hassabis](https://www.dwarkesh.com/p/demis-hassabis)
- **Algorithmic progress has been mostly scale-dependent:**
  - Gundlach et al. (MIT FutureTech, Nov 2025) find that between 2017 and 2025 "the overwhelming majority of algorithmic progress can be accounted for by two scale-dependent innovations": LSTM to Transformer, and Kaplan to Chinchilla rebalancing. Together these account for about 91% of efficiency gains when extrapolated to the 2025 frontier.
  - Scale-invariant innovations contribute less than 10× overall.
  - [arXiv 2511.21622](https://arxiv.org/abs/2511.21622)
  - A secondary synthesis adds more detail from the paper: the Transformer's gain over LSTMs is about 6.28× at 1e15 FLOP but over 100× at frontier scale; post-2017 scale-invariant tricks combined are only about 1.33×; progress at small scales (~1e18 FLOP) is only about 20×, "actually below hardware progress rates"; and "limits to compute scaling pose obstacles not only to realizing efficiency gains but also to discovering them." — [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/algorithmic-efficiency-trends.md)
- **Compute has dominated LLM progress to date:** Ho et al. (Epoch, 2024) find 60–95% of LM performance gains from 2012–2023 came from compute and data scaling, with algorithms accounting for the rest. — [Epoch AI](https://epoch.ai/blog/algorithmic-progress-in-language-models)
- **Jevons effect:** Epoch argues that algorithmic progress "likely spurs more spending on compute, not less." The synthesis's summary of the piece: compute cost fell more than a trillion-fold between 1945 and 2006 while computing's share of GDP rose. — [Epoch Gradient Updates](https://epoch.ai/gradient-updates/algorithmic-progress-likely-spurs-more-spending-on-compute-not-less) (details via secondary synthesis)
- **Hinton flips the "brain is more efficient" intuition (EmTech Digital, MIT Tech Review, May 2023):** LLMs "have about a trillion connections and things like GPT-4 know much more than we do… probably know a thousand times as much as a person. But they've got a trillion connections and we've got a hundred trillion connections… I think it's because back propagation may be a much, much better learning algorithm than what we've got." — [Hinton EmTech 2023 transcript (GitHub mirror)](https://github.com/JimLiu/translations/tree/main/Geoffrey%20Hinton%20talks%20about%20the%20%E2%80%9Cexistential%20threat%E2%80%9D%20of%20AI)
- **Carmack's own caveat to "small code":** even if the AGI core fits "on this thumb drive… they're still gonna have to build the right data center to deploy it and have the right kind of life experience curriculum to take it up to the point where it's valuable." — [Lex Fridman #309, 4 Aug 2022](https://lexfridman.com/john-carmack/) (transcript via [GitHub mirror](https://github.com/adidor/AskExperts))

### Inferences
- The best form of the counter-case is not "intelligence needs 1e27 FLOP". It is "finding the efficient algorithm may need either (i) an evolution-like search whose cost we cannot cut far, or (ii) large-scale experiments, because the big wins only show at scale."
- Hinton's point splits "efficiency" in two. Backprop is more *parameter/knowledge*-efficient than brains, while brains are far more *sample*-efficient. A low-compute ASI would need both properties.
- Karpathy's "burn a forest" point implies that pure tabula-rasa experiential RL (Sutton's strongest version) may be the *most* compute-hungry route unless it comes with strong priors or representations. That would make an LLM or world-model prior plus continual learning the more compute-efficient hybrid.

### Gaps
- I found no rigorous estimate of how much evolutionary compute is *irreducible*, as opposed to wasteful. Critiques of the evolution anchor exist (e.g., [Nuño Sempere, 2022](https://nunosempere.com/blog/2022/08/10/evolutionary-anchor/)) but were not read in full.
- I found no quantitative 2025–2026 primary measurement of the "neurons do more compute" hypothesis.

---

## 3. The strongest case that it does *not* need massive compute (low-compute plausibility)

### Takeaway
The pro case rests on four pillars:
1. **Existence proof.** The human brain reaches general intelligence at about 20 W and roughly 1e15 FLOP/s, from about a lifetime of data. That shows a far more sample-efficient algorithm exists.
2. **Compressibility.** The genome is small, so the "core" (architecture plus learning rules plus reward/loss functions) must be compact (Carmack, Marblestone, Byrnes).
3. **History.** Paradigm-defining ideas were found at small compute (AlexNet on 2 GPUs; Transformer on 8–64 GPUs).
4. **Early data points.** Non-LLM systems show multi-order-of-magnitude efficiency on narrow tasks: Monty at about 1e11 FLOP, VL-JEPA with fewer parameters, and ARC-style program synthesis.

The most explicit low-compute ASI thesis is Steven Byrnes': "one consumer gaming GPU," even for training.

### Cited Findings
- **Sutskever (25 Nov 2025): human learning shows a better principle exists:**
  - "These models somehow just generalize dramatically worse than people. It's super obvious. That seems like a very fundamental thing."
  - "A human being, after even 15 years with a tiny fraction of the pre-training data, they know much less. But whatever they do know, they know much more deeply."
  - "Language, math, and coding… suggests that whatever it is that makes people good at learning is probably not so much a complicated prior, but something more, some fundamental thing."
  - On a teenager learning to drive in about 10 hours: "The fact that people are like that, I think it's a proof that it can be done… I do think it points to the existence of some machine learning principle that I have opinions on. But unfortunately, circumstances make it hard to discuss in detail."
  - [Dwarkesh Podcast](https://www.dwarkesh.com/p/ilya-sutskever-2)
- **Sutskever: research does not need frontier compute:** "AlexNet was built on two GPUs… The transformer was built on 8 to 64 GPUs. No single transformer paper experiment used more than 64 GPUs of 2017, which would be like, what, two GPUs of today?… for research, you definitely need some amount of compute, but it's far from obvious that you need the absolutely largest amount of compute ever for research." — [Dwarkesh Podcast](https://www.dwarkesh.com/p/ilya-sutskever-2)
- **Marblestone (30 Dec 2025): the brain's "secret sauce is its reward functions, not its architecture":**
  - "Machine learning tends to like mathematically simple loss functions… I think evolution may have built a lot of complexity into the loss functions… many different loss functions for different areas turned on at different stages of development. A lot of Python code, basically, generating a specific curriculum."
  - Dwarkesh sums it up: "in Python the reward function is literally a line. So you just have a thousand lines like this, and that doesn't take up that much space."
  - The episode notes the brain "runs at 200 hertz… on 20 watts."
  - [Dwarkesh Podcast, Marblestone](https://www.dwarkesh.com/p/adam-marblestone)
- **Carmack (4 Aug 2022):** "It is likely that the code for artificial general intelligence is going to be tens of thousands of lines of code not millions of lines of code… it's likely that the important things that we don't know are relatively simple… probably a handful of things." He also says "our brains are… maybe 50 megabytes" of genome. On timing: "I think there's a 50% chance we're gonna have signs of life of AGI" in about 8 years, i.e. around 2030. — [Lex Fridman #309](https://lexfridman.com/john-carmack/) (via [transcript mirror](https://github.com/adidor/AskExperts))
- **Byrnes, "Foom & Doom 1: 'Brain in a box in a basement'" (LessWrong, 23 Jun 2025):** the most explicit low-compute ASI thesis.
  - There is a "yet-to-be-discovered 'simple(ish) core of intelligence'", and the cortex is the existence proof: "100,000,000 repeating units… Nobody knows how it works."
  - "Human-level human-speed AGI will require not a data center, but rather something like one consumer gaming GPU—and not just for inference, but even for training from scratch."
  - On why current scaling curves do not apply: "different ML approaches can have different quantitative relationships between compute and performance… scaling laws for LSTMs and transformers… do not overlay… After the new paradigm, all bets are off."
  - R&D needed to go from "seemingly irrelevant" to ASI is "maybe 0–30 person-years."
  - Timelines: "probably 5 to 25 years."
  - On the "someone would have found it" objection, his analogies are the Riemann Hypothesis and Pearl's causal inference: elegant in hindsight but slow to discover.
  - [LessWrong post](https://www.lesswrong.com/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement) (read via verbatim clipping)
- **Karpathy's "cognitive core" (17 Oct 2025):**
  - "Figure out ways to remove some of the knowledge and to keep what I call this cognitive core. It's this intelligent entity that is stripped from knowledge but contains the algorithms and contains the magic of intelligence and problem-solving."
  - "I almost feel like we can get cognitive cores that are very good at even a billion parameters. If you talk to a billion parameter model, I think in 20 years, you can have a very productive conversation… if you ask it some factual question, it might have to look it up."
  - Current models "memorized way too much"; state-of-the-art model sizes "have gone up and now they've come down."
  - Pretraining is described as compressing "15 trillion tokens… to just your final neural network of a few billion parameters", which gives a "hazy recollection."
  - [Dwarkesh Podcast, Karpathy](https://www.dwarkesh.com/p/andrej-karpathy)
- **Hassabis (28 Feb 2024): world-model quality cuts inference-time search by orders of magnitude.** "The better your world model is, the more efficient your search can be." Deep Blue/Stockfish-style systems "look at millions of possible moves for every decision"; AlphaZero/AlphaGo "tens of thousands"; "a human grandmaster… probably only looks at a few hundred moves." — [Dwarkesh Podcast, Hassabis](https://www.dwarkesh.com/p/demis-hassabis)
- **Thousand Brains Project / Monty (Hawkins' Numenta spin-out; paper "Thousand Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference", Jul 2025 talk):**
  - Training FLOP "going up to 10^20, where Monty is all the way down at 10^11… The VIT, which did significantly worse… still uses orders of magnitude more flops… particularly staggering is the amount needed for the pre-training stage."
  - The task is 3D object recognition on the 77-object YCB set, with 14 rotations.
  - The authors claim "rapid, continual, and efficient learning… without needing entire GPU clusters."
  - [TBP talk transcript (GitHub mirror)](https://github.com/nunoatgithub/monty-video-transcription/blob/main/data/post-processed/2025_07_Thousand_Brains_Systems_Sensorimotor_Intelligence_for_Rapid_Robust_Learning_and_Inference.3d4DmnODLnE.cp.gpt-4.1.clean.md)
- **JEPA efficiency data point (LeCun camp):** Welch Labs' 2026 video on LeCun reports that Meta's VL-JEPA (late 2025), with the same encoder and data as a VLM, reached 35% video-classification accuracy after 5M examples versus 20% for the VLM. It also reportedly beat 7B-parameter models on GQA "while using just 1.6 billion parameters." — [Welch Labs, "Can Yann LeCun Reshape AI (again)?"](https://www.youtube.com/watch?v=v_jDvpEGTIg) (secondary description of a Meta paper)
- **Existence of small-compute innovation more generally:** Fogelson et al. (ICML 2025) catalogue 36 pre-training innovations in Llama 3 and DeepSeek-V3. About half "could have been developed" at GPT-2-level compute (~1e20 FLOP) or on 8 H100s. — [arXiv 2507.10618](https://arxiv.org/abs/2507.10618) (details via [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/theory-vs-scale-in-ai-progress.md))

### Inferences
- **Order-of-magnitude arithmetic (my own, from the cited figures):** brain ≈ 1e15 FLOP/s × ~1e9 s (about 30 years) ≈ 1e24 FLOP for a human "lifetime" of learning. That is roughly 1–2 orders of magnitude below today's frontier LLM training runs (≥1e26 FLOP; see §5) and is reached with about 20 W. If a brain-like algorithm ran at brain efficiency on digital hardware, human-level learning would sit within reach of a modest cluster. Byrnes argues it would need even less, because of digital speed and parallel copies.
- The evidence is lopsided. The *existence proof* is strong (brains exist). The *demonstrations* (Monty, VL-JEPA, ARC program synthesis) are all narrow. No non-LLM, low-compute system has yet shown broad general competence.
- Several sources converge on the same "compact core" idea from different directions: Carmack (tens of thousands of lines), Marblestone (a genome-sized set of reward/loss functions), Karpathy (a ~1B-parameter cognitive core) and Byrnes (a simple(ish) cortical algorithm).

### Gaps
- LeCun's frequently cited "four-year-old has seen ~1e14 bytes, as much as the largest LLM" arithmetic was referenced but not quoted verbatim in the sources I could read. The exact figure is unverified in-session.
- Hawkins' own 2025–2026 statements on AGI compute were not retrieved. Only the TBP team talk was.
- I could not verify human linguistic input (~1e8 words by adulthood) from a source in this session.

---

## 4. What named experts and labs said, 2023–2026, about the most promising path to AGI/ASI and about compute efficiency

### Takeaway
Across rival camps there is a striking 2025–2026 convergence that *pure pretraining scale-up is not enough*. What is named as missing:
- **Continual/online learning:** Sutton, Silver, Sutskever, Karpathy, Hassabis, Carmack.
- **Generalization/sample efficiency:** Sutskever, Chollet, Sutton.
- **World models plus planning:** LeCun, Hassabis, Sutton, Silver.
- **Program-like abstraction/test-time adaptation:** Chollet.
- **Better reward/value functions:** Sutskever, Marblestone, Karpathy.

Only a minority (Byrnes, and to a degree Carmack) explicitly predicts *low absolute compute*. Most expect "research plus big computers".

### Cited Findings
**Ilya Sutskever (SSI), Dwarkesh, 25 Nov 2025**
- "From 2012 to 2020, it was the age of research. Now, from 2020 to 2025, it was the age of scaling… Is the belief that if you just 100x the scale, everything would be transformed? I don't think that's true. So it's back to the age of research again, just with big computers." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- "At some point though, pre-training will run out of data. The data is very clearly finite." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- He concedes a data point on the other side: "it appears that Gemini have found a way to get more out of pre-training." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- "Scaling sucked out all the air in the room… we are in a world where there are more companies than ideas by quite a bit." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- Value functions: "the value function is something that's going to make RL more efficient… anything you can do with a value function, you can do without, just more slowly." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- Superintelligence as a learner: "a human being is not an AGI… we rely on continual learning… I produce a superintelligent 15-year-old that's very eager to go… the deployment itself will involve some kind of a learning trial-and-error period." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- Timeline to a system "which can learn as well as a human and subsequently… become superhuman": "5 to 20" years. — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- SSI compute: SSI raised "$3 billion"; other labs' big numbers are "earmarked for inference"; "we have sufficient compute to prove, to convince ourselves and anyone else, that what we are doing is correct." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)

**Andrej Karpathy, Dwarkesh, 17 Oct 2025**
- "It will take about a decade to work through all of those issues". Current agents "don't have continual learning… They're cognitively lacking." — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
- On RL: "you're sucking supervision through a straw… It's just stupid and crazy. A human would never do this." He expects "some major update to how we do algorithms for LLMs" in reflect-and-review, and says "we need three or four or five more." — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
- Missing sleep-like consolidation: "These models don't really have a distillation phase of taking what happened, analyzing it obsessively… and distilling it back into the weights… Maybe it's a LoRA." — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
- Model collapse limits synthetic self-training: "all of the samples you get from models are silently collapsed… ChatGPT… only has like three jokes." — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
- On future progress sources: "I do expect that nothing dominates. Everything plus 20%" (data, hardware, kernels, algorithms). ASI is "a progression of automation"; "we're in an intelligence explosion already and have been for decades." — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)

**Richard Sutton, Dwarkesh, 26 Sep 2025**
- LLMs are a "dead end"; the future is experience-based continual learning. See §1. — [Dwarkesh](https://www.dwarkesh.com/p/richard-sutton)

**David Silver & Sutton, Apr 2025**
- Era of Experience: streams, grounded rewards, world models, planning. See §1. — [PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)

**Yann LeCun (Meta, then AMI Labs)**
- Left Meta in Nov 2025 to found Advanced Machine Intelligence (AMI) Labs, building JEPA-based world models. He calls LLMs a "dead end" on the path to human-level intelligence. — [36Kr](https://eu.36kr.com/en/p/3571987975018880); [Let's Data Science](https://letsdatascience.com/blog/yann-lecun-told-meta-he-could-do-it-faster-alone-then-he-raised-1-billion) (secondary)
- AMI Labs reportedly raised a $1.03B seed at a $3.5B pre-money valuation, announced 10 Mar 2026. — [Let's Data Science](https://letsdatascience.com/blog/yann-lecun-told-meta-he-could-do-it-faster-alone-then-he-raised-1-billion) (secondary; figures vary across outlets, one headline says "$30M")
- In a 2026 interview: "I do not understand how you can even think of building an agentic system without… the ability of predicting the consequences of its actions." VLAs "are doomed… the only way to get them to work is to essentially collect tons and tons and tons of… examples… when they face a… slightly new situation are completely helpless." He argues for hierarchical world models: "your cat can do hierarchical planning… they don't have language." — [Welch Labs interview](https://www.youtube.com/watch?v=v_jDvpEGTIg)

**Demis Hassabis (Google DeepMind)**
- 2024: LLMs as prior plus AlphaZero-style planning and search on top; a better world model means more efficient search (see §2–3). — [Dwarkesh, 28 Feb 2024](https://www.dwarkesh.com/p/demis-hassabis)
- Reported 2025–2026 statements (secondary):
  - "We're only one or two AlphaGo-level technological breakthroughs away from AGI." The breakthroughs are "along the lines of… continual learning, better memory, longer context windows… and better long-term reasoning and planning."
  - Models "still don't have the ability to continually learn. They have goldfish brain."
  - About 50% probability of AGI by 2030.
  - [36Kr](https://eu.36kr.com/en/p/3584784831797634); [Big Technology](https://www.bigtechnology.com/p/google-deepmind-ceo-demis-hassabis-946); [EA Forum summary](https://forum.effectivealtruism.org/posts/YvFjpAKkJNErkiFTN/google-deepmind-ceo-demis-hassabis-on-what-s-still-needed)
  - Exact dates of these quotes were not verified.

**François Chollet (ARC Prize; Ndea, founded Jan 2025 with Mike Knoop)**
- 11 Jun 2024: "General intelligence is the ability to approach any problem, any skill, and very quickly master it using very little data… This fundamentally requires the ability to adapt, to learn on the fly efficiently." — [Dwarkesh](https://www.dwarkesh.com/p/francois-chollet)
- Same episode: "LLMs are basically this big interpolative memory." And: "OpenAI basically set back progress towards AGI by quite a few years, probably like 5-10 years… I see LLMs as more of an off-ramp on the path to AGI." — [Dwarkesh](https://www.dwarkesh.com/p/francois-chollet)
- Same episode: the approaches working on ARC are "discrete program search, program synthesis." — [Dwarkesh](https://www.dwarkesh.com/p/francois-chollet)
- 2025: the "pre-training scaling era" (2020–2024) confused performance on known tasks with intelligence. 2024 marked a shift to "test-time adaptation" (program synthesis, chain-of-thought synthesis). — [The Decoder](https://the-decoder.com/francois-chollet-on-the-end-of-scaling-arc-3-and-his-path-to-agi/) (secondary)
- Timelines: "AGI by 2030, early 2030s, most likely." — [OfficeChai](https://officechai.com/ai/agi-likely-by-early-2030s-when-arc-agi-6-or-7-will-be-released-arc-prizes-francois-chollet/)
- 2026, on ARC-AGI-3 (the interactive benchmark launched in 2026): "like all high-performing approaches on ARC-AGI-3, it uses deep learning-guided on-the-fly synthesis of symbolic world models, i.e. navigating the world by generating programs to represent what you know." — [Chollet on X](https://x.com/fchollet/status/2090838046937645398); [ARC-AGI-3 paper](https://arxiv.org/pdf/2603.24621)

**Geoffrey Hinton**
- May 2023: backprop "may be a much, much better learning algorithm than what we've got". Digital models pack more knowledge per connection. See §2. — [EmTech 2023 transcript mirror](https://github.com/JimLiu/translations)

**John Carmack (Keen Technologies)**
- 2022: AGI core is "tens of thousands of lines of code"; 50% chance of "signs of life" around 2030. — [Lex Fridman #309](https://lexfridman.com/john-carmack/)
- Sep 2023: Sutton joins Keen. Carmack's quote: "The AI space is awash in capital, compute, and data, but it is still dominated by fashions that may yet hinder important breakthroughs." — [secondary compilation](https://github.com/coco-research/coco/blob/main/systems/superintelligence/engineering/research/john-carmack/notes.md)
- Upper Bound talk, May 2025: a physical Atari robot (camera plus robotic joystick, real-time, no turn-based waiting). His team reached reasonable performance "using effectively no replay-buffer, no batch updates and no target networks" (streaming deep RL). He argues RL must be grounded in messy real-time reality. — [Amii video page](https://www.amii.ca/videos/keen-technologies-research-directions-john-carmack-upper-bound-2025); [HN discussion](https://news.ycombinator.com/item?id=44070042) (via search summary)
- Nov 2025: "We're trying to learn fundamental things about architecture and learning that nobody knows right now." — [D Magazine](https://www.dmagazine.com/business-economy/2025/11/conversation-with-john-carmack-keen-technologies/) (via secondary compilation)

**Gary Marcus**
- Jan 2025 self-review: his 2022 prediction that "pure scaling of LLMs would eventually start to run out" "appear[s] to have been confirmed in November and December [2024]". Among his 2025 predictions: "Neurosymbolic AI will become much more prominent." — [Marcus on AI](https://garymarcus.substack.com/p/25-ai-predictions-for-2025-from-marcus)

**Adam Marblestone (Convergent Research; ex-DeepMind neuroscience)**
- Dec 2025: the path runs through brain-like reward/loss-function design and model-based RL. He says his view is "extremely similar to what Yann LeCun would say" about energy-based, omnidirectional inference. — [Dwarkesh](https://www.dwarkesh.com/p/adam-marblestone)

**Steven Byrnes**
- Jun 2025: brain-like AGI; training on a consumer GPU; LLMs will not scale to ASI. — [LessWrong](https://www.lesswrong.com/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement)

### Inferences
- **Near-consensus on what is missing:** continual learning and generalization/sample efficiency. Sutton, Silver, Sutskever, Karpathy, Hassabis, Chollet, LeCun and Carmack all name one or both, even though they disagree on whether LLMs are the base.
- **The key split is "LLM as prior plus a new learning layer"** (Hassabis, Karpathy, Dwarkesh, possibly Sutskever) **versus "a new substrate from scratch"** (Sutton's pure experiential RL, LeCun's JEPA world models, Chollet's program synthesis, Byrnes and Hawkins' brain-like approach). The second group is where the strongest *low-compute* claims live.
- **Sutskever is the most important mainstream voice** saying that proving the next paradigm does not need frontier compute. He is also the most secretive about what the "machine learning principle" is.

### Gaps
- **Bengio:** I found no 2023–2026 statement specifically on compute-efficient paths to AGI. His recent focus is "Scientist AI" and safety; this was not retrieved.
- **Schmidhuber:** no 2023–2026 primary statement was retrieved (web search budget exhausted).
- **Hawkins:** only the TBP team's July 2025 talk was retrieved, not Hawkins' own quotes.
- **Hassabis 2025–2026 quotes:** these came through secondary outlets, and exact dates and venues were not verified.
- **Sutskever's "machine learning principle":** he explicitly declined to disclose it. No public SSI technical output was found.

---

## 5. Empirical evidence: are returns to scaling compute still coming (2025–2026), and how much progress comes from algorithms versus compute?

### Takeaway
Physical compute still grows about 4–5× per year, and algorithmic efficiency about 3× per year (a 2–6× range). Historically, compute and data explain most LLM gains, and almost all algorithmic gains came from two *scale-dependent* innovations.

Qualitative 2024–2025 evidence (insiders, Sutskever, Silver & Sutton, Marcus) points to diminishing returns from *pretraining on human data specifically*. Epoch still projects that compute scaling can continue to about 2e29 FLOP by 2030.

### Cited Findings
- **Algorithmic efficiency:** "the level of compute needed to achieve a given level of performance has halved roughly every 8 months (95% CI 5–14 months)" in LMs from 2012–2023. Shapley analysis says 60–95% of gains came from compute scaling. — [Epoch AI, Ho et al. 2024](https://epoch.ai/blog/algorithmic-progress-in-language-models); [arXiv 2403.05812](https://arxiv.org/abs/2403.05812)
- **Broader and updated estimates:** algorithmic progress cuts compute needed by 2–6× per year. Across domains, "every nine months, the introduction of better algorithms contributes the equivalent of a doubling of compute budgets (4 to 25 months)". Frontier training compute grows 4–5× per year. — [Epoch trends](https://epoch.ai/trends); [Epoch, "Revisiting algorithmic progress"](https://epoch.ai/publications/revisiting-algorithmic-progress) (via search summary)
- **Epoch dashboard (early 2026, secondary):** pre-training compute efficiency is improving about 3.0× per year (doubling about every 7.6 months). Frontier training compute grew about 5.3× per year from 2010–2024. Grok-4 (2025) is estimated at about 5e26 FLOP, and GPT-4 at about 2e25. — [secondary synthesis of Epoch data](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/algorithmic-efficiency-trends.md)
- **Scale-dependence:** two innovations (Transformers; Chinchilla rebalancing) account for about 91% of efficiency gains extrapolated to the 2025 frontier. Scale-invariant innovations each give small gains, less than 10× combined. — [Gundlach et al., arXiv 2511.21622](https://arxiv.org/abs/2511.21622)
- **Epoch 2030 projection:** the historical rate of compute scaling "can likely be sustained until at least 2030". That would reach about 2e29 FLOP per frontier run and clusters costing over $100B. — [Epoch, "Can AI scaling continue through 2030?"](https://epoch.ai/blog/can-ai-scaling-continue-through-2030) (via secondary synthesis)
- **Diminishing returns to human-data pretraining:**
  - Silver & Sutton (Apr 2025): "The pace of progress driven solely by supervised learning from human data is demonstrably slowing." — [PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
  - Sutskever (Nov 2025): 100× more scale would not transform everything, and data "is very clearly finite". — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
  - Marcus cites Nov–Dec 2024 reporting (The Verge, TechCrunch: "AI scaling laws are showing diminishing returns, forcing AI labs to change course"; WSJ on GPT-5 "Orion" delays). — [Marcus on AI](https://garymarcus.substack.com/p/25-ai-predictions-for-2025-from-marcus)
- **Counter-evidence (returns continue):**
  - Sutskever notes "Gemini have found a way to get more out of pre-training". — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
  - Karpathy expects steady broad-based gains ("everything plus 20%") rather than a wall. — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
  - The Metaculus community did lengthen AGI timelines in 2025–2026 (see §6), which suggests some perceived slowdown.
- **Small model efficiency trend (secondary):** Stanford HAI AI Index 2025 reports that the smallest model scoring >60% on MMLU fell from PaLM (540B, 2022) to Phi-3-mini (3.8B, 2024), a 142× parameter reduction. Inference cost for GPT-3.5-level performance fell about 280× between Nov 2022 and Oct 2024. — [Stanford HAI AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report) (via secondary synthesis)

### Inferences
- **Moving from a fixed capability to the frontier:** 3× per year in algorithmic efficiency compounds to roughly 1,000× per ~6 years for a *fixed* capability. So any given capability level gets cheap fast. This is the "trailing-edge low-compute" path. But the *frontier* has been set by compute growth plus scale-dependent tricks.
- **The historical record does not favour low-compute ASI.** Past efficiency gains mostly *required* scale to find. A low-compute ASI would need a qualitatively different algorithm class, one whose scaling curve sits orders of magnitude to the left. That is Byrnes' "curves do not overlay" argument.
- **Whether a "wall" exists is contested.** The better-supported claim is narrower: returns from *human-data pretraining* are falling, and the frontier is shifting to RL, test-time compute and experience. That shift is exactly what Silver, Sutton and Sutskever predict.

### Gaps
- I did not retrieve quantitative 2025–2026 primary data on returns, e.g., GPT-4.5's compute-versus-benchmark gains, Gemini 3 pretraining ablations, or METR time-horizon doubling (web search budget exhausted). One related paper title surfaced: ["Forecasting AI Time Horizon Under Compute Slowdowns" (arXiv 2511.19492)](https://arxiv.org/pdf/2511.19492).
- Epoch's 2026 dashboard figures are taken from a secondary synthesis dated Mar 2026, not read directly.

---

## 6. Forecasts and minimum-compute analyses

### Takeaway
Aggregate forecasts put AGI or HLMI in the 2030s (Metaculus median about 2033) or 2040s (academic survey: 50% by 2047). Leading researchers cluster at "about 2030" (Hassabis, Chollet, Carmack's "signs of life") to "5–20 years" (Sutskever, Byrnes 5–25, Karpathy about a decade).

Compute-anchored models (Cotra) implicitly assume ML must spend large amounts of compute. Their anchors span a lifetime anchor near the brain (~1e24 FLOP by arithmetic) up to the evolution anchor (~1e41). Algorithmic progress shifts these anchors down over time.

### Cited Findings
- **AI Impacts 2023 ESPAI (survey Oct 2023, 2,778 researchers):** aggregate 50% chance of HLMI by 2047, 13 years earlier than the 2022 survey (2060). 50% chance of full automation of labor by 2116, 48 years earlier than 2164. — [AI Impacts wiki](https://wiki.aiimpacts.org/ai_timelines/predictions_of_human-level_ai_timelines/ai_timeline_surveys/2023_expert_survey_on_progress_in_ai); [Grace et al., "Thousands of AI Authors on the Future of AI"](https://arxiv.org/html/2401.02843v3)
- **Metaculus (secondary aggregators, 2026):**
  - The community median for the first "general AI system" is about Jan 2033. The Feb 2026 update puts 25% by 2029 and 50% by 2033.
  - These probabilities include a robotics requirement. Removing it reportedly shifts estimates 2–3 years earlier.
  - [AIToolsReview, Sep 2026](https://aitoolsreview.co.uk/insights/agi-timeline-predictions-2026); [Metaculus notebook "AI Forecasting in 2026"](https://www.metaculus.com/notebooks/43363/ai-forecasting-in-2026/)
- **Direction of updates:** from 2025 to 2026, Kokotajlo and Lifland (AI Futures Project), the Metaculus community, Dario Amodei and Peter Wildeford all pushed their timelines *out*. — [FutureSearch AGI timeline tracker](https://futuresearch.ai/blog/agi-timeline-tracker/) (search summary)
- **Individual researchers:**
  - Sutskever: "5 to 20" years to a human-level learner that becomes superhuman (Nov 2025). — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
  - Karpathy: about a decade (Oct 2025). — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
  - Chollet: "AGI by 2030, early 2030s". — [OfficeChai](https://officechai.com/ai/agi-likely-by-early-2030s-when-arc-agi-6-or-7-will-be-released-arc-prizes-francois-chollet/)
  - Hassabis: about 50% by 2030 (secondary). — [36Kr](https://eu.36kr.com/en/p/3584784831797634)
  - Byrnes: "probably 5 to 25 years" for the new paradigm. — [LessWrong](https://www.lesswrong.com/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement)
  - Carmack (2022): 50% "signs of life" around 2030. — [Lex Fridman #309](https://lexfridman.com/john-carmack/)
- **Cotra's biological anchors:**
  - Evolution anchor about 1e41 FLOP (10% weight). Brain 1e15 FLOP/s (range 1e13–1e16).
  - 2020 median for transformative AI was 2050, shortened by about 10 years in 2022.
  - The framework adds algorithmic progress that lowers required FLOP over time.
  - [Epoch, "Grokking bioanchors"](https://epoch.ai/blog/grokking-bioanchors); [ACX retrospective](https://www.astralcodexten.com/p/what-happened-with-bio-anchors)
- **Minimum-compute outliers:** Byrnes argues that one consumer GPU would suffice for training *and* inference of human-level, human-speed AGI in a brain-like paradigm. — [LessWrong](https://www.lesswrong.com/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement)

### Inferences
- **The lifetime anchor on its own numbers (my arithmetic):** 1e15 FLOP/s × ~1e9 s ≈ 1e24 FLOP. That is the natural "low-compute" target. It is already exceeded by frontier training runs by about 100×, and yet it does not produce human-like learning. This supports Sutskever and Chollet: the bottleneck is the *algorithm* (generalization), not raw FLOP.
- **Few timelines depend on the low-compute route.** Mainstream timelines (Metaculus, Hassabis) mostly assume the LLM-plus-scaling route with add-ons. Forecasts conditioned on a new low-compute paradigm (Byrnes) have much wider variance and sharper takeoff.

### Gaps
- Epoch's "direct approach" and Davidson's compute-centric takeoff model could not be read (domains blocked, search budget exhausted), so their specific minimum-compute figures are not reported.
- I have no updated AI Impacts survey (2024/2025 ESPAI) figures in hand.
- Metaculus numbers come from secondary aggregators; the question page was not read directly.

---

## 7. Implications: compute overhang, the speed of an "intelligence explosion", and compute governance

### Takeaway
If ASI-level capability turns out to need little compute, the world would carry a large *compute overhang*. The existing installed base of frontier datacenters, built for LLM training runs of 1e26 FLOP and up, could immediately run and train enormous numbers of such systems. That favours sharp, local takeoff (Byrnes; Forethought's software intelligence explosion work) and undermines compute-threshold governance. Fogelson et al. find that even an 8-H100 cap would block only about half of past innovations. Epoch's counterpoint: efficiency gains historically *increase* compute demand, and the big gains have been scale-dependent. Compute would therefore remain a lever, but a weaker one.

### Cited Findings
- **Byrnes (Jun 2025):**
  - "Once the new paradigm is known… the actors able to train ASI from scratch will probably number in the tens of thousands, spread all around the world… if governments know where all the giant data centers are… it's only marginally helpful."
  - He expects a "very sharp takeoff in wall-clock time", with 0–2 years from "seemingly irrelevant" to ASI, and possibly "a single training run" crossing both thresholds.
  - He mentions a "neuroscience overhang" of 100,000 papers that could be integrated quickly.
  - Researchers would use "ten H100s or whatever… very inexpensive, widely available, and all-but-impossible to track or govern."
  - [LessWrong](https://www.lesswrong.com/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement)
- **Forethought (Davidson, Eth, 2025):**
  - Automating AI R&D could trigger a "software intelligence explosion". This "will probably (~60%) compress >3 years of AI progress into <1 year, but is somewhat unlikely (~20%) to compress >10 years into <1 year."
  - A companion paper asks whether compute bottlenecks would prevent it.
  - Davidson's newsletter post "Plan A's problem with dry tinder" appears to discuss accumulated overhang (title only; not read).
  - [Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be); [arXiv 2507.23181](https://arxiv.org/pdf/2507.23181); [Forethought newsletter](https://newsletter.forethought.org/p/plan-as-problem-with-dry-tinder)
- **Fogelson et al. (ICML 2025):** "Even very restrictive caps—such as capping total operations to the compute used to train GPT-2 or capping hardware capacity to 8 H100 GPUs—would both only disallow about half of the cataloged innovations." Compute requirements of the more compute-intensive innovations are growing about 2.5× per year. — [arXiv 2507.10618](https://arxiv.org/abs/2507.10618) (quote via [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/theory-vs-scale-in-ai-progress.md))
- **Epoch on Jevons:** algorithmic progress "likely spurs more spending on compute, not less." — [Epoch](https://epoch.ai/gradient-updates/algorithmic-progress-likely-spurs-more-spending-on-compute-not-less)
- **Sutskever on how differentiation might work:** within one paradigm "compute becomes one of the big differentiators." If "the correct solution does emerge", he expects "eventual convergence on the technical approach." — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
- **Carmack's "thumb drive" framing (2022):** "we cracked the secret to AGI and it fits on this thumb drive and anyone could steal it." — [Lex Fridman #309](https://lexfridman.com/john-carmack/)
- **Gundlach et al. (secondary quote):** compute limits obstruct "not only… realizing efficiency gains but also… discovering them." This supports compute governance as a brake on *discovery*. — [secondary synthesis](https://github.com/JoernStoehler/xrisk-pause-game/blob/main/literature/algorithmic-efficiency-trends.md) summarizing [arXiv 2511.21622](https://arxiv.org/abs/2511.21622)

### Inferences
- **The governance consequence is asymmetric.**
  - If the efficient algorithm is *scale-dependent to discover* (the Gundlach pattern), compute governance slows its discovery.
  - If it is *discoverable small and then scales* (the Transformer pattern, and Byrnes' and Sutskever's framing), compute governance mainly limits *deployment scale*. Deployment on a large installed base could then be very fast (overhang).
- **A low-compute ASI does not mean a low-compute world.** A cheap algorithm would, by Jevons and the Bitter Lesson, be run with all available compute. Whoever controls large compute when the algorithm appears would get outsized advantage (the overhang), even though a lone actor could also build a basic version.

### Gaps
- I could not read AI Impacts' or Epoch's dedicated "hardware overhang" analyses, or Pilz and Heim on the effects of compute efficiency (search budget exhausted). Their quantitative framing of "access effect vs performance effect" is not included.

---

## 8. Convergence: which specific direction do the most credible sources expect to yield multi-order-of-magnitude compute-efficiency gains toward general intelligence?

### Takeaway
The clearest convergence among otherwise disagreeing experts is on:

**agents that learn continually and online from their own experience (grounded feedback), using learned world models for planning, with strong generalization/sample efficiency,**

together with **separating a compact reasoning or learning "core" from stored knowledge** and **program-like abstraction at test time.**

Proposed mechanisms differ:
- TD/value-function RL (Sutton, Silver, Sutskever)
- JEPA latent world models (LeCun)
- program synthesis (Chollet)
- a small "cognitive core" plus retrieval and sleep-like distillation (Karpathy)
- brain-like reward/loss functions and cortical algorithms (Marblestone, Byrnes, Hawkins)

Mainstream labs mostly expect these to be built *on top of* LLM priors. The low-compute maximalists expect a from-scratch brain-like learner.

### Cited Findings
- **Continual/online learning is named as missing** by:
  - Sutton ("no special training phase"). — [Dwarkesh](https://www.dwarkesh.com/p/richard-sutton)
  - Silver & Sutton ("streams of experience"). — [PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
  - Sutskever ("we rely on continual learning… superintelligent 15-year-old"). — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
  - Karpathy ("they don't have continual learning"; missing a sleep-like distillation). — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
  - Hassabis ("goldfish brain"; continual learning among the "one or two breakthroughs"). — [36Kr](https://eu.36kr.com/en/p/3584784831797634)
  - Carmack (streaming RL). — [Amii](https://www.amii.ca/videos/keen-technologies-research-directions-john-carmack-upper-bound-2025)
  - The Thousand Brains Project ("rapid, continual, and efficient learning"). — [TBP transcript](https://github.com/nunoatgithub/monty-video-transcription/blob/main/data/post-processed/2025_07_Thousand_Brains_Systems_Sensorimotor_Intelligence_for_Rapid_Robust_Learning_and_Inference.3d4DmnODLnE.cp.gpt-4.1.clean.md)
- **Generalization/sample efficiency is named as the crux** by:
  - Sutskever ("the most fundamental"). — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
  - Chollet (intelligence means "very quickly master it using very little data"). — [Dwarkesh](https://www.dwarkesh.com/p/francois-chollet)
  - Sutton ("We're not seeing transfer anywhere"). — [Dwarkesh](https://www.dwarkesh.com/p/richard-sutton)
- **World models plus planning are named** by:
  - Sutton (transition model). — [Dwarkesh](https://www.dwarkesh.com/p/richard-sutton)
  - Silver & Sutton ("build a world model… plan"). — [PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
  - LeCun ("predicting the consequences of its actions"). — [Welch Labs](https://www.youtube.com/watch?v=v_jDvpEGTIg)
  - Hassabis ("better world model, more efficient search"). — [Dwarkesh](https://www.dwarkesh.com/p/demis-hassabis)
  - Chollet (ARC-AGI-3 winners synthesize "symbolic world models"). — [X](https://x.com/fchollet/status/2090838046937645398)
- **Compact core over memorized knowledge:**
  - Karpathy (~1B-parameter cognitive core). — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
  - Carmack (tens of thousands of lines of code). — [Lex Fridman](https://lexfridman.com/john-carmack/)
  - Marblestone (a genome-sized set of loss functions). — [Dwarkesh](https://www.dwarkesh.com/p/adam-marblestone)
  - Byrnes ("simple(ish) core"). — [LessWrong](https://www.lesswrong.com/posts/yew6zFWAKG4AGs3Wk/foom-and-doom-1-brain-in-a-box-in-a-basement)
- **Better learning signals than outcome-only RL:**
  - Karpathy ("straw"; reflect-and-review). — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
  - Sutskever (value functions and emotions make RL more efficient). — [Dwarkesh](https://www.dwarkesh.com/p/ilya-sutskever-2)
  - Marblestone (rich innate reward functions). — [Dwarkesh](https://www.dwarkesh.com/p/adam-marblestone)
  - Silver & Sutton (grounded rewards; revisit value functions). — [PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
- **The dissent on starting from scratch:**
  - Hassabis: using web knowledge as a prior is "by far the quickest way". — [Dwarkesh](https://www.dwarkesh.com/p/demis-hassabis)
  - Karpathy: pure RL from scratch will "burn a forest". — [Dwarkesh](https://www.dwarkesh.com/p/andrej-karpathy)
  - Dwarkesh: imitation is complementary to RL (the fossil-fuel analogy). — [Dwarkesh](https://www.dwarkesh.com/p/thoughts-on-sutton)

### Inferences
- **The most defensible candidate for multi-order-of-magnitude efficiency across the *whole pipeline*** is **model-based, continual (online) learning from grounded experience.** Its components are:
  - a learned (latent/abstract) world model plus value function for planning. Hassabis' search-count ladder (millions to tens of thousands to hundreds) shows how world-model quality cuts inference compute by orders of magnitude.
  - learning during deployment. This removes the "train once, serve frozen" split, which is where Dwarkesh locates most wasted compute.
  - a compact reasoning core separated from knowledge (retrieval or memory). This cuts both parameter count and inference cost (Karpathy's ~1B core).
  - richer, brain-like learning signals (value functions, dense or intrinsic rewards). These fix RL's "straw" sample inefficiency.
- **Program synthesis and symbolic world models (Chollet)** look like the best candidate for the *abstraction and generalization* component on novel-task benchmarks. They may join the above as a module rather than compete with it.
- **The honest bottom line for the report:** most credible experts believe (a) a far more sample- and compute-efficient algorithm exists (brain existence proof; Sutskever: "proof that it can be done") and (b) finding it is a *research* problem, not a scaling problem. Only a minority (Byrnes, Carmack in spirit) predict *absolutely* low compute, e.g., a single GPU. The majority expect the winning method to be run on big computers anyway (Sutskever: "back to the age of research… just with big computers").

### Gaps
- There is no quantitative, peer-reviewed estimate of the achievable efficiency multiplier for any of these directions at general-intelligence scale. The existing numbers (Monty ~1e11 FLOP; VL-JEPA 1.6B beating 7B models; ARC program synthesis) are narrow-domain.
- The exact technical content of SSI's, Keen's, AMI Labs' and Ndea's approaches is undisclosed.
