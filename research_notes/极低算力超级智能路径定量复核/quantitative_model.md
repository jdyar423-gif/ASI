# Quantitative Re-check: An Explicit Monte Carlo Model of Low-Compute General Superintelligence (as of Sept 2026)

*Scope and status.* This note turns the evidence from rounds 1–2 into a runnable, transparent Monte Carlo model, which I wrote and ran myself. The model is at [/home/user/ASI/models/low_compute_asi_model.py](/home/user/ASI/models/low_compute_asi_model.py). Invoke it with `python3 low_compute_asi_model.py`. It uses a fixed seed (20260926) and numpy 2.4.6 when available. It falls back automatically to stdlib `random`, which can also be forced with `--no-numpy`, and `--selftest` checks that both engines give identical results on identical inputs; the self-test passed. A full run takes about 24 s. The question being served is 以极低算力（全程：训练+推理+借来的算力）实现通用超级智能的最可能方向是什么？为什么？ The ledger convention is round 2's: full-pipeline = own training + inference + all borrowed teacher/base/distillation/synthetic compute; discovery compute is kept in a separate ledger; only a ≤ genome-sized description may cross between ledgers ([round-2 report](/home/user/ASI/reports/极低算力超级智能路径深化.md)).

*Source status.* metaculus.com, lesswrong.com, blog.aifutures.org and agi.goodheartlabs.com were blocked by the egress proxy in this session. Every 2026 external forecast below therefore comes from search-engine snippets and is marked **(snippet)**. Evidence inherited from round 1 and round 2 is cited to the round-2 notes and to the primary URLs those notes give. **Every probability in this note is a model output conditional on the stated input distributions. None is a fact.** Where a model number is quoted to one decimal place, that shows Monte Carlo resolution (standard error ±0.06 pp), not real precision. Real uncertainty spans several-fold (see Q3 and Q6).

## Q1. Model structure: what the inputs are, where each comes from, and how they combine

### Takeaway
The model has 19 explicit input distributions and three routes to ASI. Route F is the big-compute frontier ASI. Path A is a compact general learner, discovered by humans or by AI, that is then trained from scratch. Path B re-derives the frontier paradigm from scratch at small scale after frontier ASI exists; this is "big-then-small without distillation". All three routes share a common **ASI floor**, f = log C_brain + log M_asi − log E_max, which is the lowest full-pipeline compute at which *any* learner could reach ASI. Crossing times are solved analytically; there is no time grid. Distillation from a big ASI is excluded by construction, because the ledger charges the teacher's compute to the student.

### Cited Findings
**Inputs for the floor f**
- **C_brain** (brain-equivalent lifetime compute): log10 ~ N(24.0, 1.0), so p10–p90 is 10^22.7–10^25.3. Parameters come from the following sources.
  - Carlsmith judges it "more likely than not that 1e15 FLOP/s is enough", with a mechanistic range of 1e13–1e17 FLOP/s. — [Coefficient Giving / Open Phil](https://coefficientgiving.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/)
  - Cotra's lifetime anchor has a median of ~1e24 FLOP; the evolution anchor is ~1e41. — [Epoch, Grokking bio-anchors](https://epoch.ai/blog/grokking-bioanchors)
  - 30 years ≈ 9.5e8 s, which gives 1e22–1e26. — [theory_limits.md Q1](/home/user/ASI/research_notes/极低算力超级智能路径/theory_limits.md)
- **E_max** (efficiency of the best possible learner relative to the brain's algorithm): log10 ~ N(0.5, 1.0), so the median is ~3× and p10–p90 is 0.17×–60×. *Assumption*, bracketed by the evidence below.
  - Narrow systems are already 3–6 OOM below the brain anchor: KataGo reached superhuman Go at ~1e21 full-pipeline; EfficientZero reached human-level Atari-100k at ~1e18 per game. — [compute_ledger.md Q4](/home/user/ASI/research_notes/极低算力超级智能路径深化/compute_ledger.md); [KataGo paper text](https://github.com/FoAKTEE/az/blob/main/ref-paper/arxiv-1902.10565/src/Accelerating_Self_Play_Learning_In_Go_2020.tex)
  - No general, LLM-class system has come in below ~3.3e24 without borrowing (DeepSeek-V3). — [DeepSeek-V3 README](https://github.com/deepseek-ai/DeepSeek-V3); [compute_ledger.md Q4](/home/user/ASI/research_notes/极低算力超级智能路径深化/compute_ledger.md)
  - Genome-style compressed priors "did not change learning trajectories"; they only raised the starting point. — [Shuvaev et al., PNAS](https://www.pnas.org/doi/10.1073/pnas.2409160121) (via [compressed_priors.md §6](/home/user/ASI/research_notes/极低算力超级智能路径深化/compressed_priors.md))
  - General-domain prior multipliers measured so far are 1.3–10×. — [compressed_priors.md §6](/home/user/ASI/research_notes/极低算力超级智能路径深化/compressed_priors.md)
- **M_asi** (the multiplier over one human lifetime needed for superhuman breadth and depth, with lifetime inference also charged): log10 ~ N(1.0, 0.5), so the median is 10× and p10–p90 is 2.3×–44×. *Assumption*. Evidence:
  - Round 2 says ≤1e24 ASI requires "superhuman breadth without proportionally more experience", for which there is no evidence. — [round-2 report](/home/user/ASI/reports/极低算力超级智能路径深化.md)
  - TD-MPC2's multitask score rises roughly log-linearly from 1M to 317M parameters (16.0 → 70.6 over 80 tasks), with no "small is enough" break. — [TD-MPC2](https://arxiv.org/abs/2310.16828) (via [world_model_agents.md](/home/user/ASI/research_notes/极低算力超级智能路径深化/world_model_agents.md))
- **Derived floor f**: median 10^24.5, p10–p90 10^22.6–10^26.4. P(f ≤ 1e24) = 36.8%, P(f ≤ 1e23) = 15.8%, P(f ≤ 1e21) = 0.96%. These are upper bounds on "ever achievable".

**Inputs for timing (frontier ASI)**
- **Years to full AI R&D automation, T_auto**: lognormal with median 6.0 yr (≈2032.8) and σ_ln 0.9, so p10 is 2028.7 and p90 is 2045.8. Calibration sources (all snippets):
  - AIFP Q1-2026: full AI R&D automation median ~early 2031, 25th percentile ~mid-2028. — [AIFP Q1 2026](https://www.lesswrong.com/posts/XLLjqMxETva3ABtsK/q1-2026-timelines-update) (snippet)
  - METR: 99% AI R&D automation median late 2032. — [METR](https://metr.org/notes/2026-02-10-simpler-ai-timelines-model/) (snippet)
  - Metaculus "general AI": median Jan 2033, 25% by 2029. — [AIToolsReview Sep 2026](https://aitoolsreview.co.uk/insights/agi-timeline-predictions-2026); [Metaculus notebook](https://www.metaculus.com/notebooks/43363/ai-forecasting-in-2026/) (snippets)
  - The wide upper tail covers Cotra 2022 and the AI Impacts 2023 survey (see Q4).
- **Take-off gap T_auto → frontier ASI**: lognormal with median 1.5 yr and σ_ln 0.8, so p10–p90 is 0.5–4.2 yr.
  - Metaculus weak AGI → superintelligence: 42.6 months. — [Metaculus q9062](https://www.metaculus.com/questions/9062/time-from-weak-agi-to-superintelligence/) (snippet; date of the snapshot unknown)
- **Frontier run size FC(t)**: 10^27 in 2026 plus N(0, 0.4) noise, growing 0.6 OOM/yr to 2030 and 0.25 OOM/yr afterwards, capped at 10^30.5.
  - Grok 4's 5.0e26 is the largest run Epoch records. — [compute_ledger.md Q3](/home/user/ASI/research_notes/极低算力超级智能路径深化/compute_ledger.md)
  - Frontier training compute has grown 4–5×/yr. — [Epoch](https://epoch.ai/blog/training-compute-of-frontier-ai-models-grows-by-4-5x-per-year)
  - Scaling "can likely be sustained until at least 2030", reaching ~2e29 FLOP. — [Epoch, via round-1 notes](https://epoch.ai/blog/can-ai-scaling-continue-through-2030)
  - The post-2030 slowdown and the cap are *assumptions*.

**Inputs for small-scale algorithmic progress (path B, and path A's floor approach)**
- **s_pre**, small-scale "from scratch, no teacher" progress before automation: U(0.12, 0.50) OOM/yr, i.e. 1.3×–3.2× per year.
  - Lower end: Gundlach et al. find that small-scale ablations explain <10× and total small-scale progress is <100× over 2012–2023, i.e. <1.52×/yr. — [arXiv 2511.21622](https://arxiv.org/abs/2511.21622)
  - Upper end: Ho et al. measure an 8-month halving (95% CI 5–14 months), i.e. 2.8×/yr. — [arXiv 2403.05812](https://arxiv.org/abs/2403.05812)
  - Upper end: GPT-4 (2.1e25) → DeepSeek-V3 (3.3e24) on the full ledger is ~6× in 21 months (≈2.8×/yr), measured at the 1e24–1e25 scale that matters for these targets. — [compute_ledger.md Q4](/home/user/ASI/research_notes/极低算力超级智能路径深化/compute_ledger.md)
  - One analysis puts "catch-up" algorithmic progress at 16–60×/yr in 2023–2025. — [LessWrong](https://www.lesswrong.com/posts/yXLqrpfFwBW5knpgc/catch-up-algorithmic-progress-might-actually-be-60-per-year) (snippet). Catch-up progress includes distillation from frontier teachers, so it is borrowed compute under this ledger and is **not** used.
- **k**, the AI-era speed-up of small-scale progress (a multiplier on the log-rate; √k applies between T_auto and T_F): log-uniform on [1, 10]. *Assumption*, bracketed by:
  - Forethought: a software intelligence explosion "could increase effective compute by approximately 12 orders of magnitude", excluding paradigm shifts. — [Forethought](https://www.forethought.org/research/how-quick-and-big-would-a-software-intelligence-explosion-be) (snippet)
  - Epoch: the compute-bottleneck objection is "suggestive but the empirical evidence is shaky". — [Epoch](https://epoch.ai/gradient-updates/the-least-understood-driver-of-ai-progress) (snippet)
  - Cunningham et al.: feedback loops are "not currently strong enough to generate a self-sustaining acceleration". — [arXiv 2609.15802](https://arxiv.org/abs/2609.15802) (snippet)
  - METR: Anthropic researcher uplift is plausibly above 2×. — [METR](https://metr.org/notes/2026-07-08-anthropic-researcher-uplift/) (snippet)

**Inputs for discovering a complete compact learner (path A)**
- **Human-led hazard λ_H**: log-uniform on 0.2–3%/yr (median 0.77%/yr). **AI-led hazard λ_A** after T_auto: log-uniform on 2–30%/yr (median 7.75%/yr). Multiplier **m** after frontier ASI: 1–5×. All three are *assumptions*, bracketed by:
  - DiscoRL's Disco103 is a 754,778-parameter update rule (~2.4e7 bits). It was found for ~1e22 FLOP (round-2 estimate), beat hand-designed rules, and generalised to unseen environments. — [disco_rl](https://github.com/google-deepmind/disco_rl); [Nature](https://www.nature.com/articles/s41586-025-09761-x)
  - Every discovery so far is a *component*; no complete learner has been found; a complete-learner search is estimated at ~1e25–1e26. — [compressed_priors.md §6](/home/user/ASI/research_notes/极低算力超级智能路径深化/compressed_priors.md)
  - AIDE²: 7 successive self-improvements in 8 days, but only to the scaffold. — [arXiv 2609.26457](https://arxiv.org/abs/2609.26457) (snippet)
  - ASI-Arch: 20k GPU-hours produced only modest gains. — [ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch)
- **G**, the gap between the first discovered version and the floor: U(0, 2) OOM. **r_fresh**, the refinement rate of a fresh paradigm: U(0.2, 1.0) OOM/yr. Anchors:
  - AlphaGo Zero 3-day (2.7e22) → KataGo (~1e21) in ~1.5–2 years; GPT-4 → V3 at ~0.37 OOM/yr. — [compute_ledger.md Q4](/home/user/ASI/research_notes/极低算力超级智能路径深化/compute_ledger.md)
- **lag** from discovery to a verified instance: U(0.5, 2) yr (*assumption*). My estimate is that training a 1e24 run takes ~1 month on 1,000 H100s, so the lag is research and verification, not compute.

**Inputs for path B (cognitive-core catch-up)**
- **K_core**, a one-time discount for core + retrieval relative to a monolithic frontier model: U(0, 1.5) OOM. **P_B**, the extra floor penalty of the LLM lineage: U(0, 1.5) OOM. Both are *assumptions*. Anchors:
  - RETRO at 7.5B parameters ≈ GPT-3 at 175B (~25× fewer parameters, not FLOP). — [RETRO](https://arxiv.org/pdf/2112.04426)
  - VibeThinker's self-reported, hidden-teacher results. — [red_team.md Q2](/home/user/ASI/research_notes/极低算力超级智能路径深化/red_team.md)
  - The brain spends ~4e15 FLOP per word; V3 spends ~2.2e11 FLOP per token. — [round-2 report](/home/user/ASI/reports/极低算力超级智能路径深化.md)

**Direction mapping (only affects the direction weights)**
- Path-A forms follow round 2's relative weights for D1:D2:D3:D5:D7:D8 = 30:24:15:7:3:2, with Dirichlet noise (concentration 20). **This input is circular**: it is inherited from round 2, not independent evidence.
- An AI-found learner is D4 ("unlike any named paradigm") with p_novel ~ U(0.3, 0.7). *Assumption*; DiscoRL's rule "invents its own prediction targets". — [red_team.md Q1](/home/user/ASI/research_notes/极低算力超级智能路径深化/red_team.md)
- Path-B successes are D1 with probability U(0, 0.3), then D2 with probability U(0.6, 0.9), otherwise D6.

### Inferences
- **Why this structure.** Round 2's numbers mix three different questions: *whether* a ≤X learner can exist (the floor), *when* someone finds or derives it (hazards and progress rates), and *what it looks like* (the mapping). The model separates the three, so each can be attacked with different evidence.
- **Two routes to low compute are modelled, not one.** Round 2 quantified the "big-then-small" route only as distillation, which is disqualified under the ledger. The model adds **path B**: once a frontier ASI exists, its paradigm is re-derived from scratch at shrinking compute, with the frontier ASI itself speeding up small-scale algorithmic progress. This path is legitimate under the ledger and was not quantified in round 2.
- **The floor is a product of three uncertain factors.** Its σ ≈ 1.5 OOM is mostly E_max and C_brain; M_asi contributes less. E_max is the least evidenced number in the whole model.
- **Every input distribution comes from either a note file, an external source, or the label "Assumption".** The input table printed by the model (Table 0 below) repeats each source tag.

### Gaps
- No source quantifies M_asi (compute for superhuman breadth relative to one human lifetime) or E_max (the best-learner vs brain ratio). Both are assumptions, and they dominate the output (Q3).
- No source gives a per-year probability of discovering a *complete* compact general learner, by humans or by AI. λ_H and λ_A are bracketed by analogies only.
- The within-path-A split of directions is inherited from round 2 (circular). The model cannot independently rank D1 against D2 *within* discovered learners.

## Q2. Model outputs: P(ASI ≤ 1e24 / 1e23 / 1e21 by 2035 / 2040 / 2050), P(first ASI is low-compute), engines, direction weights

### Takeaway
Base case (N = 400,000):
- **P(ASI with full-pipeline ≤1e24 exists)**: **~10% by 2035, ~20% by 2040, ~31% by 2050.** It saturates toward the 37% floor ceiling.
- **≤1e23**: ~4% / ~8% / ~13%.
- **≤1e21**: ~0.2% / ~0.5% / ~0.8%, rising to ~1–1.5% under fat tails.
- **P(the first ASI is already achievable at ≤1e24)**: **~3%** (strict); ~5% (loose: a ≤1e24 ASI exists before the frontier would have produced ASI).
- **Timing relative to the frontier**: when a ≤1e24 ASI comes after the frontier ASI, it lags by a median **~3.6 years** (p10 1.2, p90 12.8).
- **Engines**: 84% of ≤1e24-by-2040 successes are produced in the AI-research era: path A AI-led 62%, path B catch-up 22%, human-led discovery 16%.
- **Direction weights** (≤1e24 by 2040): **D4 AI-discovered novel paradigm ~31%, D2 compact cognitive core ~28%, D1 world-model continual-learning agent ~21%**, D3 ~9%, D6 ~5%, D5 ~4%, D7 ~2%, D8 ~1%. The top three sit within each other's assumption noise (see Q3).

### Cited Findings
- All numbers in this section are outputs of [low_compute_asi_model.py](/home/user/ASI/models/low_compute_asi_model.py). The verbatim run log follows. It was run on 26 Sept 2026 with numpy 2.4.6, seed 20260926, in 23.6 s. The stdlib-only fallback (`--no-numpy --quick`) reproduced the headline within Monte Carlo noise: 20.2% vs 20.4% for ≤1e24 by 2040.

#### Verbatim model output (python3 /home/user/ASI/models/low_compute_asi_model.py)
```
======================================================================================================================
LOW-COMPUTE ASI MONTE CARLO MODEL  (as of Sept 2026; seed 20260926; backend: numpy 2.4.6)
Full-pipeline ledger: training + inference + ALL borrowed (teacher/base/distillation/synthetic) compute.
All probabilities are model outputs conditional on the stated input distributions -- not facts.
======================================================================================================================

TABLE 0. Input distributions as sampled (p10 / p50 / p90), with evidential source
----------------------------------------------------------------------------------------------------------------------
input      meaning                                                 p10      p50      p90  source
logCbrain  log10 FLOP, brain lifetime compute (~30 yr)          22.720   23.998   25.282  Carlsmith 1e13-1e17 FLOP/s x 30 yr; Cotra lifetime anchor ~1e24 [theory_limits Q1]
logM       log10 x, ASI breadth/depth multiplier over C_brain    0.360    1.001    1.642  Assumption; round-2 report (breadth w/o proportional experience); TD-MPC2 scaling
logE       log10 x, best learner efficiency vs brain            -0.783    0.500    1.783  Assumption; narrow 3-6 OOM wins vs no general <3e24 [compute_ledger Q4; compressed_priors 6]
P_B        log10 x, path-B paradigm floor penalty                0.151    0.751    1.350  Assumption; round-2 per-sample compute comparison (brain 4e15/word vs V3 2.2e11/token)
logK       log10 x, core+retrieval discount at T_F               0.149    0.751    1.350  Assumption; V3 vs GPT-4 6x; RETRO 25x params; VibeThinker (self-rep.) [red_team Q2]
G          log10 x, first-version gap above floor                0.200    1.001    1.799  Assumption; AGZ->KataGo, GPT-4->V3 refinement [compute_ledger Q4]
r_fresh    OOM/yr, post-discovery refinement rate                0.280    0.600    0.920  GPT-4->V3 0.37/yr; AGZ->KataGo ~1/yr; Ho 0.45/yr [compute_ledger Q4; theory_limits Q3]
s_pre      OOM/yr, small-scale algorithmic progress (pre-auto    0.158    0.310    0.462  Gundlach <1.5x/yr (0.18) .. Ho/V3-ledger 2.8x/yr (0.45) [theory_limits Q3]
log_k      log10 x, AI-era speed-up of small-scale progress      0.099    0.499    0.900  Assumption; Forethought SIE ~12 OOM; Cunningham not self-sustaining; METR >2x (snippets) [red_team Q1,Q5]
yrs_auto   years from 2026.75 to full AI R&D automation          1.887    5.990   18.978  AIFP ~2031, METR ~2032.9, Metaculus 2033, Cotra 2040, AI Impacts 2047 (snippets)
gap        years from T_auto to frontier ASI                     0.538    1.499    4.183  Metaculus weak AGI->ASI ~42.6 mo (snippet); AI-2027 takeoff ~1 yr (snippet)
fc_noise   log10 x, frontier run-size noise                     -0.513   -0.000    0.513  Assumption; Grok 4 5e26 record; 2026 frontier 5e26-3e27 assumed [compute_ledger Q3]
log_lamH   log10 /yr, human-led discovery hazard                -2.581   -2.111   -1.640  Assumption; only component-level discoveries to date [compressed_priors 2,6; round-2 report]
log_lamA   log10 /yr, AI-led discovery hazard (post-automatio   -1.580   -1.109   -0.640  Assumption; DiscoRL ~1e22 discovery; AIDE2; ASI-Arch modest [red_team Q1; compressed_priors 2]
log_m      log10 x, AI-led hazard multiplier after T_F           0.070    0.350    0.629  Assumption
lag        years, discovery -> verified instance                 0.650    1.250    1.850  Assumption; 1e24 FLOP ~ 1 month on 1k H100 (my estimate)
p_novel    prob., AI-found learner is a novel form (D4)          0.340    0.499    0.660  Assumption; DiscoRL invents own prediction targets [red_team Q1]
p_core     prob., path-B success is D2 (else D6)                 0.630    0.750    0.870  Assumption; round-2 D2 vs D6 = 24:6
p_B_wm     prob., path-B success is classified D1                0.030    0.150    0.270  Assumption; functions moving into big models [red_team Q4,Q6]
-> derived ASI floor f = logCbrain+logM-logE: p5 22.04  p10 22.58  p50 24.50  p90 26.42  p95 26.96
   P(f<=24) =  36.8%   P(f<=23) =  15.8%   P(f<=21) =  0.96%   (upper bounds on 'ever')

TABLE 1. [BASE] P(an ASI with full-pipeline compute <= X exists by year Y)   (N=400000)
--------------------------------------------------------------------------------------
threshold                          2035       2040       2050   round-2 (2040)
<= 1e24 FLOP                      9.95%     20.42%     30.88%   20-25%
<= 1e23 FLOP                      3.95%      8.42%     12.98%   ~6%
<= 1e21 FLOP                      0.22%      0.48%      0.77%   1-2%
any ASI (context)                 54.9%      78.2%      93.5%
big-compute frontier ASI          50.4%      74.7%      91.7%
MC standard error on P(<=1e24 by 2040): +-0.06 pp

TABLE 2. [BASE] Is the first ASI low-compute?
--------------------------------------------------------------------------------------
P(first ASI already achievable at <=1e24)  [strict]           2.88%   round 2: <=5%
P(first ASI already achievable at <=1e23)  [strict]           0.97%
P(a <=1e24 ASI exists before the big-compute frontier ASI)    5.23%  [loose]
P(first ASI comes from a newly discovered compact learner, any size) 16.25%
P(first ASI used >=1e26 full-pipeline | some ASI by 2040)     90.0%   round 2: ~95%
Lag from frontier ASI to first <=1e24 ASI (when it comes after): p10 1.2 / p50 3.6 / p90 12.8 yr

TABLE 3. [BASE] Engine and path of low-compute success
--------------------------------------------------------------------------------------
<=1e24 by 2040 (n= 81685): path A human-led  15.9% | path A AI-led  62.3% | path B core catch-up  21.8%
<=1e24 by 2050 (n=123513): path A human-led  13.7% | path A AI-led  64.1% | path B core catch-up  22.2%
<=1e23 by 2040 (n= 33674): path A human-led  16.9% | path A AI-led  67.7% | path B core catch-up  15.4%
P(complete compact learner discovered by 2040)  56.1%; of which AI-led  82.2%   (round 2: ~60%)
Share of <=1e24-by-2040 successes produced in the AI-research era (path A AI-led + path B):  84.1%

TABLE 4. [BASE] Direction weights = P(direction is the main engine | low-compute ASI achieved)
--------------------------------------------------------------------------------------------------------
     direction                                    <=1e24/40 <=1e24/50 <=1e23/50   round 2
D1   World-model continual-learning agent             20.9%     20.4%     21.0%       30%
D2   Compact cognitive core (+RLVR, retrieval)        27.8%     27.7%     24.9%       24%
D3   Verifier-driven self-improvement                  8.6%      8.5%      9.0%       15%
D4   AI-discovered novel paradigm                     31.1%     31.9%     34.5%       13%
D5   Radical brain-inspired                            4.1%      4.0%      4.4%        7%
D6   Mainstream efficiency stack                       4.6%      4.7%      3.3%        6%
D7   Tiny recursive models                             1.7%      1.7%      1.7%        3%
D8   New compute substrate                             1.2%      1.1%      1.3%        2%
(n successes: 81685 / 123513 / 51911)

TABLE 5. [BASE] Cheapest ASI in existence (full-pipeline log10 FLOP), by year
--------------------------------------------------------------------------------------
2035: some ASI exists in  54.9% of worlds; conditional on existing, cheapest p10 1e23.3 / p50 1e25.9 / p90 1e28.8
2040: some ASI exists in  78.2% of worlds; conditional on existing, cheapest p10 1e22.9 / p50 1e25.1 / p90 1e28.0
2050: some ASI exists in  93.5% of worlds; conditional on existing, cheapest p10 1e22.7 / p50 1e24.7 / p90 1e26.8

TABLE 6. Tornado: pin each input at its p10 / p90 (others unchanged; common random numbers)
Base: P(<=1e24 by 2040)  20.4% | P(<=1e23 by 2040)   8.4% | P(first ASI <=1e24, strict)  2.83%
----------------------------------------------------------------------------------------------------------------------
input      family         p10      p90 | P<=1e24/40 lo->hi | P<=1e23/40 lo->hi | first-low lo->hi  | swing(24/40)
logE       floor       -0.779    1.785 |   2.8% ->  43.2% |   0.3% ->  22.5% |   0.2% ->   7.0% |  40.3 pp
logCbrain  floor       22.721   25.278 |  43.0% ->   2.9% |  22.5% ->   0.3% |   6.9% ->   0.2% |  40.1 pp
yrs_auto   timeline     1.912   18.950 |  31.6% ->   3.9% |  13.1% ->   1.6% |   1.9% ->   4.3% |  27.6 pp
logM       floor        0.359    1.643 |  30.2% ->  11.3% |  14.6% ->   3.3% |   4.6% ->   1.3% |  18.9 pp
log_lamA   discovery   -1.581   -0.641 |  15.6% ->  25.3% |   6.1% ->  10.8% |   1.9% ->   4.2% |   9.7 pp
log_k      progress     0.101    0.900 |  18.0% ->  22.7% |   7.5% ->   9.3% |   2.8% ->   2.8% |   4.7 pp
log_m      discovery    0.069    0.629 |  18.4% ->  22.4% |   7.4% ->   9.4% |   2.8% ->   2.8% |   4.0 pp
log_lamH   discovery   -2.581   -1.640 |  18.9% ->  22.6% |   7.8% ->   9.4% |   1.8% ->   4.3% |   3.7 pp
gap        timeline     0.535    4.174 |  21.8% ->  18.2% |   9.0% ->   7.5% |   1.3% ->   5.7% |   3.7 pp
P_B        pathB        0.151    1.351 |  21.8% ->  19.2% |   9.0% ->   7.9% |   2.8% ->   2.8% |   2.6 pp
s_pre      progress     0.158    0.462 |  19.0% ->  21.5% |   7.9% ->   8.9% |   2.8% ->   2.8% |   2.5 pp
G          pathA        0.201    1.800 |  21.0% ->  19.4% |   8.8% ->   7.9% |   5.1% ->   1.0% |   1.6 pp
lag        pathA        0.650    1.849 |  21.1% ->  19.6% |   8.8% ->   8.0% |   3.5% ->   2.3% |   1.6 pp
logK       pathB        0.151    1.351 |  20.1% ->  20.6% |   8.3% ->   8.5% |   2.8% ->   2.8% |   0.4 pp
r_fresh    pathA        0.280    0.919 |  20.1% ->  20.5% |   8.3% ->   8.5% |   2.8% ->   2.8% |   0.4 pp
fc_noise   timeline    -0.509    0.513 |  20.5% ->  20.2% |   8.5% ->   8.4% |   2.8% ->   2.8% |   0.4 pp
p_novel    mapping      0.340    0.660 |  20.4% ->  20.4% |   8.4% ->   8.4% |   2.8% ->   2.8% |   0.0 pp
p_core     mapping      0.630    0.870 |  20.4% ->  20.4% |   8.4% ->   8.4% |   2.8% ->   2.8% |   0.0 pp
p_B_wm     mapping      0.030    0.270 |  20.4% ->  20.4% |   8.4% ->   8.4% |   2.8% ->   2.8% |   0.0 pp

TABLE 6b. Direction-weight tornado: weights among <=1e24-by-2050 successes, input at p10 -> p90
         (sorted by the largest swing in any of D1/D2/D4)
--------------------------------------------------------------------------------------------------------
input      family    | D1 world-model    | D2 cog. core      | D4 AI-novel       | D6 mainstream    
p_novel    mapping   |  23.9% ->  16.2% |  30.7% ->  24.7% |  22.0% ->  42.6% |   4.7% ->   4.7%
log_lamA   discovery |  20.2% ->  20.0% |  36.8% ->  20.9% |  21.5% ->  40.3% |   8.4% ->   1.9%
log_k      progress  |  20.8% ->  19.3% |  20.7% ->  35.1% |  39.0% ->  25.2% |   1.6% ->   8.0%
P_B        pathB     |  19.5% ->  20.6% |  33.2% ->  22.7% |  26.8% ->  37.0% |   7.4% ->   2.5%
log_lamH   discovery |  18.3% ->  22.6% |  27.5% ->  28.0% |  35.8% ->  27.2% |   5.3% ->   4.0%
s_pre      progress  |  20.5% ->  19.7% |  23.2% ->  31.4% |  36.6% ->  28.8% |   2.7% ->   6.4%
log_m      discovery |  20.1% ->  20.0% |  31.3% ->  24.6% |  27.9% ->  35.9% |   6.3% ->   3.4%
yrs_auto   timeline  |  18.8% ->  25.0% |  29.1% ->  26.2% |  33.0% ->  25.1% |   5.7% ->   2.4%
logE       floor     |  20.7% ->  20.0% |  22.9% ->  29.4% |  36.8% ->  30.5% |   3.0% ->   5.5%
logCbrain  floor     |  20.1% ->  21.1% |  29.4% ->  23.2% |  30.4% ->  36.5% |   5.5% ->   2.7%
p_B_wm     mapping   |  17.4% ->  22.8% |  29.7% ->  25.6% |  32.3% ->  32.3% |   5.4% ->   4.1%
p_core     mapping   |  20.1% ->  20.1% |  25.3% ->  29.9% |  32.3% ->  32.3% |   7.1% ->   2.5%
gap        timeline  |  19.8% ->  20.4% |  29.6% ->  25.4% |  30.7% ->  34.3% |   5.6% ->   3.6%
logM       floor     |  20.1% ->  20.4% |  28.8% ->  26.0% |  31.1% ->  33.6% |   5.2% ->   4.0%
lag        pathA     |  20.1% ->  20.0% |  26.6% ->  28.9% |  33.3% ->  31.1% |   4.2% ->   5.3%
logK       pathB     |  20.1% ->  20.0% |  26.8% ->  28.7% |  33.1% ->  31.3% |   4.4% ->   5.2%
fc_noise   timeline  |  20.0% ->  20.1% |  28.5% ->  26.9% |  31.5% ->  33.0% |   5.1% ->   4.4%
G          pathA     |  20.1% ->  20.0% |  27.4% ->  28.2% |  32.5% ->  31.7% |   4.6% ->   5.0%

TABLE 7. First-order sensitivity index S1 = Var(E[outcome|input]) / Var(outcome)
         (= expected share of outcome variance removed by learning that input exactly; VOI proxy)
--------------------------------------------------------------------------------------
input      family      P<=1e24/2040   P<=1e23/2040  first ASI<=1e24
floor_f*   derived            0.432          0.474            0.087
logE       floor              0.139          0.125            0.029
logCbrain  floor              0.138          0.121            0.028
yrs_auto   timeline           0.073          0.026            0.005
logM       floor              0.032          0.025            0.006
log_lamA   discovery          0.007          0.003            0.002
log_k      progress           0.002          0.000            0.000
gap        timeline           0.002          0.000            0.014
log_m      discovery          0.001          0.001            0.000
log_lamH   discovery          0.001          0.000            0.003
s_pre      progress           0.001          0.000            0.000
P_B        pathB              0.001          0.000            0.000
G          pathA              0.000          0.000            0.008
lag        pathA              0.000          0.000            0.001
fc_noise   timeline           0.000          0.000            0.000
r_fresh    pathA              0.000          0.000            0.000
logK       pathB              0.000          0.000            0.000
p_novel    mapping            0.000          0.000            0.000
p_core     mapping            0.000          0.000            0.000
p_B_wm     mapping            0.000          0.000            0.000
Sum of S1 over the 19 primitive inputs (P<=1e24/2040): 0.40  (1 - sum = interactions + event noise)

TABLE 7b. Headline conditional on an input's tercile (what the answer becomes if evidence pins it)
--------------------------------------------------------------------------------------------------------
input      | P(<=1e24 by 2040): lo/mid/hi     | P(<=1e23 by 2040): lo/mid/hi     | tercile cut-points
floor_f*   |    56.5%     4.8%     0.0%        |    25.3%     0.0%     0.0%        | 23.86 / 25.14
yrs_auto   |    30.8%    24.4%     6.1%        |    12.7%    10.0%     2.5%        | 4.07 / 8.80
log_lamA   |    16.4%    20.4%    24.4%        |     6.5%     8.4%    10.4%        | -1.30 / -0.91
log_k      |    18.4%    20.6%    22.4%        |     7.7%     8.4%     9.2%        | 0.33 / 0.67
s_pre      |    19.3%    20.6%    21.3%        |     8.0%     8.4%     8.8%        | 0.25 / 0.37
logM       |    28.7%    19.8%    12.7%        |    13.6%     7.6%     4.0%        | 0.79 / 1.22

TABLE 8. Scenarios (each re-sampled with the same seed; N=200000)
----------------------------------------------------------------------------------------------------------------------------
scenario                                        24/2035  24/2040  24/2050  23/2040  21/2040  21/2050 first<=24     D1     D2     D3     D4
BASE (independent inputs)                         10.0%    20.6%    31.1%     8.3%    0.48%    0.76%     2.78%    20%    28%     8%    32%
rho=0.3: cheap floor <-> early automation         12.6%    23.7%    32.8%    10.1%    0.65%    0.85%     2.39%    20%    28%     8%    32%
Fat-tailed floor (t4 on C_brain, E)                9.7%    20.0%    30.3%     8.0%    0.95%    1.48%     2.76%    20%    28%     9%    32%
Gundlach world: s_pre=0.15, k=1                    7.9%    17.5%    28.7%     7.3%    0.43%    0.74%     2.78%    22%    17%    11%    42%
Ho world: s_pre=0.45 (k as base)                  11.0%    21.6%    31.5%     8.7%    0.50%    0.77%     2.78%    20%    31%     8%    29%
No AI research edge: k=1, m=1, lamA=lamH range     2.8%     6.2%    15.5%     2.4%    0.14%    0.31%     1.69%    25%    39%    10%    12%
Slow frontier (survey-like, median ~2040)          3.8%    10.9%    24.6%     4.4%    0.26%    0.61%     3.73%    22%    27%     9%    30%
Fast frontier (AIFP-like, median ~2029)           19.3%    30.0%    35.1%    12.2%    0.71%    0.87%     1.96%    19%    29%     8%    33%
Brain-optimal: logE ~ N(-0.5, 0.5)                 2.7%     5.7%     8.9%     1.0%    0.00%    0.01%     0.50%    21%    25%     9%    35%
Optimistic floor: logE ~ N(1.5, 1.0)              18.0%    36.2%    53.8%    20.2%    2.42%    3.81%     6.18%    20%    30%     8%    30%
Slow AI discovery: lamA 0.5%-5%/yr                 6.5%    14.6%    25.4%     5.4%    0.29%    0.55%     1.81%    21%    40%     7%    17%
Low novelty: p_novel ~ U(0.1, 0.3)                10.0%    20.6%    31.1%     8.3%    0.48%    0.76%     2.78%    27%    34%    12%    13%
(D1/D2/D3/D4 columns = direction weights conditional on <=1e24 by 2050)

TABLE 9. Consistency checks against round 2 (N=200000 each)
----------------------------------------------------------------------------------------------------------------------
(a) P(<=1e24 by 2040 | frontier ASI by 2040) =  25.9% ; P(<=1e24 by 2040 | no frontier ASI by 2040) =   4.8%
(b) Frontier ASI year p10/p25/p50/p75/p90: 2030.2 / 2031.9 / 2034.9 / 2040.1 / 2048.2
    AI R&D automation year p10/p25/p50/p75/p90: 2028.7 / 2030.0 / 2032.8 / 2037.8 / 2045.8
(c) Year at which P(<=1e24 ASI exists) reaches: 5% by 2032.8; 10% by 2035.0; 20% by 2039.7; 30% by 2048.3
(d) Scaling the AI-led hazard down: share of discoveries by 2040 that are AI-led, and the headline
    lamA/1  median  7.75%/yr (10.0x lamH): discovered by 2040  55.9%, AI-led  82.1%, P(<=1e24 by 2040)  20.6%
    lamA/2  median  3.87%/yr ( 5.0x lamH): discovered by 2040  43.9%, AI-led  75.4%, P(<=1e24 by 2040)  17.8%
    lamA/3  median  2.58%/yr ( 3.3x lamH): discovered by 2040  37.3%, AI-led  70.2%, P(<=1e24 by 2040)  16.3%
    lamA/5  median  1.55%/yr ( 2.0x lamH): discovered by 2040  29.9%, AI-led  61.6%, P(<=1e24 by 2040)  14.7%
    lamA/10 median  0.77%/yr ( 1.0x lamH): discovered by 2040  22.4%, AI-led  47.2%, P(<=1e24 by 2040)  13.2%
    Round-2 implied P(novel form | AI-found) = D4 13% / AI engine 60% = 0.22
(e) Direction weights (<=1e24 by 2050) if path-A learners may be D2 [base] vs may NOT be D2 [variant]:
    D1 20%/27%  D2 28%/14%  D3 8%/12%  D4 32%/32%  D5 4%/6%  D6 5%/5%  D7 2%/2%  D8 1%/2%

[done in 23.6 s]
```

### Inferences
**Summary table of model outputs vs round 2** (model = base case; range = min–max across the 12 scenarios of Table 8):

| Quantity | Round 2 (subjective) | Model base | Scenario range |
|---|---|---|---|
| P(≤1e24 by 2035) | — | ~10% | 3–19% |
| P(≤1e24 by 2040) | 20–25% | **~20%** | 6–36% |
| P(≤1e24 by 2050) | — | ~31% | 9–54% |
| P(≤1e23 by 2040) | ~6% | **~8%** | 1–20% |
| P(≤1e21 by 2040) | 1–2% | **~0.5%** (~1% with fat tails) | 0.0–2.4% |
| P(first ASI achievable at ≤1e24) | ≤5% | **~3%** strict / ~5% loose | 0.5–6% |
| P(first ASI used ≥1e26 \| ASI by 2040) | ~95% | **~90%** | — |
| P(discovery engine is AI \| learner found by 2040) | ~60% | **~82%** | 47–82% (Table 9d) |
| D1 world-model CL agent | 30% | **~21%** | 19–27% |
| D2 compact cognitive core | 24% | **~28%** | 14–40% |
| D3 verifier-driven self-improvement | 15% | ~9% | 7–12% |
| D4 AI-discovered novel paradigm | 13% | **~31%** | 12–42% |
| D5 / D6 / D7 / D8 | 7 / 6 / 3 / 2% | ~4 / 5 / 2 / 1% | — |

- **What the model says about 以极低算力实现ASI的最可能方向.** Conditional on a ≤1e24 ASI existing, the robust answers concern *path and timing*, not *form*.
  - (i) About 84% of successes are produced in the AI-research era, and >60% come from an AI-led search for a compact learner.
  - (ii) The typical low-compute ASI appears a few years *after* a big-compute ASI (median lag ~3.6 yr). It is a by-product of the first ASI, not a rival to it.
  - (iii) The first ASI is almost never low-compute (~3%).
- **Form is not robust.** The plurality bucket is D4, "whatever automated research finds, unlike any named paradigm". That is effectively a statement of ignorance about the form, and its size is set by p_novel. Among human-nameable forms, D2 and D1 are close, and their order flips with the mapping assumption (Table 9e: D1 20% / D2 28% in the base, D1 27% / D2 14% if a discovered learner can never be a cognitive core).
- **D1 is never the plurality in any of the 12 scenarios.** Its weight stays at 19–27% and no scenario shows it as the top bucket. Round 2's "D1 ≈30%, clear first" is therefore not supported by the model's structure. At most, D1 ties for first *among human-named forms*.
- **Time profile.** P(≤1e24 exists) reaches 5% in ~2032.8, 10% in ~2035.0, 20% in ~2039.7 and 30% in ~2048.3 (Table 9c). The curve flattens toward the floor ceiling of ~37%. Of the ~69% of worlds without a ≤1e24 ASI by 2050, ~63 percentage points are worlds in which the floor itself sits above 1e24, so no learner could get there at any date.
- **Heavy dependence on the frontier.** P(≤1e24 by 2040 | frontier ASI by 2040) ≈ 26%, versus ≈ 5% without a frontier ASI by 2040 (Table 9a). The low-compute outcome is mostly downstream of big-compute ASI.
- **The cheapest ASI in 2040, when any ASI exists, has median ~10^25.1 FLOP** (p10 10^22.9, p90 10^28.0) (Table 5). A "typical" 2040 world with ASI still has its cheapest ASI above the 1e24 tier.

### Gaps
- The model does not represent *hybrids* explicitly. Each success is assigned one "main engine", as in round 2, but the real system would likely mix D1, D2 and D3 parts.
- The "first ASI" metric measures whether ASI is *achievable* at ≤1e24 when ASI first appears. In practice a lab would likely train a newly discovered learner at far more than 1e24, so the first actual ASI *run* could be large even in "low-compute-form-first" worlds.

## Q3. Sensitivity: which 2–3 inputs swing the headline most, and which evidence would reduce uncertainty most efficiently (value of information)

### Takeaway
Three input families dominate P(≤1e24 by 2040):
1. **The ASI floor.** The three factors are best-learner efficiency vs brain (logE) and brain lifetime compute (logC_brain), each swinging the headline by **~40 pp** between their p10 and p90, and the ASI breadth multiplier (logM, ~19 pp). The derived floor alone explains **~43%** of the outcome variance (first-order index S1), and ~47% for ≤1e23.
2. **Timing of AI R&D automation** (yrs_auto: 31.6% → 3.9%, ~28 pp; S1 ≈ 0.07).
3. **The AI-led discovery hazard** (λ_A: 15.6% → 25.3%, ~10 pp).

Everything else moves the headline by less than 5 pp. That includes the Gundlach-vs-Ho small-scale progress rate, which round 2 emphasised: only ~2.5 pp. For the *direction weights*, the dominant inputs are p_novel, λ_A, k, P_B and s_pre. The highest-value evidence is anything that pins the floor. If the floor lands in its lowest tercile (≤10^23.9), the headline becomes ~56%; in the middle tercile ~5%; in the top tercile ~0%.

### Cited Findings
- Tornado (Table 6; each input pinned at its sampled p10 and p90, all other inputs and event draws held fixed):

| Input | P(≤1e24/2040), p10 → p90 | P(≤1e23/2040) | P(first ASI ≤1e24) |
|---|---|---|---|
| logE | 2.8% → 43.2% | 0.3% → 22.5% | 0.2% → 7.0% |
| logC_brain | 43.0% → 2.9% | 22.5% → 0.3% | 6.9% → 0.2% |
| yrs_auto | 31.6% → 3.9% | 13.1% → 1.6% | 1.9% → 4.3% |
| logM | 30.2% → 11.3% | 14.6% → 3.3% | 4.6% → 1.3% |
| λ_A | 15.6% → 25.3% | 6.1% → 10.8% | 1.9% → 4.2% |
| k | 18.0% → 22.7% | — | — |
| s_pre (Gundlach 1.4×/yr vs Ho 2.9×/yr) | 19.0% → 21.5% | 7.9% → 8.9% | no change |

  Source: [model output, Table 6](/home/user/ASI/models/low_compute_asi_model.py)
- First-order indices S1 for P(≤1e24 by 2040): floor (derived) 0.43; logE 0.14; logC_brain 0.14; yrs_auto 0.07; logM 0.03; λ_A 0.007; every other input ≤0.002. The primitive S1 values sum to 0.40; the rest is interaction (mainly within the floor) and event noise. — [model output, Table 7](/home/user/ASI/models/low_compute_asi_model.py)
- For *first ASI low-compute*, the take-off gap matters more than for the headline (1.3% → 5.7%). A longer gap between automation and ASI gives an AI-led search time to find a compact learner before the frontier arrives. — [Table 6](/home/user/ASI/models/low_compute_asi_model.py)
- Direction-weight tornado (Table 6b; weights among ≤1e24-by-2050 successes, input p10 → p90):
  - p_novel: D4 22% → 43%, D1 24% → 16%.
  - λ_A: D2 37% → 21%, D4 22% → 40%.
  - k: D2 21% → 35%, D4 39% → 25%.
  - P_B: D2 33% → 23%.
  - s_pre: D2 23% → 31%.
  - Source: [model output](/home/user/ASI/models/low_compute_asi_model.py)
- Headline conditional on terciles (Table 7b), for P(≤1e24 by 2040):
  - floor: 56.5% / 4.8% / 0.0% (cut points 10^23.86, 10^25.14);
  - yrs_auto: 30.8% / 24.4% / 6.1% (cut points 4.1 and 8.8 yr, i.e. ~2030.8 and ~2035.6);
  - λ_A: 16.4% / 20.4% / 24.4%;
  - s_pre: 19.3% / 20.6% / 21.3%;
  - logM: 28.7% / 19.8% / 12.7%.
  - Source: [model output](/home/user/ASI/models/low_compute_asi_model.py)
- Scenario swings (Table 8), for P(≤1e24 by 2040):
  - "brain-optimal" (logE ~ N(−0.5, 0.5)): 5.7%; "optimistic floor" (logE ~ N(1.5, 1.0)): 36.2%;
  - "no AI research edge": 6.2%; "slow AI discovery": 14.6%;
  - survey-like slow frontier: 10.9%; AIFP-like fast frontier: 30.0%;
  - Gundlach world (s_pre = 0.15, k = 1): 17.5%; Ho world (s_pre = 0.45): 21.6%.
  - Source: [model output](/home/user/ASI/models/low_compute_asi_model.py)

### Inferences
- **The dominant uncertainty is not "which architecture" but "how low the ASI floor is".** The question is whether any learner can reach superhuman breadth near the brain's lifetime budget. It is almost entirely unmeasured. The model's floor has σ ≈ 1.5 OOM. That is *why* its headline agrees with round 2: both encode "floor median ~3e24 with wide spread". It is agreement of priors, not independent confirmation.
- **Value of information, ranked.** S1 is the expected share of outcome variance removed by learning an input exactly.
  1. **Floor evidence (S1 ≈ 0.43–0.47).** The most informative experiments are the ones round 2 already identified:
     - (a) The **"brain-scale compute on child-scale data" experiment**: 1e22–1e23 FLOP on ≤1e9 words plus child-view video, scored on knowledge and reasoning evals such as EWoK. It directly measures E_max near the relevant scale. — [round-2 report, 近期研究第二项](/home/user/ASI/reports/极低算力超级智能路径深化.md)
     - (b) A **≥1,000-task compounding benchmark**, which tests whether breadth costs grow sub-linearly (M_asi). — [round-2 report](/home/user/ASI/reports/极低算力超级智能路径深化.md)
     - (c) Tracking the **lowest non-borrowed full-pipeline compute for GPT-4-class and then frontier-class capability** each year, currently 3.3e24 from DeepSeek-V3. That is a public running estimate of E_max/M_asi for the LLM lineage.

     C_brain itself cannot realistically be narrowed before 2030. It needs neuroscience progress; connectomics is a 2030s–2040s project per [round-2 report](/home/user/ASI/reports/极低算力超级智能路径深化.md).
  2. **AI-R&D automation timing (S1 ≈ 0.07).** METR time-horizon and uplift measurements, plus AIFP/METR forecast updates. If automation lands before ~2031, the headline goes to ~31%; if after ~2035.5, to ~6%.
  3. **AI-led discovery rate (S1 ≈ 0.01, but larger for direction weights).** Is a *complete* learner (not a component) discovered by DiscoRL-style or AlphaEvolve-style search with "train from scratch under budget, score on held-out task families"? This resolves λ_A and p_novel at once.
  4. **Low VOI for the headline:** small-scale progress rate (s_pre), k, K_core, G, r_fresh, lag and frontier run size. The Gundlach-vs-Ho question round 2 dwelt on moves the 2040 headline by only ~2.5 pp. It mainly shifts D2's share (it drives path B's speed).
- **The "which direction" answer is governed by different inputs than the "whether" answer.** Direction weights are governed by p_novel and by the race between AI-led discovery (λ_A, m) and frontier-lineage catch-up (k, s_pre, P_B). Direction-relevant evidence therefore comes from the automated-discovery literature and from the **world-model agent vs cognitive-core head-to-head** that round 2 proposed. The floor experiments do not settle it.

### Gaps
- The indices are first-order and one-at-a-time. Interactions (notably among C_brain, E and M) are not decomposed; a full Sobol total-order analysis was not run.
- The value-of-information ranking is expressed as variance reduction in a probability. It is not tied to any decision, because no decision (e.g., governance thresholds or research funding) was specified.

## Q4. External calibration: what outside forecasts say, and how the model's timing inputs compare

### Takeaway
Timing inputs were calibrated to the 2026 forecaster community (AIFP, METR, Metaculus). The resulting frontier ASI has a median of ~2034.9 (p10 2030.2, p90 2048.2) and P(any ASI by 2040) ≈ 78%. That is **much more aggressive than Cotra's 2022 update (TAI 50% by 2040) or the AI Impacts 2023 survey (HLMI 50% by 2047)**. Under survey-like timelines the headline roughly halves, to ~11% for ≤1e24 by 2040. No outside forecaster publishes P(paradigm), P(low-compute ASI), or anything that could calibrate the floor or the direction mapping. Those inputs are, and must remain, explicit assumptions.

### Cited Findings
- **Metaculus.**
  - The community median for the first "general AI" system is January 2033, with a 25% probability by 2029 (as of Feb–mid 2026). — [AIToolsReview, Sep 2026](https://aitoolsreview.co.uk/insights/agi-timeline-predictions-2026); [Metaculus notebook "AI Forecasting in 2026"](https://www.metaculus.com/notebooks/43363/ai-forecasting-in-2026/) (snippets)
  - "Weakly general AI" median June 2028. — same aggregator (snippet)
  - Conflict: another snippet says "Metaculus now predicts weak AGI before the end of 2026 and strong AGI at the start of 2031". This appears to be an older LessWrong post title ("Metaculus Predicts Weak AGI in 2 Years and AGI in 10"). — [LessWrong](https://www.lesswrong.com/posts/CiYSFaQvtwj98csqG/metaculus-predicts-weak-agi-in-2-years-and-agi-in-10) (snippet; **conflict, unverified**)
  - Weak AGI → superintelligence: community prediction 42.6 months. — [Metaculus q9062](https://www.metaculus.com/questions/9062/time-from-weak-agi-to-superintelligence/) (snippet; snapshot date unknown)
  - Round 2 recorded a separate aggregator claiming a "Feb 2028" median, which conflicts with the figures above. — [red_team.md Q5](/home/user/ASI/research_notes/极低算力超级智能路径深化/red_team.md)
- **AI Futures Project.**
  - Q1 2026: Kokotajlo's Automated Coder median moved "from late 2029 to mid 2028"; Lifland's moved from early 2032 to mid-2030. Full AI R&D automation median ~early 2031, 25th percentile ~mid-2028. — [AIFP Q1 2026](https://www.lesswrong.com/posts/XLLjqMxETva3ABtsK/q1-2026-timelines-update) (snippet)
  - Dec 2025: superhuman AI researcher median 2031 (adjusted from the model's 2030). — [AIFP Dec 2025](https://www.lesswrong.com/posts/YABG5JmztGGPwNFq2/ai-futures-timelines-and-takeoff-model-dec-2025-update) (snippet)
  - Kokotajlo's ASI median is reported inconsistently in snippets as 2029, ~2030, 2031 and even 2035. — [FutureSearch](https://futuresearch.ai/blog/ai-2027-6-months-later/) (snippet; **conflicting, unverified**)
- **METR** (Feb 2026): a simple model puts the median for >99% AI R&D automation in late 2032, and "any reasonable timelines model will predict superhuman AI researchers before 2036 unless AI progress hits a wall". — [METR](https://metr.org/notes/2026-02-10-simpler-ai-timelines-model/) (snippet, via [red_team.md Q5](/home/user/ASI/research_notes/极低算力超级智能路径深化/red_team.md))
- **Cotra 2022 update**: TAI 15% by 2030, 35% by 2036, 50% by 2040, 60% by 2050. — [EA Forum, "Timelines to Transformative AI: an investigation"](https://forum.effectivealtruism.org/posts/hzhGL7tb56hG5pRXY/timelines-to-transformative-ai-an-investigation) (snippet); [Alignment Forum, two-year update](https://www.alignmentforum.org/posts/AfH2oPHCApdKicM4m/two-year-update-on-my-personal-ai-timelines) (snippet)
  - Bio-anchors: lifetime anchor ~1e24, evolution anchor ~1e41. — [Epoch](https://epoch.ai/blog/grokking-bioanchors)
- **AI Impacts 2023 survey** (2,778 researchers): "the chance of unaided machines outperforming humans in every possible task was estimated at 10% by 2027, and 50% by 2047". — [Grace et al., arXiv 2401.02843](https://arxiv.org/html/2401.02843v3) (snippet)
- **Epoch GATE.**
  - It links investment to effective compute, and effective compute to automation of a fixed spectrum of tasks, each requiring progressively more effective compute. It assumes LLM algorithmic progress halves compute requirements every ~8 months. — [GATE, arXiv 2503.04941](https://arxiv.org/html/2503.04941v2) (snippet)
  - A snippet claims an AGI requirement of "1e36 eFLOP, updated to ~1e32". It appears to come from a post on Davidson's takeoff-speeds model, not from GATE. — [readtheoom substack](https://readtheoom.substack.com/p/the-takeoff-speeds-model-predicts) (snippet; **attribution uncertain, not used**)
- **Epoch "Direct Approach"**: uses scaling laws to give an *upper bound* on transformative-AI training compute. No specific number was retrievable. — [Epoch](https://epoch.ai/blog/the-direct-approach) (snippet)
- **No forecaster publishes P(paradigm)** or P(low-compute ASI). — [red_team.md Q5](/home/user/ASI/research_notes/极低算力超级智能路径深化/red_team.md)

### Inferences
| Source | Milestone | Median | Model counterpart |
|---|---|---|---|
| AIFP Q1-2026 (snippet) | full AI R&D automation | ~early 2031 | T_auto median 2032.8 |
| METR Feb 2026 (snippet) | 99% AI R&D automation | late 2032 | T_auto median 2032.8 |
| Metaculus (snippet) | first "general AI" | Jan 2033 | T_auto median 2032.8; frontier ASI median 2034.9 |
| Metaculus chain (snippet) | weak AGI (Jun 2028) + 42.6 mo → ASI | ~end 2031 | frontier ASI p25 2031.9 |
| Metaculus chain (snippet) | strong AGI (Jan 2033) + 42.6 mo | ~mid 2036 | frontier ASI p50–p75 |
| Cotra 2022 (snippet) | TAI | 2040 (50%) | any ASI by 2040: 78% (**more aggressive**) |
| AI Impacts 2023 (snippet) | HLMI | 2047 (50%) | frontier ASI p75 2040.1 (**more aggressive**) |

- The model's timing sits in the **middle of the 2026 forecaster pool** and **well ahead of the 2022–2023 expert pool**. Choosing between these pools is worth about 2× on the headline: 20.6% (base) vs 10.9% (survey-like) vs 30.0% (AIFP-like). The report should state which pool it trusts.
- GATE's 8-month halving matches the upper end of s_pre (0.45 OOM/yr). GATE is a *frontier* effective-compute model, though, and says nothing about *small-scale* progress. This is exactly the Gundlach caveat, and the reason s_pre is a range rather than GATE's point value.
- The floor inputs (C_brain, M, E) and the direction mapping have **no external calibration source**. The bio-anchor lifetime number calibrates only C_brain, and E and M dominate the variance jointly with it.

### Gaps
- The Metaculus, AIFP, METR, AI Impacts and Cotra-update numbers are all snippet-only in this session (direct fetches were blocked). The Metaculus figures conflict across snippets.
- GATE's and the Direct Approach's quantitative requirement distributions could not be read, so they calibrate nothing beyond the 8-month halving.
- No 2024–2026 expert survey on "which paradigm yields AGI or ASI" was found.

## Q5. Where the round-2 subjective numbers look miscalibrated or internally inconsistent

### Takeaway
Round 2's **headline tier-1 number (20–25% for ≤1e24 by 2040) and "first ASI low-compute ≤5%" survive**: the model gives ~20% and ~3–5%. Five other round-2 numbers look off, or inconsistent with each other, once they are forced into one coherent model:
1. **D4 "AI-discovered novel paradigm" at 13% is too low** given round 2's own ~60% AI-engine belief; the model gives ~31%.
2. **D1 at 30% as a clear first is not robust.** The model gives 19–27%, and D1 is never the plurality in any scenario.
3. **"AI engine ~60%" and "≤1e24 by 2040 20–25%" are mutually inconsistent** in the model: 60% AI-led pairs with a ~15% headline.
4. **≤1e21 at 1–2% by 2040 is high.** The model gives ~0.5%, or ~1% with fat tails, and that number is a pure tail assumption.
5. **≤1e23 at ~6% is slightly low** relative to ≤1e24; the model gives ~8%.

### Cited Findings
- Round-2 numbers (≤1e24 20–25%, ≤1e23 ~6%, ≤1e21 1–2%, first ASI ≤5%, AI engine ~60%, first ASI via ≥1e26 ~95%, and the direction weights) — [round-2 report, 最终排名 and 无条件概率 tables](/home/user/ASI/reports/极低算力超级智能路径深化.md)
- Model base outputs and Table 9 consistency checks — [model output](/home/user/ASI/models/low_compute_asi_model.py):
  - Share of discoveries by 2040 that are AI-led, against the AI/human hazard ratio: 82% at 10×, 75% at 5×, 70% at 3.3×, 62% at 2×, 47% at 1×.
  - The matching headlines: 20.6%, 17.8%, 16.3%, 14.7% and 13.2%.
  - Round-2-implied P(novel form | AI-found) = 13%/60% = 0.22.
  - Direction weights under the alternative mapping: D1 27%, D2 14%, D3 12%, D4 32%.
  - Fat-tailed floor: P(≤1e21 by 2040) 0.95%, by 2050 1.5% (Gaussian: 0.48% and 0.77%).

### Inferences
- **(1) D4 is under-weighted relative to round 2's own engine belief.** Round 2 says ~60% of the learners would be found by big-compute AI, and it justifies raising D4 on the grounds that an AI-found learner "has no reason to look like any human-named paradigm" (DiscoRL invents its own targets). With 60% AI-led discovery, D4 = 13% implies only ~22% of AI-found learners are novel, which contradicts that justification. With p_novel ~0.5, D4 comes to ~25–35% in the model. Caveat: D4 is a "form unknown" bucket. Raising it lowers confidence in *any* nameable answer; it is not a new answer.
- **(2) D1's lead is an artifact of the within-path weights.** Two model features cap D1 at 19–27%. AI-led novelty takes a share of every discovery. And the new, ledger-legal path B, big-compute ASI then re-deriving its paradigm from scratch at falling compute, mostly produces cognitive-core-type (D2) systems. D1 ties or beats D2 only if a discovered learner can never be a cognitive core (Table 9e) or if path B is weak (Gundlach world: D1 22% vs D2 17%). Even then, D4 is larger in those cases.
- **(3) The engine share and the headline must move together.** Round 2 raised P(≤1e24 by 2040) *because* the AI discovery engine got stronger (">2× uplift, DiscoRL, earlier timelines"). In the model, an AI engine strong enough to sustain a ~20% headline makes 75–85% of discoveries AI-led, not 60%. Conversely, an AI-led share of only ~60% corresponds to a ~15% headline. A consistent round-2 pair would be either (~20%, ~80% AI) or (~15%, ~60% AI).
- **(4) Tier 3 (≤1e21) is a tail-shape statement.** Under a Gaussian floor with σ ≈ 1.5 OOM, P(f ≤ 21) ≈ 1%, and only part of that is realised by 2040 (0.5%). Getting to 1–2% by 2040 requires fat tails *and* fast discovery. The 1–2% figure is not wrong so much as unanchored. A better statement is "~0.5–1%, determined almost entirely by the assumed tail of the floor".
- **(5) Tier 2 (≤1e23) vs tier 1.** With a floor spread of ~1.5 OOM, P(≤1e23)/P(≤1e24) ≈ 0.4; round 2's 6/22.5 ≈ 0.27. Keeping round 2's ratio needs a narrower floor (σ ≈ 1 OOM). Nothing in the evidence justifies that, because the Carlsmith range alone is ±2 OOM. ~8% (range 5–10%) is better supported.
- **(6) "First ASI via ≥1e26: ~95%" is slightly high.** The model gives ~90%. About 16% of worlds have the first ASI come from a newly discovered compact learner at *some* size, often 1e24–1e26 (Table 2). Round 2 did not separate this "mid-compute new paradigm first" case.
- **(7) D3 at 15% is high as a "main engine".** In the model, verifier-driven self-improvement appears only as one of the path-A forms (~9%). Round 2 itself says it is "more an inner loop than an independent main engine".
- **What round 2 got right, quantitatively.** P(first ASI low-compute) is small, ~3% strict and ~5% loose. The low-compute ASI is plausibly "a by-product of the first ASI", with a median lag of ~3.6 years. Most successes are AI-era (84%). The Gundlach/Ho dispute matters less than round 2 implied (~2.5 pp).

### Gaps
- These consistency judgments depend on the model's structure: a single discovery event, piecewise-constant hazards, and one shared floor for all learners. A differently structured model could reconcile round 2's numbers differently. For example, if the floor were correlated with discovery difficulty, the (headline, engine) pair would be less constrained.
- The D1-vs-D2 order cannot be settled by modelling. It needs the compute-matched head-to-head experiment that round 2 proposed.

## Q6. Limitations: correlated uncertainties, garbage-in-garbage-out, and what the model cannot say

### Takeaway
The model is a consistency and sensitivity tool, not an oracle. Its headline agrees with round 2 mainly because both encode the same central belief: an ASI floor near 3e24 with ~1.5 OOM of spread. What it adds is an explicit decomposition, a ranking of what matters, and consistency checks. The largest limitations are:
- the two floor inputs (E_max, M_asi) are unsourced and dominate the output;
- independence is assumed between inputs that are plausibly correlated;
- the tier-3 answer depends on tail shape;
- the direction weights are partly circular (inherited from round 2).

### Cited Findings
- Correlation test: coupling a low floor with early automation (ρ = 0.3) raises P(≤1e24 by 2040) from 20.6% to 23.7% and ≤1e23 from 8.3% to 10.1% (Table 8). — [model output](/home/user/ASI/models/low_compute_asi_model.py)
- Tail test: rescaled Student-t(4) floor components roughly double ≤1e21 (0.76% → 1.48% by 2050) and leave ≤1e24 almost unchanged (20.6% → 20.0%) (Table 8). — [model output](/home/user/ASI/models/low_compute_asi_model.py)
- Monte Carlo error on the headline is ±0.06 pp at N = 400k, negligible next to the input uncertainty. — [model output, Table 1](/home/user/ASI/models/low_compute_asi_model.py)
- Round 2 itself warned that its weights and probabilities were subjective, and that many 2025–2026 inputs were snippet-level or self-reported. — [round-2 report, 证据口径说明](/home/user/ASI/reports/极低算力超级智能路径深化.md)

### Inferences
- **Garbage in, garbage out.** E_max (best learner vs brain) and M_asi (breadth multiplier) are assumptions with no direct measurement. Moving E_max's mean down 1 OOM (with a narrower spread) or up 1 OOM moves the headline to ~6% or ~36% respectively (Table 8 "brain-optimal" and "optimistic floor"). Any reader who disagrees with "median best learner ≈ 3× brain" should re-run with their own value. That is a one-line edit to the `INPUTS` list.
- **Correlated uncertainties are under-modelled.**
  - Several inputs probably share hidden drivers. "AI research strength" drives λ_A, m and k together. "Algorithmic tractability of intelligence" drives the floor, T_auto and s_pre together.
  - The base case treats them as independent. Only one correlation (floor ↔ T_auto) is tested, and positive correlations raise the headline and fatten both tails.
  - The tornado's one-at-a-time swings therefore *understate* joint swings. For example, "no AI research edge" (k = m = 1, λ_A = λ_H) cuts the headline to ~6%, much more than any single discovery input does.
- **Structural simplifications.**
  - There is one discovery event per world, and no repeated or partial discoveries.
  - There are no governance pauses, compute-export controls, or deliberate slowdowns.
  - Distillation is excluded by ledger definition. That is correct for the question, but it means the model says nothing about *deployment* cost.
  - Path B starts from the frontier run size, which may overstate its starting point if the first ASI is compute-efficient.
  - FLOP is the only currency. Hardware substrates (D8) cannot help by construction, which matches round 2's ledger logic.
  - "ASI" and "full-pipeline ≤1e24" are treated as crisply measurable. In reality no lab publishes full ledgers (round 2's milestone table).
- **Circularity in the direction weights.** The split among D1/D2/D3/D5/D7/D8 *within* discovered learners is copied from round 2's proportions. Only the between-path structure (path A vs path B, AI vs human, novelty) is model-derived. The model therefore cannot *confirm* round 2's ordering of named forms; it can only show that the ordering is fragile.
- **Snippet-level external inputs.** The timing calibration rests on search snippets of Metaculus, AIFP, METR, AI Impacts and Cotra. Some of them conflict. The timing input is correspondingly wide (σ_ln = 0.9), and the survey-like and AIFP-like scenarios bracket the choice.
- **What the model cannot say.** Whether any particular 2026 system is on the path. What the low-compute learner concretely looks like, beyond the round-2 taxonomy. Anything about safety, deployment or governance, except that compact learner *descriptions* (DiscoRL-style, ~3 MB) are the artifact to watch, which is a round-2 point the model supports: 84% AI-era discovery, and a median lag of 3.6 years after frontier ASI.

### Gaps
- There is no empirical basis for the correlation structure among inputs. ρ = 0.3 is illustrative.
- The model was not validated against any historical analogue, e.g. back-testing the path-B catch-up rate on GPT-3 → small open models under a full ledger. The needed per-model full ledgers exist only for a handful of systems ([compute_ledger.md](/home/user/ASI/research_notes/极低算力超级智能路径深化/compute_ledger.md)).
- A full global sensitivity analysis (Sobol total-order indices, Morris screening) and an explicit decision-theoretic value-of-information calculation were not performed.
