#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
low_compute_asi_model.py -- a transparent Monte Carlo / Fermi model of low-compute ASI (as of Sept 2026).

Question (final report, in Chinese):
    以极低算力（全程：训练+推理+借来的算力）实现通用超级智能的最可能方向是什么？为什么？

What the model does
-------------------
It turns the evidence gathered in two earlier research rounds into explicit input distributions,
propagates them by Monte Carlo, and reports:
  * P(an ASI whose FULL-PIPELINE instance compute is <= 1e24 / 1e23 / 1e21 FLOP exists by 2035/2040/2050);
  * P(the first ASI is low-compute);
  * the share of low-compute successes by path and discovery engine, and the resulting "direction weights";
  * a tornado table (one input at a time pinned to its p10/p90) and first-order sensitivity indices
    (share of outcome variance that learning one input would remove, a value-of-information proxy).

Ledger convention (from round 2, research_notes/极低算力超级智能路径深化/compute_ledger.md):
  full-pipeline = own pretraining + own RL/post-training + inference + ALL borrowed compute
  (teacher/base pretraining, distillation, synthetic data, self-play). Discovery compute (finding the
  learner) goes in a separate ledger and is not charged. Only a <= genome-sized description (~5e8 bits)
  may cross from the discovery ledger to the instance ledger. Distilling a big ASI into a small model
  therefore inherits the teacher's compute; it is NOT a low-compute route under this ledger.

Model structure (each Monte Carlo sample is one possible world; all compute values are log10 FLOP)
-----------------------------------------------------------------------------------------------
  floor  f  = logCbrain + logM - logE
      The lowest full-pipeline compute at which ANY learner could reach ASI (brain lifetime compute
      x ASI breadth multiplier / best-possible efficiency advantage over the brain's algorithm).
  Frontier ("big-compute") ASI:
      Full automation of AI R&D at T_auto; frontier ASI at T_F = T_auto + gap, using C_F = FC(T_F),
      where FC(t) is the frontier training-run size.
  Path A, "a complete compact learner is discovered":
      Discovery time T_d comes from a hazard process: a human-led rate lamH from now on; an AI-led rate
      lamA after T_auto, multiplied by m after T_F. The discovered learner becomes usable at
      T_d + lag at cost f + G (first version), then is refined at max(s(t), r_fresh) OOM/yr down to f.
  Path B, "cognitive-core catch-up" (big-then-re-derive-from-scratch; no distillation):
      After T_F, the frontier paradigm is re-derived at small scale. It starts at C_F - logK
      (core + retrieval discount) and falls at s(t) OOM/yr down to its own floor f + P_B.
  Small-scale algorithmic progress s(t), in OOM/yr: s_pre before T_auto, s_pre*sqrt(k) between
      T_auto and T_F, and s_pre*k after T_F.
  An ASI at <= X exists at the first time either path's cost is <= X (solved analytically; no time grid).

Run:  python3 low_compute_asi_model.py            (full run, numpy backend if available)
      python3 low_compute_asi_model.py --quick    (smaller N)
      python3 low_compute_asi_model.py --no-numpy (force the pure-stdlib backend)
      python3 low_compute_asi_model.py --selftest (checks that the numpy and stdlib engines agree)

Every input below cites its evidential source, or says "Assumption". The outputs are only as good as
these inputs (garbage in, garbage out). See the notes file for the limitations.
"""

import argparse
import math
import random
import sys
import time

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # the stdlib fallback is used automatically
    np = None
    HAVE_NUMPY = False

SEED = 20260926
T0 = 2026.75                 # "now" = end of Sept 2026
THRESHOLDS = (24, 23, 21)    # log10 FLOP full-pipeline targets (tier 1 / tier 2 / tier 3 of round 2)
YEARS = (2035, 2040, 2050)
INF = math.inf

# ---------------------------------------------------------------------------------------------
# Frontier training-run size FC(t), log10 FLOP. Used for C_F (the compute of the first big ASI) and
# to check whether a path-A learner is trainable at all.
#   * 2026 frontier: 5e26-3e27 (assumption; the largest Epoch-recorded run is still Grok 4 at 5.0e26,
#     compute_ledger.md Q1/Q3) -> 27.0 central, with per-world noise `fc_noise`.
#   * +0.6 OOM/yr (4x/yr) to 2030, consistent with Epoch's 4-5x/yr trend and its "2e29 feasible by 2030"
#     (theory_limits.md Q1).
#   * +0.25 OOM/yr after 2030, capped at 10^30.5 (assumption: power and capital constraints).
FC_2026, FC_SLOPE1, FC_FAST_YEARS, FC_SLOPE2, FC_CAP = 27.0, 0.6, 4.0, 0.25, 30.5

# ---------------------------------------------------------------------------------------------
# Direction taxonomy (round 2 report, final ranking table) and round-2 subjective weights (percent).
DIRS = [
    ("D1", "World-model continual-learning agent", "经验中持续学习的世界模型智能体"),
    ("D2", "Compact cognitive core (+RLVR, retrieval)", "紧凑认知内核"),
    ("D3", "Verifier-driven self-improvement", "验证器驱动的自我改进"),
    ("D4", "AI-discovered novel paradigm", "AI发现的新范式"),
    ("D5", "Radical brain-inspired", "激进类脑范式"),
    ("D6", "Mainstream efficiency stack", "主流效率堆栈"),
    ("D7", "Tiny recursive models", "微型递归模型"),
    ("D8", "New compute substrate", "新型计算基底"),
]
ROUND2_WEIGHTS = [30, 24, 15, 13, 7, 6, 3, 2]
ROUND2_HEADLINE = {"p24_2040": "20-25%", "p23_2040": "~6%", "p21_2040": "1-2%",
                   "first_low": "<=5%", "engine_ai": "~60%", "first_big": "~95%"}

# Path-A forms when the discovered learner resembles a human-named paradigm: D1, D2, D3, D5, D7, D8 in
# round-2 relative proportions 30:24:15:7:3:2 (D6 is incremental by definition, so it is path B only; D4 is
# the "unlike any named paradigm" outcome of AI-led discovery). CIRCULAR INPUT: inherited from round 2, not
# independent evidence. The model's own contribution is the split BETWEEN paths and engines (which sets D4
# and the D6 share), not the split within path A.
PATH_A_FORMS = [0, 1, 2, 4, 6, 7]                  # indices into DIRS
PATH_A_BASE = [x / 81.0 for x in (30, 24, 15, 7, 3, 2)]
DIRICHLET_CONC = 20.0                              # per-world noise around the base split (assumption)

# ---------------------------------------------------------------------------------------------
# INPUT DISTRIBUTIONS. kind: "normal"(mean, sd) | "uniform"(lo, hi) | "lognormal"(median, sigma_ln).
# Log-uniform quantities are written as uniform on log10.
INPUTS = [
    # (1) Brain-equivalent lifetime compute. Carlsmith (2020): 1e15 FLOP/s "more likely than not" enough;
    #     mechanistic range 1e13-1e17 FLOP/s; x ~9.5e8 s (30 yr) -> 1e22-1e26, central 1e24. Cotra's lifetime
    #     anchor median is ~1e24. [theory_limits.md Q1; round-2 report section 1.] The 1e22-1e26 range is
    #     treated as ~+-2 sd.
    ("logCbrain", "normal", 24.0, 1.0,
     "log10 FLOP, brain lifetime compute (~30 yr)",
     "Carlsmith 1e13-1e17 FLOP/s x 30 yr; Cotra lifetime anchor ~1e24 [theory_limits Q1]"),
    # (2) Multiplier for superhuman BREADTH + DEPTH over one human lifetime (one human is top-level in at
    #     most a few domains; ASI must beat top humans at most cognitive tasks). Lifetime inference is also
    #     charged: 1e15 FLOP/s for 1 yr is ~3e22. Assumption. Anchors: round 2 says <=1e24 ASI needs
    #     "superhuman breadth without proportionally more experience"; TD-MPC2's multitask score grows ~log-
    #     linearly with size (no "small is enough" break). Median 10x, 90% CI ~1.5x-70x.
    ("logM", "normal", 1.0, 0.5,
     "log10 x, ASI breadth/depth multiplier over C_brain",
     "Assumption; round-2 report (breadth w/o proportional experience); TD-MPC2 scaling"),
    # (3) Efficiency of the best possible learner relative to the brain's own algorithm (per-sample compute
    #     and sample efficiency combined). For >1: narrow AI already beats the brain by 3-6 OOM (KataGo 1e21,
    #     EfficientZero 1e18/game; compute_ledger Q4), and the brain spends much of its compute on non-
    #     cognitive work. For <1: no general system below ~3e24 without borrowing (DeepSeek-V3); compressed
    #     priors only shift learning curves (Shuvaev et al.; compressed_priors section 6); measured prior
    #     multipliers are 1.3-10x. Assumption. Median ~3x, 90% CI ~0.07x-150x.
    ("logE", "normal", 0.5, 1.0,
     "log10 x, best learner efficiency vs brain",
     "Assumption; narrow 3-6 OOM wins vs no general <3e24 [compute_ledger Q4; compressed_priors 6]"),
    # (4) Extra floor penalty for the LLM-lineage (path B) relative to the best learner: LLMs are ~1e4-1e5x
    #     less sample-efficient, only partly offset by cheaper per-sample compute (round-2 "sample
    #     efficiency != compute efficiency"). Assumption: 1x-30x.
    ("P_B", "uniform", 0.0, 1.5,
     "log10 x, path-B paradigm floor penalty",
     "Assumption; round-2 per-sample compute comparison (brain 4e15/word vs V3 2.2e11/token)"),
    # (5) One-time discount from a compact reasoning core plus retrieval versus a monolithic frontier model
    #     of the same capability: GPT-4 2.1e25 -> V3 3.3e24 (6x, not only architecture); RETRO ~25x fewer
    #     parameters (parameters, not FLOP); VibeThinker (self-reported, hidden teacher compute).
    #     Assumption: 1x-30x.
    ("logK", "uniform", 0.0, 1.5,
     "log10 x, core+retrieval discount at T_F",
     "Assumption; V3 vs GPT-4 6x; RETRO 25x params; VibeThinker (self-rep.) [red_team Q2]"),
    # (6) The gap between the FIRST discovered compact learner and the floor. First versions of a paradigm
    #     are typically 10-100x off: AlphaGo Zero 3-day 2.7e22 -> KataGo ~1e21 (~1.4 OOM in ~1.5 yr);
    #     GPT-4 -> V3 ~0.8 OOM in 21 months [compute_ledger Q4]. Assumption: 1x-100x.
    ("G", "uniform", 0.0, 2.0,
     "log10 x, first-version gap above floor",
     "Assumption; AGZ->KataGo, GPT-4->V3 refinement [compute_ledger Q4]"),
    # (7) Refinement rate of a freshly discovered learner (low-hanging fruit): GPT-4 -> V3 ~0.37 OOM/yr at
    #     full pipeline; AGZ(3-day) -> KataGo ~1 OOM/yr (narrow); Ho et al. 0.45 OOM/yr.
    ("r_fresh", "uniform", 0.2, 1.0,
     "OOM/yr, post-discovery refinement rate",
     "GPT-4->V3 0.37/yr; AGZ->KataGo ~1/yr; Ho 0.45/yr [compute_ledger Q4; theory_limits Q3]"),
    # (8) Small-scale ("from scratch, no teacher") algorithmic progress before AI R&D automation.
    #     Lower end: Gundlach et al. 2025, <100x over 2012-2023 at small scale, i.e. <1.52x/yr = 0.18
    #     OOM/yr. Upper end: Ho et al. 8-month halving = 2.8x/yr = 0.45 OOM/yr (CI 1.8-5.3x), and the full-
    #     ledger GPT-4 -> V3 result of 2.8x/yr, measured at the relevant 1e24-1e25 scale.
    #     [theory_limits Q3; round-2 report section 2]
    ("s_pre", "uniform", 0.12, 0.50,
     "OOM/yr, small-scale algorithmic progress (pre-automation)",
     "Gundlach <1.5x/yr (0.18) .. Ho/V3-ledger 2.8x/yr (0.45) [theory_limits Q3]"),
    # (9) Acceleration of small-scale progress once AI does AI research: log-rate multiplier k after T_F,
    #     sqrt(k) between T_auto and T_F. Forethought: a software intelligence explosion could add ~12 OOM of
    #     effective compute (excluding paradigm shifts) (snippet). Epoch: the compute-bottleneck objection is
    #     "suggestive but shaky" (snippet). Cunningham et al. 2026: loops "not yet self-sustaining" (snippet).
    #     METR: >2x researcher uplift in 2026 (snippet). Assumption: k in [1, 10], log-uniform.
    ("log_k", "uniform", 0.0, 1.0,
     "log10 x, AI-era speed-up of small-scale progress",
     "Assumption; Forethought SIE ~12 OOM; Cunningham not self-sustaining; METR >2x (snippets) [red_team Q1,Q5]"),
    # (10) Years from Sept 2026 to full automation of AI R&D (superhuman AI researcher). Median ~2032.75.
    #      AIFP Q1-2026: full AI R&D automation median ~early 2031, 25th pct ~mid-2028 (snippet); METR:
    #      99% AI R&D automation median late 2032 (snippet); Metaculus "general AI" median Jan 2033, 25% by
    #      2029 (snippet); Cotra 2022: TAI 50% by 2040; AI Impacts 2023 survey: HLMI 50% by 2047, 10% by
    #      2027. sigma_ln 0.9 gives p10 ~2028.6 and p90 ~2046, which covers the survey tail. The median leans
    #      toward the 2026 forecaster community (AIFP/METR/Metaculus) rather than the 2023 academic survey;
    #      see the "Slow frontier" scenario for the survey-like alternative.
    ("yrs_auto", "lognormal", 6.0, 0.9,
     "years from 2026.75 to full AI R&D automation",
     "AIFP ~2031, METR ~2032.9, Metaculus 2033, Cotra 2040, AI Impacts 2047 (snippets)"),
    # (11) Take-off gap from full AI R&D automation to ASI. Metaculus weak-AGI -> superintelligence
    #      ~42.6 months (snippet; that gap starts from an earlier milestone); AI-2027 takeoff ~1 yr.
    #      Median 1.5 yr, p10 ~0.5, p90 ~4.2.
    ("gap", "lognormal", 1.5, 0.8,
     "years from T_auto to frontier ASI",
     "Metaculus weak AGI->ASI ~42.6 mo (snippet); AI-2027 takeoff ~1 yr (snippet)"),
    # (12) Noise on the frontier run size (2026 frontier 5e26-3e27 is an assumption).
    ("fc_noise", "normal", 0.0, 0.4,
     "log10 x, frontier run-size noise",
     "Assumption; Grok 4 5e26 record; 2026 frontier 5e26-3e27 assumed [compute_ledger Q3]"),
    # (13) Human-led hazard (per year) of discovering a COMPLETE compact general learner (not a component).
    #      No success in ~70 years of AI; only components exist (DiscoRL, VeLO); OaK has no integrated demo;
    #      BabyLM is far below children; Byrnes expects a brain-like paradigm possibly within decades.
    #      Assumption: 0.2%-3%/yr, log-uniform (median ~0.8%/yr).
    ("log_lamH", "uniform", math.log10(0.002), math.log10(0.03),
     "log10 /yr, human-led discovery hazard",
     "Assumption; only component-level discoveries to date [compressed_priors 2,6; round-2 report]"),
    # (14) AI-led hazard (per year) after full automation. For: DiscoRL found a SOTA RL rule for ~1e22 FLOP;
    #      "train from scratch under budget X, then score" is a cheap evaluator; a complete-learner search is
    #      estimated at 1e25-1e26 FLOP, affordable post-automation. Against: labs chase scale-dependent frontier
    #      gains first (Gundlach pattern); Goodhart on small-budget evaluations; ASI-Arch gains were modest.
    #      Assumption: 2%-30%/yr, log-uniform (median ~7.7%/yr).
    ("log_lamA", "uniform", math.log10(0.02), math.log10(0.30),
     "log10 /yr, AI-led discovery hazard (post-automation)",
     "Assumption; DiscoRL ~1e22 discovery; AIDE2; ASI-Arch modest [red_team Q1; compressed_priors 2]"),
    # (15) Multiplier on the AI-led hazard once a frontier ASI exists (ASI-level researchers). Assumption: 1-5x.
    ("log_m", "uniform", 0.0, math.log10(5.0),
     "log10 x, AI-led hazard multiplier after T_F",
     "Assumption"),
    # (16) Lag from discovery to a trained, verified ASI-level instance. Training 1e24 FLOP takes ~1 month
    #      on 1,000 H100s at 40% utilisation (my estimate), so the lag is research and verification, not compute.
    #      Assumption: 0.5-2 yr.
    ("lag", "uniform", 0.5, 2.0,
     "years, discovery -> verified instance",
     "Assumption; 1e24 FLOP ~ 1 month on 1k H100 (my estimate)"),
    # (17) P(an AI-discovered learner looks unlike any human-named paradigm, i.e. counts as D4). DiscoRL's rule
    #      invents its own prediction targets (red_team Q1). Assumption: 0.3-0.7.
    ("p_novel", "uniform", 0.3, 0.7,
     "prob., AI-found learner is a novel form (D4)",
     "Assumption; DiscoRL invents own prediction targets [red_team Q1]"),
    # (18) Among path-B successes, the share that is a compact cognitive core (D2) rather than the plain
    #      mainstream efficiency stack (D6). Assumption.
    ("p_core", "uniform", 0.6, 0.9,
     "prob., path-B success is D2 (else D6)",
     "Assumption; round-2 D2 vs D6 = 24:6"),
    # (19) Among path-B successes, the share whose capability mainly comes from post-"birth" interactive
    #      learning, so that round 2's criterion classifies it as D1. Chollet (Sep 2026): the harness is
    #      "moving inside the model"; Cursor real-time RL (self-reported). Assumption: 0-0.3.
    ("p_B_wm", "uniform", 0.0, 0.3,
     "prob., path-B success is classified D1",
     "Assumption; functions moving into big models [red_team Q4,Q6]"),
]
INPUT_NAMES = [x[0] for x in INPUTS]
SPEC = {x[0]: (x[1], x[2], x[3]) for x in INPUTS}
DOC = {x[0]: (x[4], x[5]) for x in INPUTS}

# The five human-readable "families" of inputs, used for grouping in the sensitivity tables.
FAMILY = {
    "logCbrain": "floor", "logM": "floor", "logE": "floor", "P_B": "pathB", "logK": "pathB",
    "G": "pathA", "r_fresh": "pathA", "s_pre": "progress", "log_k": "progress",
    "yrs_auto": "timeline", "gap": "timeline", "fc_noise": "timeline",
    "log_lamH": "discovery", "log_lamA": "discovery", "log_m": "discovery", "lag": "pathA",
    "p_novel": "mapping", "p_core": "mapping", "p_B_wm": "mapping",
}
FLOOR_SD = math.sqrt(1.0 ** 2 + 0.5 ** 2 + 1.0 ** 2)   # sd of f under base specs (used for rho)


# =============================================================================================
# Sampling
# =============================================================================================
class RNG:
    """Thin wrapper so that the same sampling code runs with numpy or with the stdlib `random`."""

    def __init__(self, seed, use_numpy):
        self.np = use_numpy
        self.r = np.random.default_rng(seed) if use_numpy else random.Random(seed)

    def normal(self, n):
        return self.r.standard_normal(n) if self.np else [self.r.gauss(0.0, 1.0) for _ in range(n)]

    def uniform(self, n):
        return self.r.random(n) if self.np else [self.r.random() for _ in range(n)]

    def expo(self, n):
        return self.r.standard_exponential(n) if self.np else [self.r.expovariate(1.0) for _ in range(n)]

    def t4(self, n):
        # Student-t with 4 d.o.f., rescaled so that its 5-95% interval equals a standard normal's.
        scale = 1.6449 / 2.1318
        if self.np:
            return self.r.standard_t(4, n) * scale
        out = []
        for _ in range(n):
            z = self.r.gauss(0.0, 1.0)
            chi2 = sum(self.r.gauss(0.0, 1.0) ** 2 for _ in range(4))
            out.append(z / math.sqrt(chi2 / 4.0) * scale)
        return out

    def dirichlet(self, alpha, n):
        if self.np:
            return self.r.dirichlet(alpha, n)
        rows = []
        for _ in range(n):
            g = [self.r.gammavariate(a, 1.0) for a in alpha]
            s = sum(g)
            rows.append([x / s for x in g])
        return rows


def _vmap(fn, *cols):
    """Element-wise map for the stdlib backend (numpy arrays are handled by broadcasting)."""
    return [fn(*vals) for vals in zip(*cols)]


def sample_world(n, seed=SEED, use_numpy=HAVE_NUMPY, overrides=None, rho=0.0, fat_tail=False):
    """Draw n worlds. `overrides` maps an input name to a replacement spec tuple or ("const", value).
    rho correlates a low floor with early AI R&D automation (Gaussian copula). fat_tail=True replaces
    the normal draws for logCbrain and logE with rescaled Student-t(4) draws."""
    spec = dict(SPEC)
    if overrides:
        spec.update(overrides)
    g = RNG(seed, use_numpy)
    S, Z = {}, {}
    # Draw the standard variates in a FIXED order, so that different scenarios share random numbers
    # (common random numbers). This reduces noise in the comparisons.
    for name in INPUT_NAMES:
        if fat_tail and name in ("logCbrain", "logE"):
            Z[name] = g.t4(n)
            g.normal(n)  # burn the same number of normals to keep the later streams aligned
        else:
            Z[name] = g.normal(n)
        Z[name + "_u"] = g.uniform(n)
    extras = {"E_exp": g.expo(n), "u_eng": g.uniform(n), "u_nov": g.uniform(n),
              "u_formA": g.uniform(n), "u_formB": g.uniform(n), "u_formB2": g.uniform(n)}
    dir_rows = g.dirichlet([DIRICHLET_CONC * p for p in PATH_A_BASE], n)

    # Standardised floor shock (negative = cheaper floor), used only when rho != 0.
    zf_parts = (Z["logCbrain"], Z["logM"], Z["logE"])
    if use_numpy:
        z_floor = (1.0 * zf_parts[0] + 0.5 * zf_parts[1] - 1.0 * zf_parts[2]) / FLOOR_SD
    else:
        z_floor = _vmap(lambda a, b, c: (a + 0.5 * b - c) / FLOOR_SD, *zf_parts)

    for name in INPUT_NAMES:
        kind, a, b = (tuple(spec[name]) + (None,))[:3]   # ("const", v) has no third field
        z, u = Z[name], Z[name + "_u"]
        if name == "yrs_auto" and rho != 0.0:
            c = math.sqrt(1.0 - rho * rho)
            z = (rho * z_floor + c * z) if use_numpy else _vmap(lambda zf, zz: rho * zf + c * zz, z_floor, z)
        if kind == "const":
            S[name] = (np.full(n, float(a)) if use_numpy else [float(a)] * n)
        elif kind == "normal":
            S[name] = (a + b * z) if use_numpy else [a + b * v for v in z]
        elif kind == "uniform":
            S[name] = (a + (b - a) * u) if use_numpy else [a + (b - a) * v for v in u]
        elif kind == "lognormal":
            S[name] = (a * np.exp(b * z)) if use_numpy else [a * math.exp(b * v) for v in z]
        else:
            raise ValueError(kind)
    S.update(extras)
    S["dir_probs"] = dir_rows
    return S


# =============================================================================================
# Core model -- numpy (vectorised) version
# =============================================================================================
def fc_np(t, noise):
    y = np.maximum(t - T0, 0.0)
    base = FC_2026 + FC_SLOPE1 * np.minimum(y, FC_FAST_YEARS) + FC_SLOPE2 * np.maximum(0.0, y - FC_FAST_YEARS)
    return np.minimum(FC_CAP, base) + noise


def ttd_np(t_start, amount, TA, TF, r1, r2, r3):
    """Time at which a cost that starts falling at t_start has fallen by `amount` OOM, when the rate is
    r1 before TA, r2 in [TA, TF), and r3 after TF."""
    amount = np.maximum(amount, 0.0)
    cap1 = r1 * np.maximum(0.0, TA - t_start)
    start2 = np.maximum(t_start, TA)
    cap2 = r2 * np.maximum(0.0, TF - start2)
    start3 = np.maximum(t_start, TF)
    return np.where(amount <= cap1, t_start + amount / r1,
                    np.where(amount <= cap1 + cap2, start2 + (amount - cap1) / r2,
                             start3 + (amount - cap1 - cap2) / r3))


def decline_np(t1, t2, TA, TF, r1, r2, r3):
    """OOM of cost reduction accumulated between t1 and t2 (0 if t2 <= t1)."""
    def ov(a, b, c, d):
        return np.maximum(0.0, np.minimum(b, d) - np.maximum(a, c))
    return (r1 * ov(t1, t2, -np.inf, TA) + r2 * ov(t1, t2, TA, TF) + r3 * ov(t1, t2, TF, np.inf))


def model_np(S):
    f = S["logCbrain"] + S["logM"] - S["logE"]          # ASI floor (any learner)
    fB = f + S["P_B"]                                     # path-B floor
    TA = T0 + S["yrs_auto"]                               # full AI R&D automation
    TF = TA + S["gap"]                                    # frontier (big-compute) ASI
    CF = fc_np(TF, S["fc_noise"])                         # its full-pipeline compute
    k, m = 10.0 ** S["log_k"], 10.0 ** S["log_m"]
    lamH, lamA = 10.0 ** S["log_lamH"], 10.0 ** S["log_lamA"]
    s1 = S["s_pre"]; s2 = s1 * np.sqrt(k); s3 = s1 * k
    rf = S["r_fresh"]
    r1, r2, r3 = np.maximum(s1, rf), np.maximum(s2, rf), np.maximum(s3, rf)

    # Discovery time from a piecewise-constant hazard (inverse of the cumulative hazard at an Exp(1) draw).
    E = S["E_exp"]
    H1 = lamH * (TA - T0)
    H2 = H1 + (lamH + lamA) * (TF - TA)
    Td = np.where(E <= H1, T0 + E / lamH,
                  np.where(E <= H2, TA + (E - H1) / (lamH + lamA), TF + (E - H2) / (lamH + lamA * m)))
    pAI = np.where(E <= H1, 0.0, np.where(E <= H2, lamA / (lamH + lamA), lamA * m / (lamH + lamA * m)))
    engAI = S["u_eng"] < pAI                               # competing-risks attribution of the engine
    Tav = Td + S["lag"]
    A0 = f + S["G"]
    B0 = np.maximum(fB, CF - S["logK"])

    out = {"f": f, "TA": TA, "TF": TF, "CF": CF, "Td": Td, "engAI": engAI}
    for X in THRESHOLDS:
        tA = np.where(f > X, np.inf, ttd_np(Tav, A0 - X, TA, TF, r1, r2, r3))
        tB = np.where(fB > X, np.inf, TF + np.maximum(0.0, B0 - X) / s3)
        out["T%d" % X] = np.minimum(tA, tB)
        out["viaA%d" % X] = tA <= tB

    # First ASI of any size: the frontier at TF, or path A once its cost fits under the frontier run size.
    # Approximation: if the new learner is too big to train at Tav, we wait until its cost falls below the
    # run size AT Tav (the frontier is held flat), which is conservative. In the ~0.1% of worlds where
    # f > FC(Tav) this is slightly wrong; it is immaterial to every reported number.
    FCav = fc_np(Tav, S["fc_noise"])
    TAany = np.where(A0 <= FCav, Tav, ttd_np(Tav, A0 - FCav, TA, TF, r1, r2, r3))  # conservative
    Tfirst = np.minimum(TF, TAany)
    out["TAany"], out["Tfirst"] = TAany, Tfirst
    out["first_compute"] = np.where(TAany < TF, np.maximum(f, np.minimum(A0, FCav)), CF)

    # Cheapest ASI that exists in a given year (inf = no ASI at all).
    for Y in YEARS:
        a = np.where(Tav <= Y, np.maximum(f, A0 - decline_np(Tav, Y, TA, TF, r1, r2, r3)), np.inf)
        a = np.where(a <= fc_np(np.full_like(a, float(Y)), S["fc_noise"]), a, np.inf)
        b = np.where(TF <= Y, np.maximum(fB, B0 - s3 * (Y - TF)), np.inf)
        c = np.where(TF <= Y, CF, np.inf)
        out["cheap%d" % Y] = np.minimum(np.minimum(a, b), c)

    # Direction (form) of the learner, for the direction weights.
    P = S["dir_probs"]
    cum = np.cumsum(P, axis=1)
    idx = (S["u_formA"][:, None] > cum).sum(axis=1)
    idx = np.minimum(idx, len(PATH_A_FORMS) - 1)
    human_form = np.array(PATH_A_FORMS)[idx]
    formA = np.where(engAI & (S["u_nov"] < S["p_novel"]), 3, human_form)
    formB = np.where(S["u_formB"] < S["p_B_wm"], 0, np.where(S["u_formB2"] < S["p_core"], 1, 5))
    out["formA"], out["formB"] = formA, formB
    return out


# =============================================================================================
# Core model -- pure-Python (scalar) version: the same equations, one world at a time
# =============================================================================================
def fc_py(t, noise):
    y = max(t - T0, 0.0)
    base = FC_2026 + FC_SLOPE1 * min(y, FC_FAST_YEARS) + FC_SLOPE2 * max(0.0, y - FC_FAST_YEARS)
    return min(FC_CAP, base) + noise


def ttd_py(t_start, amount, TA, TF, r1, r2, r3):
    amount = max(amount, 0.0)
    cap1 = r1 * max(0.0, TA - t_start)
    if amount <= cap1:
        return t_start + amount / r1
    start2 = max(t_start, TA)
    cap2 = r2 * max(0.0, TF - start2)
    if amount <= cap1 + cap2:
        return start2 + (amount - cap1) / r2
    return max(t_start, TF) + (amount - cap1 - cap2) / r3


def decline_py(t1, t2, TA, TF, r1, r2, r3):
    def ov(a, b, c, d):
        return max(0.0, min(b, d) - max(a, c))
    return r1 * ov(t1, t2, -INF, TA) + r2 * ov(t1, t2, TA, TF) + r3 * ov(t1, t2, TF, INF)


def model_one(p):
    """p: dict of scalars for one world. Returns a dict of scalars (same keys as model_np)."""
    f = p["logCbrain"] + p["logM"] - p["logE"]
    fB = f + p["P_B"]
    TA = T0 + p["yrs_auto"]
    TF = TA + p["gap"]
    CF = fc_py(TF, p["fc_noise"])
    k, m = 10.0 ** p["log_k"], 10.0 ** p["log_m"]
    lamH, lamA = 10.0 ** p["log_lamH"], 10.0 ** p["log_lamA"]
    s1 = p["s_pre"]; s2 = s1 * math.sqrt(k); s3 = s1 * k
    rf = p["r_fresh"]
    r1, r2, r3 = max(s1, rf), max(s2, rf), max(s3, rf)
    E = p["E_exp"]
    H1 = lamH * (TA - T0)
    H2 = H1 + (lamH + lamA) * (TF - TA)
    if E <= H1:
        Td, pAI = T0 + E / lamH, 0.0
    elif E <= H2:
        Td, pAI = TA + (E - H1) / (lamH + lamA), lamA / (lamH + lamA)
    else:
        Td, pAI = TF + (E - H2) / (lamH + lamA * m), lamA * m / (lamH + lamA * m)
    engAI = p["u_eng"] < pAI
    Tav = Td + p["lag"]
    A0 = f + p["G"]
    B0 = max(fB, CF - p["logK"])
    out = {"f": f, "TA": TA, "TF": TF, "CF": CF, "Td": Td, "engAI": engAI}
    for X in THRESHOLDS:
        tA = INF if f > X else ttd_py(Tav, A0 - X, TA, TF, r1, r2, r3)
        tB = INF if fB > X else TF + max(0.0, B0 - X) / s3
        out["T%d" % X] = min(tA, tB)
        out["viaA%d" % X] = tA <= tB
    FCav = fc_py(Tav, p["fc_noise"])
    TAany = Tav if A0 <= FCav else ttd_py(Tav, A0 - FCav, TA, TF, r1, r2, r3)
    Tfirst = min(TF, TAany)
    out["TAany"], out["Tfirst"] = TAany, Tfirst
    out["first_compute"] = max(f, min(A0, FCav)) if TAany < TF else CF
    for Y in YEARS:
        a = max(f, A0 - decline_py(Tav, Y, TA, TF, r1, r2, r3)) if Tav <= Y else INF
        if a > fc_py(float(Y), p["fc_noise"]):
            a = INF
        b = max(fB, B0 - s3 * (Y - TF)) if TF <= Y else INF
        c = CF if TF <= Y else INF
        out["cheap%d" % Y] = min(a, b, c)
    probs = p["dir_probs"]
    acc, idx = 0.0, len(PATH_A_FORMS) - 1
    for i, q in enumerate(probs):
        acc += q
        if p["u_formA"] <= acc:
            idx = i
            break
    human_form = PATH_A_FORMS[idx]
    out["formA"] = 3 if (engAI and p["u_nov"] < p["p_novel"]) else human_form
    if p["u_formB"] < p["p_B_wm"]:
        out["formB"] = 0
    elif p["u_formB2"] < p["p_core"]:
        out["formB"] = 1
    else:
        out["formB"] = 5
    return out


def model_py(S):
    n = len(S["E_exp"])
    keys = [k for k in S if k != "dir_probs"]
    rows = []
    for i in range(n):
        p = {k: S[k][i] for k in keys}
        p["dir_probs"] = S["dir_probs"][i]
        rows.append(model_one(p))
    return {k: [r[k] for r in rows] for k in rows[0]}


def run_model(S, use_numpy):
    return model_np(S) if use_numpy else model_py(S)


# =============================================================================================
# Aggregation helpers (these work for numpy arrays and for lists)
# =============================================================================================
def _arr(x):
    return x if (HAVE_NUMPY and isinstance(x, np.ndarray)) else list(x)


def mean_of(flags):
    if HAVE_NUMPY and isinstance(flags, np.ndarray):
        return float(np.mean(flags))
    flags = list(flags)
    return sum(1 for v in flags if v) / len(flags)


def le(xs, c):
    if HAVE_NUMPY and isinstance(xs, np.ndarray):
        return xs <= c
    return [v <= c for v in xs]


def land(a, b):
    if HAVE_NUMPY and isinstance(a, np.ndarray):
        return a & b
    return [x and y for x, y in zip(a, b)]


def lnot(a):
    if HAVE_NUMPY and isinstance(a, np.ndarray):
        return ~a
    return [not x for x in a]


def select(xs, mask):
    if HAVE_NUMPY and isinstance(xs, np.ndarray):
        return xs[mask]
    return [x for x, keep in zip(xs, mask) if keep]


def quantiles(xs, qs):
    xs = sorted(list(xs))
    n = len(xs)
    if n == 0:
        return [float("nan")] * len(qs)
    res = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(math.floor(pos)); hi = min(lo + 1, n - 1)
        a, b = xs[lo], xs[hi]
        if math.isinf(a) or math.isinf(b):
            res.append(a if (pos - lo) < 0.5 else b)
        else:
            res.append(a + (b - a) * (pos - lo))
    return res


def headline(O):
    """The three headline scalars used in the sensitivity analysis."""
    return {
        "p24_2040": mean_of(le(O["T24"], 2040.0)),
        "p23_2040": mean_of(le(O["T23"], 2040.0)),
        "first_low": mean_of(strict_first(O, 24)),
    }


def strict_first(O, X):
    """True if an ASI at <= X already exists at the moment the first ASI of any size exists."""
    T, Tf = O["T%d" % X], O["Tfirst"]
    if HAVE_NUMPY and isinstance(T, np.ndarray):
        return np.isfinite(T) & (T <= Tf + 1e-9)
    return [(not math.isinf(t)) and t <= tf + 1e-9 for t, tf in zip(T, Tf)]


def direction_weights(O, X, Y):
    """Among worlds where an ASI at <= X exists by Y: the fraction whose main engine is each direction."""
    T, viaA, fA, fB = O["T%d" % X], O["viaA%d" % X], O["formA"], O["formB"]
    counts = [0] * len(DIRS)
    tot = 0
    if HAVE_NUMPY and isinstance(T, np.ndarray):
        ok = T <= Y
        form = np.where(viaA, fA, fB)[ok]
        tot = int(ok.sum())
        for i in range(len(DIRS)):
            counts[i] = int((form == i).sum())
    else:
        for t, va, a, b in zip(T, viaA, fA, fB):
            if t <= Y:
                tot += 1
                counts[a if va else b] += 1
    return [c / tot if tot else float("nan") for c in counts], tot


def engine_split(O, X, Y):
    """Among successes (<= X by Y): shares via path A (human-led), path A (AI-led) and path B."""
    T, viaA, eng = O["T%d" % X], O["viaA%d" % X], O["engAI"]
    h = a = b = 0
    for t, va, e in zip(_arr(T), _arr(viaA), _arr(eng)):
        if t <= Y:
            if va and not e:
                h += 1
            elif va and e:
                a += 1
            else:
                b += 1
    tot = h + a + b
    return (h / tot, a / tot, b / tot, tot) if tot else (float("nan"),) * 3 + (0,)


# =============================================================================================
# Sensitivity analysis
# =============================================================================================
def input_quantiles(S, name, qs=(0.1, 0.5, 0.9)):
    return quantiles(S[name], qs)


def tornado(S, use_numpy, base):
    """One-at-a-time sensitivity: pin each input at its sampled p10 and p90 (all other inputs and all
    event draws unchanged, i.e. common random numbers) and record the headline outcomes."""
    rows = []
    n = len(S["E_exp"])
    for name in INPUT_NAMES:
        lo, hi = input_quantiles(S, name, (0.1, 0.9))
        res = {}
        for tag, val in (("lo", lo), ("hi", hi)):
            S2 = dict(S)
            S2[name] = (np.full(n, val) if use_numpy else [val] * n)
            O2 = run_model(S2, use_numpy)
            res[tag] = headline(O2)
            res[tag]["dirw"] = direction_weights(O2, 24, 2050.0)[0]   # weights among <=1e24-by-2050 successes
        swing = abs(res["hi"]["p24_2040"] - res["lo"]["p24_2040"])
        rows.append((name, lo, hi, res["lo"], res["hi"], swing))
    rows.sort(key=lambda r: -r[5])
    return rows


def first_order_index(S, indicator, name, bins=20):
    """Binned estimate of S1 = Var(E[I|X]) / Var(I): the share of the variance of the outcome indicator
    that would be removed if this input were known exactly (a value-of-information proxy). A finite-sample
    noise term is subtracted."""
    if HAVE_NUMPY and isinstance(S[name], np.ndarray):
        x = S[name]; I = np.asarray(indicator, dtype=float)
        pbar = I.mean(); var = pbar * (1 - pbar)
        if var <= 0:
            return 0.0
        chunks = np.array_split(I[np.argsort(x, kind="stable")], bins)
        w = np.array([len(c) for c in chunks], dtype=float) / len(I)
        mb = np.array([c.mean() for c in chunks])
        nb = np.array([len(c) for c in chunks], dtype=float)
        between = float(np.sum(w * (mb - pbar) ** 2))
        noise = float(np.sum(w * mb * (1 - mb) / nb))
        return max(0.0, (between - noise) / var)
    x = list(S[name]); I = [1.0 if v else 0.0 for v in indicator]
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    pbar = sum(I) / n
    var = pbar * (1 - pbar)
    if var <= 0:
        return 0.0
    between, noise = 0.0, 0.0
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        nb = len(idx)
        mb = sum(I[i] for i in idx) / nb
        between += nb / n * (mb - pbar) ** 2
        noise += nb / n * mb * (1 - mb) / nb
    return max(0.0, (between - noise) / var)


# =============================================================================================
# Reporting
# =============================================================================================
def pct(x, d=1):
    return "   n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else ("%5.*f%%" % (d, 100 * x))


def fmt_log(v):
    return "none" if math.isinf(v) else "1e%.1f" % v


def print_inputs(S):
    print("\nTABLE 0. Input distributions as sampled (p10 / p50 / p90), with evidential source")
    print("-" * 118)
    print("%-10s %-50s %8s %8s %8s  %s" % ("input", "meaning", "p10", "p50", "p90", "source"))
    for name in INPUT_NAMES:
        q = input_quantiles(S, name)
        meaning, src = DOC[name]
        print("%-10s %-50s %8.3f %8.3f %8.3f  %s" % (name, meaning[:50], q[0], q[1], q[2], src))
    fl = [a + b - c for a, b, c in zip(_arr(S["logCbrain"]), _arr(S["logM"]), _arr(S["logE"]))]
    q = quantiles(fl, (0.05, 0.1, 0.5, 0.9, 0.95))
    print("-> derived ASI floor f = logCbrain+logM-logE: p5 %.2f  p10 %.2f  p50 %.2f  p90 %.2f  p95 %.2f"
          % tuple(q))
    print("   P(f<=24) = %s   P(f<=23) = %s   P(f<=21) = %s   (upper bounds on 'ever')" %
          (pct(mean_of([v <= 24 for v in fl])), pct(mean_of([v <= 23 for v in fl])),
           pct(mean_of([v <= 21 for v in fl]), 2)))


def summarize(O, S, label="BASE"):
    n = len(O["T24"])
    print("\nTABLE 1. [%s] P(an ASI with full-pipeline compute <= X exists by year Y)   (N=%d)" % (label, n))
    print("-" * 86)
    print("%-28s %10s %10s %10s   %s" % ("threshold", "2035", "2040", "2050", "round-2 (2040)"))
    r2 = {24: ROUND2_HEADLINE["p24_2040"], 23: ROUND2_HEADLINE["p23_2040"], 21: ROUND2_HEADLINE["p21_2040"]}
    for X in THRESHOLDS:
        vals = [mean_of(le(O["T%d" % X], float(Y))) for Y in YEARS]
        print("%-28s %10s %10s %10s   %s" % ("<= 1e%d FLOP" % X, pct(vals[0], 2), pct(vals[1], 2),
                                              pct(vals[2], 2), r2[X]))
    vals = [mean_of(le(O["Tfirst"], float(Y))) for Y in YEARS]
    print("%-28s %10s %10s %10s" % ("any ASI (context)", pct(vals[0]), pct(vals[1]), pct(vals[2])))
    vals = [mean_of(le(O["TF"], float(Y))) for Y in YEARS]
    print("%-28s %10s %10s %10s" % ("big-compute frontier ASI", pct(vals[0]), pct(vals[1]), pct(vals[2])))
    p = mean_of(le(O["T24"], 2040.0))
    print("MC standard error on P(<=1e24 by 2040): +-%.2f pp" % (100 * math.sqrt(p * (1 - p) / n)))

    print("\nTABLE 2. [%s] Is the first ASI low-compute?" % label)
    print("-" * 86)
    s24 = mean_of(strict_first(O, 24)); s23 = mean_of(strict_first(O, 23))
    loose = mean_of([t < tf for t, tf in zip(_arr(O["T24"]), _arr(O["TF"]))])
    via_new = mean_of([ta < tf for ta, tf in zip(_arr(O["TAany"]), _arr(O["TF"]))])
    print("P(first ASI already achievable at <=1e24)  [strict]          %s   round 2: %s" %
          (pct(s24, 2), ROUND2_HEADLINE["first_low"]))
    print("P(first ASI already achievable at <=1e23)  [strict]          %s" % pct(s23, 2))
    print("P(a <=1e24 ASI exists before the big-compute frontier ASI)   %s  [loose]" % pct(loose, 2))
    print("P(first ASI comes from a newly discovered compact learner, any size) %s" % pct(via_new, 2))
    any40 = le(O["Tfirst"], 2040.0)
    big = [fc >= 26 for fc in _arr(O["first_compute"])]
    n_any = sum(1 for v in _arr(any40) if v)
    big_given = sum(1 for a, b in zip(_arr(any40), big) if a and b) / max(1, n_any)
    print("P(first ASI used >=1e26 full-pipeline | some ASI by 2040)    %s   round 2: %s" %
          (pct(big_given), ROUND2_HEADLINE["first_big"]))
    # lag between the frontier ASI and the first <=1e24 ASI
    lags = [t - tf for t, tf in zip(_arr(O["T24"]), _arr(O["TF"])) if not math.isinf(t) and t >= tf]
    if lags:
        q = quantiles(lags, (0.1, 0.5, 0.9))
        print("Lag from frontier ASI to first <=1e24 ASI (when it comes after): p10 %.1f / p50 %.1f / p90 %.1f yr"
              % tuple(q))

    print("\nTABLE 3. [%s] Engine and path of low-compute success" % label)
    print("-" * 86)
    for (X, Y) in ((24, 2040), (24, 2050), (23, 2040)):
        h, a, b, tot = engine_split(O, X, float(Y))
        print("<=1e%d by %d (n=%6d): path A human-led %s | path A AI-led %s | path B core catch-up %s"
              % (X, Y, tot, pct(h), pct(a), pct(b)))
    disc = le(O["Td"], 2040.0)
    nd = sum(1 for v in _arr(disc) if v)
    ai = sum(1 for d, e in zip(_arr(disc), _arr(O["engAI"])) if d and e)
    print("P(complete compact learner discovered by 2040) %s; of which AI-led %s   (round 2: %s)"
          % (pct(nd / len(_arr(disc))), pct(ai / max(1, nd)), ROUND2_HEADLINE["engine_ai"]))
    h, a, b, tot = engine_split(O, 24, 2040.0)
    print("Share of <=1e24-by-2040 successes produced in the AI-research era (path A AI-led + path B): %s"
          % pct(a + b))

    print("\nTABLE 4. [%s] Direction weights = P(direction is the main engine | low-compute ASI achieved)" % label)
    print("-" * 104)
    w40, n40 = direction_weights(O, 24, 2040.0)
    w50, n50 = direction_weights(O, 24, 2050.0)
    w23, n23 = direction_weights(O, 23, 2050.0)
    print("%-4s %-44s %9s %9s %9s %9s" % ("", "direction", "<=1e24/40", "<=1e24/50", "<=1e23/50", "round 2"))
    for i, (code, en, zh) in enumerate(DIRS):
        print("%-4s %-44s %9s %9s %9s %8d%%" % (code, en + " " * 0, pct(w40[i]), pct(w50[i]), pct(w23[i]),
                                                ROUND2_WEIGHTS[i]))
    print("(n successes: %d / %d / %d)" % (n40, n50, n23))

    print("\nTABLE 5. [%s] Cheapest ASI in existence (full-pipeline log10 FLOP), by year" % label)
    print("-" * 86)
    for Y in YEARS:
        vals = _arr(O["cheap%d" % Y])
        fin = [v for v in vals if not math.isinf(v)]
        share = len(fin) / len(vals)
        q = quantiles(fin, (0.1, 0.5, 0.9)) if fin else [float("nan")] * 3
        print("%d: some ASI exists in %s of worlds; conditional on existing, cheapest p10 %s / p50 %s / p90 %s"
              % (Y, pct(share), fmt_log(q[0]), fmt_log(q[1]), fmt_log(q[2])))


def print_tornado(rows, base):
    print("\nTABLE 6. Tornado: pin each input at its p10 / p90 (others unchanged; common random numbers)")
    print("Base: P(<=1e24 by 2040) %s | P(<=1e23 by 2040) %s | P(first ASI <=1e24, strict) %s"
          % (pct(base["p24_2040"]), pct(base["p23_2040"]), pct(base["first_low"], 2)))
    print("-" * 118)
    print("%-10s %-9s %8s %8s | %-17s | %-17s | %-17s | %s" %
          ("input", "family", "p10", "p90", "P<=1e24/40 lo->hi", "P<=1e23/40 lo->hi", "first-low lo->hi",
           "swing(24/40)"))
    for name, lo, hi, rl, rh, sw in rows:
        print("%-10s %-9s %8.3f %8.3f | %6s -> %6s | %6s -> %6s | %6s -> %6s | %5.1f pp" %
              (name, FAMILY[name], lo, hi, pct(rl["p24_2040"]), pct(rh["p24_2040"]),
               pct(rl["p23_2040"]), pct(rh["p23_2040"]), pct(rl["first_low"], 1), pct(rh["first_low"], 1),
               100 * sw))

    print("\nTABLE 6b. Direction-weight tornado: weights among <=1e24-by-2050 successes, input at p10 -> p90")
    print("         (sorted by the largest swing in any of D1/D2/D4)")
    print("-" * 104)
    print("%-10s %-9s | %-17s | %-17s | %-17s | %-17s" %
          ("input", "family", "D1 world-model", "D2 cog. core", "D4 AI-novel", "D6 mainstream"))
    drows = []
    for name, lo, hi, rl, rh, sw in rows:
        a, b = rl["dirw"], rh["dirw"]
        dsw = max(abs(b[i] - a[i]) for i in (0, 1, 3))
        drows.append((dsw, name, a, b))
    drows.sort(key=lambda r: -r[0])
    for dsw, name, a, b in drows:
        if dsw < 0.005:
            continue
        print("%-10s %-9s | %6s -> %6s | %6s -> %6s | %6s -> %6s | %6s -> %6s" %
              (name, FAMILY[name], pct(a[0]), pct(b[0]), pct(a[1]), pct(b[1]), pct(a[3]), pct(b[3]),
               pct(a[5]), pct(b[5])))


def print_s1(S, O):
    ind24 = le(O["T24"], 2040.0)
    ind23 = le(O["T23"], 2040.0)
    indF = strict_first(O, 24)
    rows = []
    for name in INPUT_NAMES:
        rows.append((name, first_order_index(S, _arr(ind24), name), first_order_index(S, _arr(ind23), name),
                     first_order_index(S, _arr(indF), name)))
    # The floor as a single derived quantity (it cannot be measured component by component)
    if HAVE_NUMPY and isinstance(S["logCbrain"], np.ndarray):
        fl = S["logCbrain"] + S["logM"] - S["logE"]
    else:
        fl = [a + b - c for a, b, c in zip(_arr(S["logCbrain"]), _arr(S["logM"]), _arr(S["logE"]))]
    S_tmp = {"floor_f": fl}
    rows.append(("floor_f*", first_order_index(S_tmp, _arr(ind24), "floor_f"),
                 first_order_index(S_tmp, _arr(ind23), "floor_f"), first_order_index(S_tmp, _arr(indF), "floor_f")))
    rows.sort(key=lambda r: -r[1])
    print("\nTABLE 7. First-order sensitivity index S1 = Var(E[outcome|input]) / Var(outcome)")
    print("         (= expected share of outcome variance removed by learning that input exactly; VOI proxy)")
    print("-" * 86)
    print("%-10s %-9s %14s %14s %16s" % ("input", "family", "P<=1e24/2040", "P<=1e23/2040", "first ASI<=1e24"))
    for name, a, b, c in rows:
        fam = FAMILY.get(name, "derived")
        print("%-10s %-9s %14.3f %14.3f %16.3f" % (name, fam, a, b, c))
    tot = sum(r[1] for r in rows if r[0] != "floor_f*")
    print("Sum of S1 over the 19 primitive inputs (P<=1e24/2040): %.2f  (1 - sum = interactions + event noise)" % tot)

    # Practical VOI: how far the headline moves if evidence places an input in its low/middle/high tercile.
    print("\nTABLE 7b. Headline conditional on an input's tercile (what the answer becomes if evidence pins it)")
    print("-" * 104)
    print("%-10s | %-32s | %-32s | %s" % ("input", "P(<=1e24 by 2040): lo/mid/hi", "P(<=1e23 by 2040): lo/mid/hi",
                                          "tercile cut-points"))
    cols = [("floor_f*", _arr(fl)), ("yrs_auto", _arr(S["yrs_auto"])), ("log_lamA", _arr(S["log_lamA"])),
            ("log_k", _arr(S["log_k"])), ("s_pre", _arr(S["s_pre"])), ("logM", _arr(S["logM"]))]
    i24, i23 = _arr(ind24), _arr(ind23)
    for name, xs in cols:
        c1, c2 = quantiles(xs, (1 / 3, 2 / 3))
        out24, out23 = [], []
        for lo_b, hi_b in ((-INF, c1), (c1, c2), (c2, INF)):
            m = [lo_b < x <= hi_b for x in xs]
            nsel = sum(1 for v in m if v)
            out24.append(sum(1 for v, s in zip(i24, m) if s and v) / max(1, nsel))
            out23.append(sum(1 for v, s in zip(i23, m) if s and v) / max(1, nsel))
        print("%-10s | %8s %8s %8s        | %8s %8s %8s        | %.2f / %.2f" %
              (name, pct(out24[0]), pct(out24[1]), pct(out24[2]), pct(out23[0]), pct(out23[1]), pct(out23[2]),
               c1, c2))


SCENARIOS = [
    ("BASE (independent inputs)", {}, 0.0, False),
    ("rho=0.3: cheap floor <-> early automation", {}, 0.3, False),
    ("Fat-tailed floor (t4 on C_brain, E)", {}, 0.0, True),
    ("Gundlach world: s_pre=0.15, k=1", {"s_pre": ("const", 0.15), "log_k": ("const", 0.0)}, 0.0, False),
    ("Ho world: s_pre=0.45 (k as base)", {"s_pre": ("const", 0.45)}, 0.0, False),
    ("No AI research edge: k=1, m=1, lamA=lamH range",
     {"log_k": ("const", 0.0), "log_m": ("const", 0.0),
      "log_lamA": ("uniform", math.log10(0.002), math.log10(0.03))}, 0.0, False),
    ("Slow frontier (survey-like, median ~2040)", {"yrs_auto": ("lognormal", 13.0, 0.7)}, 0.0, False),
    ("Fast frontier (AIFP-like, median ~2029)", {"yrs_auto": ("lognormal", 2.5, 0.6)}, 0.0, False),
    ("Brain-optimal: logE ~ N(-0.5, 0.5)", {"logE": ("normal", -0.5, 0.5)}, 0.0, False),
    ("Optimistic floor: logE ~ N(1.5, 1.0)", {"logE": ("normal", 1.5, 1.0)}, 0.0, False),
    ("Slow AI discovery: lamA 0.5%-5%/yr",
     {"log_lamA": ("uniform", math.log10(0.005), math.log10(0.05))}, 0.0, False),
    ("Low novelty: p_novel ~ U(0.1, 0.3)", {"p_novel": ("uniform", 0.1, 0.3)}, 0.0, False),
]


def print_scenarios(n, use_numpy):
    print("\nTABLE 8. Scenarios (each re-sampled with the same seed; N=%d)" % n)
    print("-" * 124)
    print("%-46s %8s %8s %8s %8s %8s %8s %9s %6s %6s %6s %6s" %
          ("scenario", "24/2035", "24/2040", "24/2050", "23/2040", "21/2040", "21/2050", "first<=24",
           "D1", "D2", "D3", "D4"))
    for label, ov, rho, fat in SCENARIOS:
        S = sample_world(n, SEED, use_numpy, ov, rho, fat)
        O = run_model(S, use_numpy)
        w, _ = direction_weights(O, 24, 2050.0)
        print("%-46s %8s %8s %8s %8s %8s %8s %9s %6s %6s %6s %6s" %
              (label[:46], pct(mean_of(le(O["T24"], 2035.0))), pct(mean_of(le(O["T24"], 2040.0))),
               pct(mean_of(le(O["T24"], 2050.0))), pct(mean_of(le(O["T23"], 2040.0))),
               pct(mean_of(le(O["T21"], 2040.0)), 2), pct(mean_of(le(O["T21"], 2050.0)), 2),
               pct(mean_of(strict_first(O, 24)), 2),
               pct(w[0], 0), pct(w[1], 0), pct(w[2], 0), pct(w[3], 0)))
    print("(D1/D2/D3/D4 columns = direction weights conditional on <=1e24 by 2050)")


def print_consistency(n, use_numpy):
    """Checks used to test the internal consistency of the round-2 subjective numbers."""
    global PATH_A_FORMS, PATH_A_BASE
    print("\nTABLE 9. Consistency checks against round 2 (N=%d each)" % n)
    print("-" * 118)
    S = sample_world(n, SEED, use_numpy)
    O = run_model(S, use_numpy)
    TF, T24 = _arr(O["TF"]), _arr(O["T24"])
    fr = [t <= 2040 for t in TF]
    a = sum(1 for f_, t in zip(fr, T24) if f_ and t <= 2040) / max(1, sum(fr))
    b = sum(1 for f_, t in zip(fr, T24) if (not f_) and t <= 2040) / max(1, len(fr) - sum(fr))
    print("(a) P(<=1e24 by 2040 | frontier ASI by 2040) = %s ; P(<=1e24 by 2040 | no frontier ASI by 2040) = %s"
          % (pct(a), pct(b)))
    q = quantiles(TF, (0.1, 0.25, 0.5, 0.75, 0.9))
    qa = quantiles(_arr(O["TA"]), (0.1, 0.25, 0.5, 0.75, 0.9))
    print("(b) Frontier ASI year p10/p25/p50/p75/p90: %.1f / %.1f / %.1f / %.1f / %.1f" % tuple(q))
    print("    AI R&D automation year p10/p25/p50/p75/p90: %.1f / %.1f / %.1f / %.1f / %.1f" % tuple(qa))
    srt = sorted(T24)
    yrs = []
    for p in (0.05, 0.10, 0.20, 0.30):
        v = srt[int(p * len(srt))]
        yrs.append("%d%% by %s" % (int(100 * p), "never" if math.isinf(v) else "%.1f" % v))
    print("(c) Year at which P(<=1e24 ASI exists) reaches: " + "; ".join(yrs))
    # (d) Which AI-led/human-led hazard ratio reproduces round 2's "~60% of discoveries are AI-led"?
    print("(d) Scaling the AI-led hazard down: share of discoveries by 2040 that are AI-led, and the headline")
    lamH_med = 10 ** ((math.log10(0.002) + math.log10(0.03)) / 2)
    for r in (1, 2, 3, 5, 10):
        lo, hi = math.log10(0.02 / r), math.log10(0.30 / r)
        S2 = sample_world(n, SEED, use_numpy, {"log_lamA": ("uniform", lo, hi)})
        O2 = run_model(S2, use_numpy)
        d = [t <= 2040 for t in _arr(O2["Td"])]
        nd = sum(d)
        ai = sum(1 for x, e in zip(d, _arr(O2["engAI"])) if x and e) / max(1, nd)
        med = 10 ** ((lo + hi) / 2)
        print("    lamA/%-2d median %5.2f%%/yr (%4.1fx lamH): discovered by 2040 %s, AI-led %s, P(<=1e24 by 2040) %s"
              % (r, 100 * med, med / lamH_med, pct(nd / len(d)), pct(ai), pct(mean_of(le(O2["T24"], 2040.0)))))
    print("    Round-2 implied P(novel form | AI-found) = D4 13%% / AI engine 60%% = %.2f" % (0.13 / 0.60))
    # (e) Mapping ambiguity: what if a discovered compact learner can never count as a cognitive core (D2)?
    saved = (PATH_A_FORMS, PATH_A_BASE)
    PATH_A_FORMS, PATH_A_BASE = [0, 2, 4, 6, 7], [x / 57.0 for x in (30, 15, 7, 3, 2)]
    try:
        S3 = sample_world(n, SEED, use_numpy)
        w, _ = direction_weights(run_model(S3, use_numpy), 24, 2050.0)
    finally:
        PATH_A_FORMS, PATH_A_BASE = saved
    w0, _ = direction_weights(O, 24, 2050.0)
    print("(e) Direction weights (<=1e24 by 2050) if path-A learners may be D2 [base] vs may NOT be D2 [variant]:")
    print("    " + "  ".join("%s %s/%s" % (DIRS[i][0], pct(w0[i], 0).strip(), pct(w[i], 0).strip())
                             for i in range(len(DIRS))))


def selftest():
    if not HAVE_NUMPY:
        print("numpy not available; self-test skipped")
        return True
    n = 3000
    S = sample_world(n, 7, True)
    O1 = model_np(S)
    S_list = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in S.items()}
    S_list["dir_probs"] = [list(r) for r in S["dir_probs"]]
    O2 = model_py(S_list)
    ok = True
    for key in O1:
        a = np.asarray(O1[key], dtype=float); b = np.asarray(O2[key], dtype=float)
        with np.errstate(invalid="ignore"):
            same = np.all((a == b) | (np.isinf(a) & np.isinf(b)) | (np.abs(a - b) < 1e-9))
        if not same:
            ok = False
            print("MISMATCH in", key)
    print("self-test (numpy engine vs stdlib engine on identical inputs, n=%d): %s" % (n, "PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=None, help="Monte Carlo worlds for the main run")
    ap.add_argument("--quick", action="store_true", help="smaller N everywhere")
    ap.add_argument("--no-numpy", action="store_true", help="force the pure-stdlib backend")
    ap.add_argument("--selftest", action="store_true", help="check numpy and stdlib engines agree")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    use_numpy = HAVE_NUMPY and not args.no_numpy
    if use_numpy:
        n_main, n_sens, n_scen = (60000, 30000, 30000) if args.quick else (400000, 200000, 200000)
    else:
        n_main, n_sens, n_scen = (6000, 3000, 3000) if args.quick else (40000, 15000, 15000)
    if args.n:
        n_main = args.n
    t_start = time.time()
    print("=" * 118)
    print("LOW-COMPUTE ASI MONTE CARLO MODEL  (as of Sept 2026; seed %d; backend: %s)" %
          (SEED, "numpy " + np.__version__ if use_numpy else "stdlib random"))
    print("Full-pipeline ledger: training + inference + ALL borrowed (teacher/base/distillation/synthetic) compute.")
    print("All probabilities are model outputs conditional on the stated input distributions -- not facts.")
    print("=" * 118)

    S = sample_world(n_main, SEED, use_numpy)
    print_inputs(S)
    O = run_model(S, use_numpy)
    summarize(O, S)

    if not args.skip_sensitivity:
        Ss = sample_world(n_sens, SEED + 1, use_numpy)
        Os = run_model(Ss, use_numpy)
        base = headline(Os)
        rows = tornado(Ss, use_numpy, base)
        print_tornado(rows, base)
        print_s1(S, O)
        print_scenarios(n_scen, use_numpy)
        print_consistency(n_scen, use_numpy)
    print("\n[done in %.1f s]" % (time.time() - t_start))


if __name__ == "__main__":
    main()
