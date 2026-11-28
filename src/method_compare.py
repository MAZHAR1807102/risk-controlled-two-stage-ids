"""Our family-wise rule against standard ways of picking a risk-controlled threshold.

Same Gatekeeper scores (runs/seed42_sqrt), same data: CICIoT2023 validation + test pooled and split at
random 200 times into a calibration half and an evaluation half (as in risk_coverage.coverage_iid).
Every method picks one threshold tau on the calibration half; the target is "every attack family
skipped at most eps = 2%". Methods (risk = share of attack flows skipped):
  point          largest tau with every family's observed rate <= eps (our efficiency point)
  crc_marginal   conformal risk control on all attacks: (n R + 1)/(n + 1) <= eps (expectation, marginal)
  sgr_marginal   selective classification (Geifman & El-Yaniv): CP bound on all attacks <= eps, conf. 1-alpha
  crc_family     group-conditional (Mondrian) conformal risk control: the CRC rule in every family
                 (holds in expectation per family, not with high probability)
  ltt_hoeffding  Learn-then-Test: Hoeffding upper bound per family, Bonferroni over families, conf. 1-alpha
  cp_family      ours: Clopper-Pearson upper bound per family, Bonferroni over families, conf. 1-alpha
  ltt_binom      Learn-then-Test with exact binomial p-values, fixed sequence over tau, every family at
                 the full alpha (intersection-union; simultaneous over families, valid under (A2))
  pac_mondrian   PAC (high-probability) Mondrian calibration: each family's CP bound at alpha, a
                 guarantee for each family separately, not simultaneous. Same thresholds as ltt_binom
                 here; listed apart because it promises less.
The fixed-split results (calibrate on validation, test once) are reported too.

Usage: .venv/bin/python src/method_compare.py  -> results/method_compare/{coverage,fixed}_v2.csv
"""
import os

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_control import ALPHA, EPS, GRID, RUNS, cp_upper, gatekeeper_scores, miss_counts

OUT = "results/method_compare"
RNG = np.random.default_rng(13)
REPS = 200
METHODS = ["point", "crc_marginal", "sgr_marginal", "crc_family", "ltt_hoeffding", "cp_family", "ltt_binom",
           "pac_mondrian"]


def pick(method, p, fam):
    att = fam != "Benign"
    if method in ("crc_marginal", "sgr_marginal"):
        n, k = miss_counts(p[att], np.zeros(att.sum()), GRID)[0.0]
        ok = (k + 1) / (n + 1) <= EPS if method == "crc_marginal" else cp_upper(k, n, ALPHA) <= EPS
    else:
        cnt = miss_counts(p[att], fam[att], GRID)
        a = ALPHA / len(cnt)
        ok = np.ones(len(GRID), bool)
        for n, k in cnt.values():
            if method == "point":
                ok &= k / n <= EPS
            elif method == "crc_family":
                ok &= (k + 1) / (n + 1) <= EPS
            elif method in ("ltt_binom", "pac_mondrian"):
                ok &= cp_upper(k, n, ALPHA) <= EPS
            elif method == "ltt_hoeffding":
                ok &= k / n + np.sqrt(np.log(1 / a) / (2 * n)) <= EPS
            else:
                ok &= cp_upper(k, n, a) <= EPS
    if not ok.any():
        return 0.0, False
    return float(GRID[np.flatnonzero(ok).max()]), True


def evaluate(p, fam, tau):
    att = fam != "Benign"
    worst = max((p[fam == g] < tau).mean() for g in np.unique(fam[att]))
    return worst, float((p[fam == "Benign"] < tau).mean()), float((p[att] < tau).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = get_dataset("ciciot2023")
    Xv, fv, _, _ = ds.load("validation")
    Xt, ft, _, _ = ds.load("test")
    pv, pt = gatekeeper_scores(RUNS["ciciot2023"], Xv), gatekeeper_scores(RUNS["ciciot2023"], Xt)

    fixed = []
    for m in METHODS:
        tau, ok = pick(m, pv, fv)
        worst, byp, overall = evaluate(pt, ft, tau) if ok else (0.0, 0.0, 0.0)
        fixed.append({"method": m, "feasible": ok, "tau": tau, "test_worst_family": worst, "test_overall_skip": overall,
                      "bypass": byp})
    pd.DataFrame(fixed).to_csv(f"{OUT}/fixed_v2.csv", index=False)
    print(pd.DataFrame(fixed).round(4).to_string(index=False))

    P, F = np.concatenate([pv, pt]), np.concatenate([fv, ft])
    res = {m: [] for m in METHODS}
    for _ in range(REPS):
        cal = RNG.random(len(P)) < 0.5
        for m in METHODS:
            tau, ok = pick(m, P[cal], F[cal])
            worst, byp, _ = evaluate(P[~cal], F[~cal], tau) if ok else (0.0, 0.0, 0.0)
            res[m].append((ok, worst > EPS, byp, worst))
    rows = []
    for m in METHODS:
        r = np.array(res[m], dtype=float)
        rows.append({"method": m, "reps": REPS, "feasible": r[:, 0].mean(), "family_violation": r[:, 1].mean(),
                     "bypass_mean": r[:, 2].mean(), "worst_family_mean": r[:, 3].mean()})
    cov = pd.DataFrame(rows)
    cov.to_csv(f"{OUT}/coverage_v2.csv", index=False)
    print(cov.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
