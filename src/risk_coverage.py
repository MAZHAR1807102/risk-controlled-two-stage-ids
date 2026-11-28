"""Empirical checks of the certified threshold rule (reviewer point 2.1).

1. Coverage under i.i.d. sampling (CICIoT2023). Many times: draw a random calibration subset
   of the validation split (10% or 50%), pick tau with the point rule and with the certified
   rule (risk_control.pick), and check it on the full test split, which is large and
   independent. A "violation" is any constrained unit (every family; every type that was
   certifiable in that subset) whose test miss rate is above its target (2% / 5%).
   The certified rule promises a violation probability of at most alpha = 5%.
2. Coverage under temporal dependence (CICIoMT2024, local re-calibration). Many times: in
   every test capture, take a contiguous window of 20% of the rows at a random position as
   calibration data, pick tau, and check it on the remaining 80% of the same captures.
3. Dependence inside the calibration data: at the certified tau, the design effect of the
   miss indicator of each unit in its original row order, deff = Var_block(mean) / Var_binomial,
   with a moving-block bootstrap (blocks of 500 rows). The effective sample size is n / deff.

Usage: .venv/bin/python src/risk_coverage.py   -> results/risk_control/coverage.csv, design_effect.csv
"""
import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_control import EPS, EPS_TYPE, ALPHA, RUNS, gatekeeper_scores, iomt_test_by_file, n_min, pick

OUT = "results/risk_control"
RNG = np.random.default_rng(7)
REPS = 200


def violations(p, fam, typ, tau, cal_typ, cal_fam):
    """Is any constrained unit above its target on the evaluation data?"""
    att = fam != "Benign"
    fam_bad = max((p[fam == g] < tau).mean() for g in np.unique(fam[att])) > EPS
    types, counts = np.unique(cal_typ[cal_fam != "Benign"], return_counts=True)  # attack types only
    certifiable = types[counts >= n_min(EPS_TYPE, ALPHA)]
    typ_bad = any((p[typ == t] < tau).mean() > EPS_TYPE for t in certifiable if (typ == t).any())
    return bool(fam_bad), bool(typ_bad)


def coverage_iid(rows):
    """CICIoT2023: validation and test are random splits of the same data, so pool them and
    split the pool at random into a calibration half and an evaluation half, many times.
    Each half has about as many flows as the validation split. A violation is judged on the
    evaluation half (a finite sample, so this slightly over-counts violations)."""
    ds = get_dataset("ciciot2023")
    Xv, fv, tv, _ = ds.load("validation")
    Xt, ft, tt, _ = ds.load("test")
    pv, pt = gatekeeper_scores(RUNS["ciciot2023"], Xv), gatekeeper_scores(RUNS["ciciot2023"], Xt)
    P, F, T = np.concatenate([pv, pt]), np.concatenate([fv, ft]), np.concatenate([tv, tt])
    for frac in (0.50,):
        for rule in ("point", "ucb_typ"):
            res = []
            for _ in range(REPS):
                cal = RNG.random(len(P)) < frac
                ev = ~cal
                tau, feasible = pick(P[cal], F[cal], T[cal], rule)
                fb, tb = violations(P[ev], F[ev], T[ev], tau, T[cal], F[cal]) if feasible else (False, False)
                res.append((tau, feasible, fb, tb, (P[ev & (F == "Benign")] < tau).mean()))
            r = np.array(res, dtype=float)
            rows.append({"dataset": "ciciot2023", "protocol": f"random {int(frac*100)}% of validation+test as calibration",
                         "rule": rule, "reps": REPS, "feasible": r[:, 1].mean(), "tau_median": np.median(r[:, 0]),
                         "violation_family": r[:, 2].mean(), "violation_type": r[:, 3].mean(),
                         "violation_any": np.maximum(r[:, 2], r[:, 3]).mean(), "bypass_mean": r[:, 4].mean()})
            print(rows[-1], flush=True)
    return pv, fv, tv


def coverage_window(rows):
    X, fam, typ, fid, pos = iomt_test_by_file()
    p = gatekeeper_scores(RUNS["ciciomt2024"], X)
    for rule in ("point", "ucb_typ"):
        res = []
        for _ in range(REPS // 2):
            start = RNG.uniform(0, 0.8, size=fid.max() + 1)[fid]
            cal = (pos >= start) & (pos < start + 0.2)
            tau, feasible = pick(p[cal], fam[cal], typ[cal], rule)
            fb, tb = violations(p[~cal], fam[~cal], typ[~cal], tau, typ[cal], fam[cal]) if feasible else (False, False)
            res.append((tau, feasible, fb, tb, (p[~cal & (fam == "Benign")] < tau).mean()))
        r = np.array(res, dtype=float)
        rows.append({"dataset": "ciciomt2024", "protocol": "random contiguous 20% window per capture", "rule": rule,
                     "reps": REPS // 2, "feasible": r[:, 1].mean(), "tau_median": np.median(r[:, 0]),
                     "violation_family": r[:, 2].mean(), "violation_type": r[:, 3].mean(),
                     "violation_any": np.maximum(r[:, 2], r[:, 3]).mean(), "bypass_mean": r[:, 4].mean()})
        print(rows[-1], flush=True)
    return p, fam, typ, pos


def design_effect(p, fam, typ, tau, name, L=500, B=300):
    out = []
    for level, labels in (("family", fam), ("type", typ)):
        for u in np.unique(labels[fam != "Benign"]):
            m = (p[labels == u] < tau).astype(float)  # original row order
            n, k = len(m), int(m.sum())
            if k < 5 or n < 4 * L:
                continue
            starts = np.arange(n - L + 1)
            means = []
            for _ in range(B):
                s = RNG.choice(starts, size=int(np.ceil(n / L)))
                means.append(m[(s[:, None] + np.arange(L)[None, :]).ravel()[:n]].mean())
            q = k / n
            deff = np.var(means) / (q * (1 - q) / n)
            out.append({"dataset": name, "level": level, "unit": u, "n": n, "missed": k, "rate": q,
                        "deff": deff, "n_eff": n / max(deff, 1.0)})
    return out


def main():
    rows = []
    pv, fv, tv = coverage_iid(rows)
    p, fam, typ, pos = coverage_window(rows)
    pd.DataFrame(rows).to_csv(f"{OUT}/coverage.csv", index=False)

    rc = pd.read_csv(f"{OUT}/summary.csv")
    tau_c = lambda d, cal: float(rc[(rc.dataset == d) & (rc.calibration == cal) & (rc.rule == "ucb_typ")].tau.iloc[0])
    de = design_effect(pv, fv, tv, tau_c("ciciot2023", "validation"), "ciciot2023 validation")
    cal = pos < 0.2
    de += design_effect(p[cal], fam[cal], typ[cal], tau_c("ciciomt2024", "first 20% of test files"),
                        "ciciomt2024 first 20% of test captures")
    pd.DataFrame(de).to_csv(f"{OUT}/design_effect.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(de).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
