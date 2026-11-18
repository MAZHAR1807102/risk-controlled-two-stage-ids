"""How much does dependence between calibration flows matter for the certified threshold? (review 3, point 2.1)

1. Is there time order to test at all? CICIoT2023 rows are shuffled in the published CSV (no capture or
   time id), so a block bootstrap there cannot detect clustering. CICIoMT2024 keeps each capture in time order.
2. Design effect at the boundary. The proof needs the miss count k_u at the boundary threshold tau_u^+ (where
   the true miss rate reaches its target) to be binomial. For every constrained unit of CICIoMT2024 we set
   tau_u so that its miss rate over the test captures equals its target, and measure
   deff = Var_block(mean miss) / Var_binomial with a moving-block bootstrap in time order (blocks of 100 / 500).
3. Sensitivity. Re-run the certified rule with the counts divided by a design effect D (k/D misses out of
   n/D flows: the usual effective-sample-size correction), for D = 1, 1.5, 2, 4, 8 and for the measured
   per-unit D, and report the threshold, bypass and test misses; and repeat the CICIoMT2024 window-coverage
   experiment with the corrected rule.

Usage: .venv/bin/python src/risk_dependence.py
Output: results/risk_control/dependence_order.csv, dependence_deff.csv, dependence_sensitivity.csv,
        dependence_coverage.csv
"""
import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_control import ALPHA, EPS, EPS_TYPE, GRID, RUNS, cp_upper, evaluate, gatekeeper_scores, iomt_test_by_file, n_min
from risk_coverage import violations

OUT = "results/risk_control"
RNG = np.random.default_rng(11)


def same_label_neighbours(labels):
    """Share of neighbouring rows with the same label, and the share expected if the rows were shuffled."""
    same = float((labels[1:] == labels[:-1]).mean())
    p = pd.Series(labels).value_counts(normalize=True)
    return same, float((p ** 2).sum())


def deff_block(m, L, B=400):
    n, q = len(m), m.mean()
    if q in (0.0, 1.0) or n < 4 * L:
        return np.nan
    starts = np.arange(n - L + 1)
    means = [m[(s[:, None] + np.arange(L)[None, :]).ravel()[:n]].mean()
             for s in (RNG.choice(starts, size=int(np.ceil(n / L))) for _ in range(B))]
    return float(np.var(means) / (q * (1 - q) / n))


def units(fam, typ, cal_typ=None, cal_fam=None):
    """Constrained units: every attack family, and every attack type certifiable on the calibration data."""
    att = fam != "Benign"
    out = [("family", g, EPS) for g in np.unique(fam[att])]
    ct, cf = (typ, fam) if cal_typ is None else (cal_typ, cal_fam)
    types, counts = np.unique(ct[cf != "Benign"], return_counts=True)
    out += [("type", t, EPS_TYPE) for t in types[counts >= n_min(EPS_TYPE, ALPHA / 2 / len(types))]]
    return out


def pick_D(p, fam, typ, D=1.0):
    """risk_control.pick(rule='ucb_typ') with every count divided by a design effect.
    D is a number, or a dict unit -> D (missing units use 1)."""
    d = (lambda u: max(1.0, D.get(u, 1.0))) if isinstance(D, dict) else (lambda u: max(1.0, D))
    att = fam != "Benign"
    types, counts = np.unique(typ[att], return_counts=True)
    certifiable = [t for t, c in zip(types, counts) if c / d(t) >= n_min(EPS_TYPE, ALPHA / 2 / len(types))]
    fams = list(np.unique(fam[att]))
    a_fam, a_typ = ALPHA / 2 / len(fams), ALPHA / 2 / max(len(certifiable), 1)
    ok = np.ones(len(GRID), bool)
    for lab, unit_list, a, eps in ((fam, fams, a_fam, EPS), (typ, certifiable, a_typ, EPS_TYPE)):
        for u in unit_list:
            s = np.sort(p[lab == u])
            k = np.searchsorted(s, GRID, side="left").astype(float)
            ok &= cp_upper(k / d(u), len(s) / d(u), a) <= eps
    if not ok.any():
        return float(GRID[0]), False
    return float(GRID[np.flatnonzero(ok).max()]), True


def main():
    rows_order, rows_deff, rows_sens, rows_cov = [], [], [], []

    # ---- data ----
    ds = get_dataset("ciciot2023")
    Xv, fv, tv, _ = ds.load("validation")
    Xt, ft, tt, _ = ds.load("test")
    pv, pt = gatekeeper_scores(RUNS["ciciot2023"], Xv), gatekeeper_scores(RUNS["ciciot2023"], Xt)
    X, fam, typ, fid, pos = iomt_test_by_file()
    p = gatekeeper_scores(RUNS["ciciomt2024"], X)

    # ---- 1. time order ----
    for name, labels in (("ciciot2023 validation (published CSV)", tv), ("ciciomt2024 test captures", typ)):
        same, exp = same_label_neighbours(labels)
        rows_order.append({"data": name, "same_label_neighbours": same, "expected_if_shuffled": exp})
    # inside one CICIoMT2024 capture the label is constant, so look at the miss indicator instead (below)

    # ---- 2. design effect at the boundary threshold, CICIoMT2024 time order ----
    D_measured = {}
    for level, u, eps in units(fam, typ):
        lab = fam if level == "family" else typ
        sel = lab == u                              # rows stay in capture (time) order
        tau_u = float(np.quantile(p[sel], eps))     # miss rate of u over the test captures = its target
        m = (p[sel] < tau_u).astype(float)
        r = {"level": level, "unit": u, "n": int(sel.sum()), "target": eps, "tau_u": tau_u}
        for L in (100, 500):
            r[f"deff_L{L}"] = deff_block(m, L)
        rows_deff.append(r)
        D_measured[u] = np.nanmax([r["deff_L100"], r["deff_L500"]])
    for level, u, eps in units(fv, tv):              # CICIoT2023: shuffled rows, for comparison
        lab = fv if level == "family" else tv
        sel = lab == u
        tau_u = float(np.quantile(pv[sel], eps))
        m = (pv[sel] < tau_u).astype(float)
        rows_deff.append({"level": f"{level} (CICIoT2023, shuffled)", "unit": u, "n": int(sel.sum()), "target": eps,
                          "tau_u": tau_u, "deff_L100": deff_block(m, 100), "deff_L500": deff_block(m, 500)})

    # ---- 3. sensitivity of the certified threshold ----
    cal = pos < 0.2
    settings = [("ciciot2023: validation -> test", pv, fv, tv, pt, ft, tt),
                ("ciciomt2024: first 20% of test captures -> rest", p[cal], fam[cal], typ[cal], p[~cal], fam[~cal], typ[~cal])]
    for name, pc, fc, tc, pe, fe, te in settings:
        Ds = [1.0, 1.5, 2.0, 4.0, 8.0] + (["measured"] if "ciciomt" in name else [])
        for D in Ds:
            tau, ok = pick_D(pc, fc, tc, D_measured if D == "measured" else D)
            r, _, _ = evaluate(pe, fe, te, tau if ok else -1.0, name)
            r.update({"setting": name, "D": D, "feasible": ok, "tau": tau if ok else np.nan})
            rows_sens.append(r)
            print(name, D, ok, round(tau, 4), round(r["benign_bypass"], 3), r["worst_family"],
                  round(r["worst_family_miss"], 4), r["worst_type"], round(r["worst_type_miss"], 4), flush=True)

    # ---- 4. window coverage with the corrected rule (CICIoMT2024, as in risk_coverage.coverage_window) ----
    for D in (1.0, 2.0, "measured"):
        res = []
        for _ in range(100):
            start = RNG.uniform(0, 0.8, size=fid.max() + 1)[fid]
            c = (pos >= start) & (pos < start + 0.2)
            tau, ok = pick_D(p[c], fam[c], typ[c], D_measured if D == "measured" else D)
            fb, tb = violations(p[~c], fam[~c], typ[~c], tau, typ[c], fam[c]) if ok else (False, False)
            res.append((ok, fb or tb, (p[~c & (fam == "Benign")] < tau).mean() if ok else 0.0))
        r = np.array(res, float)
        rows_cov.append({"D": D, "reps": 100, "feasible": r[:, 0].mean(), "violation_any": r[:, 1].mean(),
                         "bypass_mean": r[:, 2].mean()})
        print("window coverage", rows_cov[-1], flush=True)

    pd.DataFrame(rows_order).to_csv(f"{OUT}/dependence_order.csv", index=False)
    pd.DataFrame(rows_deff).to_csv(f"{OUT}/dependence_deff.csv", index=False)
    pd.DataFrame(rows_sens).to_csv(f"{OUT}/dependence_sensitivity.csv", index=False)
    pd.DataFrame(rows_cov).to_csv(f"{OUT}/dependence_coverage.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows_order).to_string(index=False))
    print(pd.DataFrame(rows_deff).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
