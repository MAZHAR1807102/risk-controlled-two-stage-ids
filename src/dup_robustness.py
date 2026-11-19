"""Does (near-)duplication across splits drive the certified threshold? (CICIoT2023)

Near-duplicate = same vector after rounding every raw feature to 3 significant digits
(exact copies are a subset). We then
  1. remove from validation every row that has a near copy in test, and keep one row per
     near-duplicate group inside validation (so calibration rows are distinct);
  2. keep only "novel" test rows: no near copy in train or validation;
  3. re-pick the certified threshold (skip and added-miss certificates) on the cleaned
     validation rows and evaluate it on the novel test rows.
Writes results/dependence/near_duplicates.csv and dedup_certificate.csv.

Usage: .venv/bin/python src/dup_robustness.py [significant digits, default 3]
"""
import os

import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import added_score, evaluate, scores
from risk_control import RUNS, pick

OUT = "results/dependence"


def near_hash(X, digits=3):
    a = np.abs(X.astype(np.float64))
    mag = np.where(a > 0, np.floor(np.log10(np.where(a > 0, a, 1))), 0)
    R = np.round(X / 10 ** mag, digits - 1) * 10 ** mag
    R = np.ascontiguousarray(R.astype(np.float32))
    v = R.view(np.dtype((np.void, R.dtype.itemsize * R.shape[1]))).ravel()
    return pd.Series(v).map(hash).to_numpy()


def main(digits=3, sel=False):
    os.makedirs(OUT, exist_ok=True)
    sfx = "" if digits == 3 else f"_{digits}sig"
    ds = get_dataset("ciciot2023")
    D = {}
    for s in ("train", "validation", "test"):
        X, fam, typ, _ = ds.load(s)
        D[s] = dict(X=X, fam=fam, typ=typ, h=near_hash(X, digits))
        print("hashed", s, flush=True)
    tr, va, te = (set(D[s]["h"]) for s in ("train", "validation", "test"))

    rows = []
    for s, others in (("validation", ("train", "test")), ("test", ("train", "validation"))):
        d = D[s]
        within = pd.Series(d["h"]).duplicated(keep=False).to_numpy()
        cross = {o: np.isin(d["h"], list({"train": tr, "validation": va, "test": te}[o])) for o in others}
        for g in ["ALL"] + sorted(np.unique(d["fam"])):
            m = np.ones(len(d["h"]), bool) if g == "ALL" else d["fam"] == g
            r = {"split": s, "family": g, "n": int(m.sum()), "near_dup_within": float(within[m].mean())}
            r.update({f"near_in_{o}": float(cross[o][m].mean()) for o in others})
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/near_duplicates{sfx}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --sel: use the Brain early-stopped on 15% of train (it never saw validation); the Gatekeeper is the same
    run = "runs/seed42_sel" if sel else RUNS["ciciot2023"]
    v, t = D["validation"], D["test"]
    pv, flv = scores(run, v["X"], ds.classes)
    pt, flt = scores(run, t["X"], ds.classes)
    keep_v = ~np.isin(v["h"], list(te)) & ~pd.Series(v["h"]).duplicated(keep="first").to_numpy()
    keep_t = ~np.isin(t["h"], list(tr | va))
    print(f"validation kept {keep_v.sum()}/{len(keep_v)}, novel test {keep_t.sum()}/{len(keep_t)}")

    out = []
    for setting, cv, ct in (("original", np.ones_like(keep_v), np.ones_like(keep_t)),
                            ("deduplicated", keep_v, keep_t)):
        for target in ("skip", "added"):
            q = pv[cv] if target == "skip" else added_score(pv[cv], v["fam"][cv], flv[cv])
            tau, ok = pick(q, v["fam"][cv], v["typ"][cv], "ucb_typ")
            r = evaluate(pt[ct], t["fam"][ct], t["typ"][ct], flt[ct], tau)
            r.update({"setting": setting, "certified_quantity": target, "feasible": ok,
                      "n_cal": int(cv.sum()), "n_test": int(ct.sum())})
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(f"{OUT}/dedup_certificate{sfx}{'_sel' if sel else ''}.csv", index=False)
    print(res[["setting", "certified_quantity", "feasible", "n_cal", "n_test", "tau", "benign_bypass",
               "max_family_skip", "worst_family", "worst_family_added", "worst_type", "worst_type_skip",
               "worst_type_added"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--sel"]
    main(int(args[0]) if args else 3, sel="--sel" in sys.argv)
