"""Cost results with the fast inference path (referee A7: the C API path inflates Stage 2's cost).

ONNX Runtime, one thread, one record per call (as in baselines.py), deployed CICIoT2023 models
(runs/seed42_sqrt). For eight benign shares, 5,000 test flows each (benign drawn without replacement,
attack families in proportion to their test share, at least 100 per family when possible), three
interleaved repeats: mean time per record of the single-stage IDS (Stage 2 on every record) and of
the two-stage IDS at the observed (0.05) and certified (0.0166) thresholds. Also the mean Stage 1 and
Stage 2 times, so Eq. (11) can predict the break-even share for this engine.

Usage: .venv/bin/python src/onnx_cost.py -> results/onnx_cost/{sweep,stages,breakeven}.csv
Run on an idle machine.
"""
import os
import time

import numpy as np
import pandas as pd

import baselines as B

OUT = "results/onnx_cost"
SHARES = [None, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.90]  # None = natural random sample
N, REPEATS, WARMUP = 5000, 3, 200
from risk_control import certified_tau
TAUS = {"observed": 0.05, "certified": certified_tau()}


def mix(g, share, rng):
    idx_all = np.arange(len(g))
    if share is None:
        return rng.choice(idx_all, N, replace=False)
    ben = np.flatnonzero(g == "Benign")
    att = np.flatnonzero(g != "Benign")
    n_ben = int(round(N * share))
    n_att = N - n_ben
    fams, cnt = np.unique(g[att], return_counts=True)
    floor = min(100, n_att // 20)
    alloc = np.maximum(np.round(cnt / cnt.sum() * n_att).astype(int), floor)
    alloc[np.argmax(cnt)] -= alloc.sum() - n_att
    parts = [rng.choice(ben, n_ben, replace=False)]
    parts += [rng.choice(att[g[att] == f], k, replace=False) for f, k in zip(fams, alloc)]
    return rng.permutation(np.concatenate(parts))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    X, y, g = B.scaled("test")
    s1 = B.onnx_session(open(f"{B.MODELS}/xgb_stage1_proposed.onnx", "rb").read())
    s2 = B.onnx_session(open(f"{B.MODELS}/xgb_stage2.onnx", "rb").read())
    n1, n2 = s1.get_inputs()[0].name, s2.get_inputs()[0].name
    for i in range(500):  # warm-up
        s1.run(None, {n1: X[i:i + 1]})
        s2.run(None, {n2: X[i:i + 1]})
    rows, t1_all, t2_all = [], [], []
    for share in SHARES:
        idx = mix(g, share, rng)
        Xm, gm = X[idx], g[idx]
        designs = ["single"] + list(TAUS)
        res = {d: [] for d in designs}
        for rep in range(REPEATS):
            for d in (designs if rep % 2 == 0 else designs[::-1]):
                ts = []
                for i in range(len(Xm)):
                    row = Xm[i:i + 1]
                    t0 = time.perf_counter_ns()
                    if d == "single":
                        s2.run(None, {n2: row})
                    else:
                        a = time.perf_counter_ns()
                        flag = np.asarray(s1.run(None, {n1: row})[1])[0, 1] >= TAUS[d]
                        b = time.perf_counter_ns()
                        if flag:
                            s2.run(None, {n2: row})
                            if d == "observed":
                                t2_all.append(time.perf_counter_ns() - b)
                        if d == "observed":
                            t1_all.append(b - a)
                    ts.append(time.perf_counter_ns() - t0)
                res[d].append(np.mean(ts[WARMUP:]) / 1e3)
        ben = gm == "Benign"
        p = np.asarray(s1.run(None, {n1: Xm})[1])[:, 1]
        for d in designs:
            rows.append({"benign_share": float(ben.mean()), "design": d, "mean_us": float(np.mean(res[d])),
                         "min_us": float(np.min(res[d])), "max_us": float(np.max(res[d])),
                         "benign_bypass": float((p[ben] < TAUS[d]).mean()) if d != "single" else 0.0})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/sweep.csv", index=False)
    sw = pd.DataFrame(rows)
    single = sw[sw.design == "single"].set_index("benign_share").mean_us
    sw["saving"] = 1 - sw.mean_us / sw.benign_share.map(single)
    sw.to_csv(f"{OUT}/sweep.csv", index=False)
    T1, T2 = float(np.mean(t1_all) / 1e3), float(np.mean(t2_all) / 1e3)
    pd.DataFrame([{"T1_us": T1, "T2_us": T2, "T1_over_T2": T1 / T2}]).to_csv(f"{OUT}/stages.csv", index=False)
    be = []
    for d in TAUS:
        q = sw[sw.design == d].sort_values("benign_share")
        B_ = float(q[q.benign_share >= 0.5].benign_bypass.mean())
        cross = None
        for (_, a), (_, b) in zip(q.iterrows(), q.iloc[1:].iterrows()):
            if a.saving <= 0 < b.saving:
                cross = a.benign_share + (b.benign_share - a.benign_share) * (-a.saving) / (b.saving - a.saving)
        be.append({"design": d, "B": B_, "predicted_pi": T1 / (B_ * T2), "measured_pi_interp": cross})
    pd.DataFrame(be).to_csv(f"{OUT}/breakeven.csv", index=False)
    print(pd.DataFrame(be), f"T1 {T1:.2f} us, T2 {T2:.2f} us")


if __name__ == "__main__":
    main()
