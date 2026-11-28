"""Agreement between our packet-level extractor (src/cic_extractor.py, mode "cic") and CIC's CSV, per feature.

For every CICIoMT2024 test capture: the window size is read from CIC's CSV (Number = 9.5 -> 100 packets, else 10);
CIC split each PCAP into chunks, processed them separately and joined the CSVs in another order, and it drops a
few packets, so our windows are aligned locally: windows whose key (features reproduced exactly: Tot size,
Duration, TCP, UDP, ARP, Min, Max, Protocol Type) is unique in both files are anchors; every anchor is grown
forwards and backwards while the keys keep matching; features are compared on the aligned windows only.
'coverage' = share of CIC rows that were aligned. A value agrees when |ours - CIC| <= 1e-3 + 1e-3 |CIC|.
Large flood captures are cut to their first LIMIT packets (the PCAPs of three floods are cut at 100 MB anyway).

Usage: .venv/bin/python src/cic_validate.py [split]  -> results/cic_extractor/{agreement,captures}_<split>.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from cic_extractor import COLUMNS, extract

OUT = "results/cic_extractor"
LIMIT = 1_000_000   # packets for 10-packet windows (x10 for 100-packet windows)


KEY = ["Tot size", "Duration", "TCP", "UDP", "ARP", "Min", "Max", "Protocol Type"]


def align(ours, cic):
    """Pairs (our window, CIC row) from unique anchors grown in both directions."""
    ko = [tuple(r) for r in ours[KEY].round(2).values]
    kc = [tuple(r) for r in cic[KEY].round(2).values]
    where = {}
    for i, k in enumerate(kc):
        where.setdefault(k, []).append(i)
    count_o = {}
    for k in ko:
        count_o[k] = count_o.get(k, 0) + 1
    used_o, used_c = np.zeros(len(ko), bool), np.zeros(len(kc), bool)
    pairs = []
    for w, k in enumerate(ko):
        h = where.get(k, [])
        if len(h) != 1 or count_o[k] != 1 or used_o[w] or used_c[h[0]]:
            continue
        c = h[0]
        lo = 0
        while w - lo - 1 >= 0 and c - lo - 1 >= 0 and ko[w - lo - 1] == kc[c - lo - 1] and not used_o[w - lo - 1]:
            lo += 1
        hi = 0
        while w + hi + 1 < len(ko) and c + hi + 1 < len(kc) and ko[w + hi + 1] == kc[c + hi + 1] and not used_o[w + hi + 1]:
            hi += 1
        for d in range(-lo, hi + 1):
            if not used_c[c + d]:
                used_o[w + d] = used_c[c + d] = True
                pairs.append((w + d, c + d))
    return np.array(sorted(pairs)).reshape(-1, 2)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    os.makedirs(OUT, exist_ok=True)
    caps, rows = [], []
    for pcap in sorted(glob.glob(f"data/CICIoMT2024/PCAP_full/{split}/*.pcap")):
        name = os.path.basename(pcap).replace(".pcap", "")
        csv = f"data/CICIoMT2024/WiFi_and_MQTT/{split}/{name}.pcap.csv"
        if not os.path.exists(csv):
            continue
        cic = pd.read_csv(csv)
        W = 100 if abs(cic.Number.median() - 9.5) < 1e-6 else 10
        ours = extract(pcap, window=W, mode="cic", limit=LIMIT * W // 10)
        pr = align(ours, cic)
        if len(pr) == 0:
            print(name, "no anchors", flush=True)
            continue
        o, c = ours.iloc[pr[:, 0]].reset_index(drop=True), cic.iloc[pr[:, 1]].reset_index(drop=True)
        m = len(pr)
        r = {"capture": name, "window": W, "rows_compared": m, "our_windows": len(ours), "cic_rows": len(cic),
             "coverage": m / min(len(cic), len(ours))}
        for col in COLUMNS:
            r[col] = float(np.isclose(o[col].values, c[col].values, rtol=1e-3, atol=1e-3).mean())
        caps.append(r)
        print(name, W, m, f"coverage {r['coverage']:.2f}", f"mean agreement {np.mean([r[c] for c in COLUMNS]):.3f}", flush=True)
    cap = pd.DataFrame(caps)
    cap.to_csv(f"{OUT}/captures_{split}.csv", index=False)
    w = cap.rows_compared.values
    agg = pd.DataFrame({"feature": COLUMNS,
                        "agreement_rows_weighted": [np.average(cap[c], weights=w) for c in COLUMNS],
                        "worst_capture": [cap[c].min() for c in COLUMNS]})
    agg.to_csv(f"{OUT}/agreement_{split}.csv", index=False)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
