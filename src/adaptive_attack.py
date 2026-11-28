"""Adaptive packet-padding attack on the deployed pMN model (worst-case over a padding grid).

Padding is realizable and monotone: a packet's length can only grow, up to the Ethernet maximum 1514.
For each candidate target size L, every attack packet is padded to max(len, L), features are re-extracted, and
the two-stage IDS scores the record. The attacker then keeps, per record, the L that best evades (lowest
Gatekeeper score). A record counts as evaded if ANY L makes the Gatekeeper skip it OR the Brain call it benign.
This upper-bounds uniform-padding evasion. Timing is not a lever here (pMN has no timing features).

Usage: .venv/bin/python src/adaptive_attack.py  -> results/packet/adaptive_attack.csv
"""
import os

import numpy as np
import pandas as pd

from datasets import get_dataset
from flow_records import PCAP_DIR, attack_mask, extract
from risk_added import scores

RUN, TAG = "runs/pkt_MN_sel", "pMN"
CAPS = ["Recon-OS_Scan", "Recon-Port_Scan", "Recon-VulScan", "Recon-Ping_Sweep", "MQTT-Malformed_Data",
        "ARP_Spoofing"]
GRID = [0, 200, 400, 600, 900, 1200, 1514]


def pad_to(L):
    if L == 0:
        return None

    def f(pk):
        return [(t, fwd, max(length, L), hdr, proto, ttl, flags, ports, ind)
                for (t, fwd, length, hdr, proto, ttl, flags, ports, ind) in pk]
    return f


def main():
    ds = get_dataset("ciciomt2024pmn")
    _, _, _, feats = ds.load("validation", 0.001)
    tau = float(pd.read_csv(f"results/packet/cert_{TAG}.csv").query("rule=='added'").tau.iloc[0])
    rows = []
    for cap in CAPS:
        base = extract(f"{PCAP_DIR}/test/{cap}_test.pcap", f"{cap}_test")
        keep = attack_mask(base, "mac")
        fam = base.family.iloc[0]
        evaded = None
        base_miss = None
        for L in GRID:
            d = extract(f"{PCAP_DIR}/test/{cap}_test.pcap", f"{cap}_test", pad_to(L))
            d = d[keep]
            p, flagged = scores(RUN, d[feats].to_numpy(np.float32), ds.classes)
            miss = (p < tau) | ~flagged
            evaded = miss if evaded is None else (evaded | miss)
            if L == 0:
                base_miss = float(miss.mean())
        rows.append({"capture": cap, "family": fam, "records": int(keep.sum()),
                     "miss_no_attack": round(base_miss, 3), "miss_adaptive_pad": round(float(evaded.mean()), 3)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/packet/adaptive_attack.csv", index=False)


if __name__ == "__main__":
    main()
