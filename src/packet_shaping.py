"""Packet-level traffic shaping against the IDS (realisable attacker changes, then features extracted again).

An attacker who wants to look benign can only (a) send later - every gap between its packets can grow, never
shrink - and (b) pad its packets - every length can grow up to the Ethernet maximum 1514 bytes, never shrink.
Protocol, flags and ports stay (the attack needs them). Donor values come from benign TRAIN traffic:
  timing  every gap g -> max(g, d), d a random within-flow gap of benign traffic
  sizes   every length L -> min(1514, max(L, l)), l a random benign packet length of the same protocol
  both    timing and sizes
The change is applied inside each record of 10 packets (record boundaries are kept), the test captures of the
low-volume families are extracted again with src/flow_records.extract(transform=...), and the records are scored
with the trained runs at their certified thresholds: Gatekeeper skip rate and two-stage miss rate per family.
'none' runs the same path without a change (must equal the normal test records).

Usage: .venv/bin/python src/packet_shaping.py <dataset>:<run>:<tag> ...  -> results/packet/shaping.csv
"""
import glob
import os
import sys

import dpkt
import numpy as np
import pandas as pd

from datasets import get_dataset
from flow_records import FEATURES, PCAP_DIR, attack_mask, extract, parse
from risk_added import scores

OUT = "results/packet"
CAPTURES = ["ARP_Spoofing", "MQTT-Malformed_Data", "Recon-OS_Scan", "Recon-Ping_Sweep", "Recon-Port_Scan",
            "Recon-VulScan"]
DONOR_PACKETS = int(os.environ.get("PS_DONORS", 1_000_000))
RNG = np.random.default_rng(7)


def donors():
    """Within-flow gaps and packet lengths per protocol of benign train traffic."""
    gaps, lens, last = [], {}, {}
    with open(f"{PCAP_DIR}/train/Benign_train.pcap", "rb") as fh:
        for i, (ts, buf) in enumerate(dpkt.pcap.Reader(fh)):
            if i >= DONOR_PACKETS:
                break
            try:
                p = parse(buf)
            except Exception:
                continue
            if p is None:
                continue
            key, length, ind = p[0], p[2], p[8]
            proto = next((k for k in ("TCP", "UDP", "ICMP", "ARP") if ind[k]), "other")
            lens.setdefault(proto, []).append(length)
            if key in last and ts - last[key] < 10.0:
                gaps.append(ts - last[key])
            last[key] = ts
    return np.array(gaps), {k: np.array(v) for k, v in lens.items()}


def make_transform(kind, gaps, lens):
    def f(pk):
        out, t_prev, t_new = [], None, None
        for x in pk:
            t, fwd, L, hdr, proto, ttl, flags, ports, ind = x
            if kind in ("timing", "both"):
                if t_prev is None:
                    t_new = t
                else:
                    t_new = t_new + max(t - t_prev, float(RNG.choice(gaps)))
                t_prev = t
            else:
                t_new = t
            if kind in ("sizes", "both"):
                pr = next((k for k in ("TCP", "UDP", "ICMP", "ARP") if ind[k]), "other")
                pool = lens.get(pr, lens["TCP"])
                L = int(min(1514, max(L, int(RNG.choice(pool)))))
            out.append((t_new, fwd, L, hdr, proto, ttl, flags, ports, ind))
        return out
    return None if kind == "none" else f


def main(specs):
    gaps, lens = donors()
    print(f"donors: {len(gaps):,} gaps (median {np.median(gaps):.4f} s), lengths per protocol "
          f"{ {k: len(v) for k, v in lens.items()} }", flush=True)
    recs = {}
    for kind in ("none", "timing", "sizes", "both"):
        parts = []
        for cap in CAPTURES:
            d = extract(f"{PCAP_DIR}/test/{cap}_test.pcap", f"{cap}_test", make_transform(kind, gaps, lens))
            parts.append(d[attack_mask(d, os.environ.get("PKT_LABEL", "proto"))])
        recs[kind] = pd.concat(parts, ignore_index=True)
        print(kind, len(recs[kind]), flush=True)
    rows = []
    for spec in specs:
        ds_name, run, tag = spec.split(":")
        ds = get_dataset(ds_name)
        _, _, _, feats = ds.load("validation", 0.001)
        cert = pd.read_csv(f"{OUT}/cert_{tag}.csv").set_index("rule")
        for kind, d in recs.items():
            p, flagged = scores(run, d[feats].to_numpy(np.float32), ds.classes)
            fam = d.family.str.replace("Spoofing", "Spoofing").to_numpy()
            for rule in ("skip", "added"):
                if not bool(cert.loc[rule, "feasible"]):
                    continue
                tau = float(cert.loc[rule, "tau"])
                for g in np.unique(fam):
                    m = fam == g
                    rows.append({"tag": tag, "rule": rule, "tau": tau, "shaping": kind, "family": g,
                                 "records": int(m.sum()), "gatekeeper_skip": float((p[m] < tau).mean()),
                                 "brain_miss": float((~flagged[m]).mean()),
                                 "ids_miss": float(((p[m] < tau) | ~flagged[m]).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/shaping{os.environ.get('PKT_SFX', '')}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.pivot_table(index=["tag", "rule", "family"], columns="shaping", values="ids_miss").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
