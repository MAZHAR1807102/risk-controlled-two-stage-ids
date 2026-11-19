"""What a Protocol B calibration window of CICIoMT2024 means in time and devices.

For every capture (CIC PCAP, Wi-Fi/MQTT part) whose PCAP we hold in full: packets, duration, and the
duration and distinct addresses of the first FRAC of its packets. The CSV rows of a capture are windows
of packets in capture order, so the first 20% of rows covers about the first 20% of packets.
Captures cut at 100 MB during download are marked truncated (flood captures; their durations are lower
bounds and they never limit the threshold). Devices: distinct source MAC and IP addresses; this counts
attacker hosts and any background hosts too, so it is an upper bound on the IoMT devices seen.

Usage: .venv/bin/python src/iomt_window_stats.py [test|train]  -> results/iomt_window/pcap_{split}.csv
"""
import glob
import os
import socket
import sys

import dpkt
import numpy as np
import pandas as pd

from datasets import CICIoMT2024

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CICIoMT2024")
OUT = "results/iomt_window"
FRACS = (0.05, 0.10, 0.20)
TRUNC = 104857600  # files cut at this size by the download script


def read(path):
    ts, smac, sip = [], [], []
    with open(path, "rb") as f:
        try:
            for t, buf in dpkt.pcap.Reader(f):
                ts.append(t)
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    smac.append(eth.src)
                    sip.append(socket.inet_ntoa(eth.data.src) if isinstance(eth.data, dpkt.ip.IP) else None)
                except Exception:
                    smac.append(None)
                    sip.append(None)
        except (dpkt.dpkt.NeedData, ValueError):
            pass  # last packet cut off
    return np.array(ts), np.array(smac, dtype=object), np.array(sip, dtype=object)


def main(split="test"):
    os.makedirs(OUT, exist_ok=True)
    ds = CICIoMT2024()
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "PCAP_full", split, "*.pcap"))):
        stem = os.path.basename(path).replace(".pcap", "")
        csv = os.path.join(ds.DIR, split, stem + ".pcap.csv")
        n_rows = sum(1 for _ in open(csv)) - 1 if os.path.exists(csv) else np.nan
        ts, smac, sip = read(path)
        n = len(ts)
        attack = ds.attack_name(csv)
        r = {"capture": stem, "family": ds.family(attack), "type": attack, "packets": n, "csv_rows": n_rows,
             "truncated": os.path.getsize(path) == TRUNC, "duration_s": float(ts[-1] - ts[0]),
             "macs": len({m for m in smac if m is not None}), "ips": len({i for i in sip if i is not None})}
        for f in FRACS:
            k = max(1, int(round(f * n)))
            r[f"dur_first{int(f*100)}_s"] = float(ts[k - 1] - ts[0])
            r[f"rows_first{int(f*100)}"] = int(round(f * n_rows)) if n_rows == n_rows else np.nan
            r[f"macs_first{int(f*100)}"] = len({m for m in smac[:k] if m is not None})
            r[f"ips_first{int(f*100)}"] = len({i for i in sip[:k] if i is not None})
        rows.append(r)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/pcap_{split}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
