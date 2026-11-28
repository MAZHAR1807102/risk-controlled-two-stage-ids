"""Run an already-trained two-stage IDS on the Kitsune ARP-MitM capture (cross-dataset generalization test).

Kitsune (Mirsky et al., NDSS 2018) ships, per attack, a raw capture and an N-by-1 per-packet label file
(1 = malicious, 0 = benign), in capture order. That is exact ground truth, so no MAC guessing is needed.
We build the same 10-packet flow records as src/flow_records.py, label each record by the fraction of its
packets that Kitsune marks malicious (a record counts as attack if any of its packets is malicious), compute
the features the chosen trained run expects, and score with that run's Gatekeeper and Brain. No retraining:
this asks whether a model trained on CICIoMT2024 catches ARP spoofing from a different testbed, tool and device.

Usage:
  .venv/bin/python src/kitsune_eval.py <pcap(.pcapng)> <labels.csv> <run> [--attack Spoofing]
     -> results/packet/kitsune_<run-basename>.csv
"""
import argparse
import os

import dpkt
import numpy as np
import pandas as pd
import xgboost as xgb

from flow_records import W, IDLE, features, parse
from risk_control import ALPHA, cp_upper
from train_gatekeeper import scaler_from_header


def feature_order(run):
    """The feature names the run was trained on (from the '// Feature order:' line of ids_params.h)."""
    for line in open(f"{run}/ids_params.h"):
        if "Feature order:" in line:
            return [f.strip() for f in line.split("Feature order:")[1].split(",")]
    raise ValueError("no feature-order comment in ids_params.h")


def reader(path):
    f = open(path, "rb")
    head = f.read(4)
    f.seek(0)
    if head == b"\x0a\x0d\x0d\x0a":          # pcapng magic
        return dpkt.pcapng.Reader(f)
    return dpkt.pcap.Reader(f)


def records(pcap, labels):
    """Yield (feature dict, malicious_fraction) for every 10-packet flow record, in the order they close."""
    open_, out = {}, []
    lab = labels
    for i, (ts, buf) in enumerate(reader(pcap)):
        try:
            p = parse(buf)
        except (dpkt.dpkt.UnpackError, ValueError, IndexError, AttributeError):
            continue
        if p is None:
            continue
        key, sender, length, hdr, proto, ttl, flags, ports, ind, hosts = p
        y = lab[i] if i < len(lab) else 0
        st = open_.get(key)
        if st is not None and ts - st["last"] > IDLE:
            out.append(close(st))
            st = None
        if st is None:
            st = open_[key] = {"pk": [], "first": sender, "last": ts, "y": []}
        st["last"] = ts
        st["pk"].append((ts, sender == st["first"], length, hdr, proto, ttl, flags, ports, ind))
        st["y"].append(y)
        if len(st["pk"]) == W:
            out.append(close(st))
            del open_[key]
    for st in open_.values():
        out.append(close(st))
    return out


def close(st):
    r = features(st["pk"])
    st["pk"], frac = [], float(np.mean(st["y"]))
    st["y"] = []
    return r, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("labels")
    ap.add_argument("run")
    ap.add_argument("--attack", default="Spoofing")
    ap.add_argument("--classes", default="Benign,DDoS,DoS,MQTT,Recon,Spoofing")
    ap.add_argument("--tau", type=float, default=None, help="Gatekeeper threshold (default: tau in ids_params.h)")
    ap.add_argument("--tag", default="", help="suffix of the output file")
    a = ap.parse_args()
    classes = a.classes.split(",")

    # Kitsune labels are an R-style CSV: a header row and a leading index column; the last column is 0/1.
    raw = pd.read_csv(a.labels)
    lab = pd.to_numeric(raw.iloc[:, -1], errors="coerce").fillna(0).to_numpy(float)
    lab = (lab > 0).astype(int)
    print(f"labels: {len(lab):,} packets, {lab.mean():.3f} malicious", flush=True)
    recs = records(a.pcap, lab)
    feats = [r for r, _ in recs]
    frac = np.array([f for _, f in recs])
    df = pd.DataFrame(feats)
    print(f"records: {len(df):,} ({(frac > 0).mean():.3f} contain an attack packet)", flush=True)

    order = feature_order(a.run)
    for c in order:
        if c not in df:
            df[c] = 0.0
    X = df[order].to_numpy(np.float32)
    s = scaler_from_header(f"{a.run}/ids_params.h")
    d = xgb.DMatrix((X - np.float32(s["means"])) / np.float32(s["scales"]))
    gk, br = xgb.Booster(), xgb.Booster()
    gk.load_model(f"{a.run}/layer1_gatekeeper.ubj")
    br.load_model(f"{a.run}/layer2_multiclass.ubj")
    tau = a.tau if a.tau is not None else float(open(f"{a.run}/ids_params.h").read().split("GATEKEEPER_TAU =")[1].split("f")[0])
    p = gk.predict(d)
    pred = br.predict(d).argmax(1)
    benign_cls = classes.index("Benign")
    att = frac > 0

    rows = []
    for name, m in (("attack (malicious packet)", att), ("benign (clean record)", ~att)):
        if m.sum() == 0:
            continue
        skip = p[m] < tau
        flagged = pred[m] != benign_cls
        as_attack = pred[m] == classes.index(a.attack)
        k = int((skip | ~flagged).sum())
        rows.append({"group": name, "records": int(m.sum()),
                     "gatekeeper_skip": float(skip.mean()),
                     "brain_flags_attack": float(flagged.mean()),
                     f"brain_predicts_{a.attack}": float(as_attack.mean()),
                     "two_stage_calls_benign": float((skip | ~flagged).mean()),
                     "miss_ci95_high": float(cp_upper(k, int(m.sum()), ALPHA)) if "attack" in name else np.nan})
    res = pd.DataFrame(rows)
    cap = os.path.basename(a.pcap).split("_pcap")[0].split(".pcap")[0]
    out = f"results/packet/kitsune_{cap}_{os.path.basename(a.run)}{a.tag}.csv"
    os.makedirs("results/packet", exist_ok=True)
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print(f"tau={tau:.4f}  ->  {out}")


if __name__ == "__main__":
    main()
