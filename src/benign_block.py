"""Benign hosts blocked by the three-verdict rule (referee A13: false alarms are under-reported).

Packet-derived CICIoMT2024 test captures, main model pMN. Every benign record (no packet to or from
an attacker MAC) gets the two-stage verdict at a threshold (attack = Gatekeeper score >= tau and the
Brain says attack) or the single-stage verdict. Records are attributed to a source host and replayed
in time order inside each capture; a host is blocked once it has at least K = 3 verdicts and at least
80% of them say attack (the rule of the gateway simulation). The CSV datasets carry no host
identity, so this check is possible only on the packet-derived records.

Source host: host_a (the endpoint that sent the record's first packet); host_b as a sensitivity check.
Output: results/packet/benign_block.csv
"""
import numpy as np
import pandas as pd

from datasets import get_dataset
from risk_added import scores

RUN, DS = "runs/pkt_MN_sel", "ciciomt2024pmn"
THRESHOLDS = {"CP-selected skip": 1.2666629280555893e-04, "observed": 0.014992746373186596,
              "whole system": 0.29785117558779395}
K, RATIO = 3, 0.8


def replay(df):
    """df sorted by time with columns host, attack; returns per-host blocked flag and records after block."""
    out = []
    for h, g in df.groupby("host", sort=False):
        a = g.attack.to_numpy()
        n = np.arange(1, len(a) + 1)
        c = np.cumsum(a)
        hit = np.flatnonzero((n >= K) & (c >= RATIO * n))
        first = hit[0] if len(hit) else None
        out.append({"host": h, "records": len(a), "blocked": first is not None,
                    "records_after_block": 0 if first is None else len(a) - first - 1})
    return pd.DataFrame(out)


def main():
    ds = get_dataset(DS)
    X, fam, _, _ = ds.load("test")
    m = ds.meta("test")
    p, flagged = scores(RUN, X, ds.classes)
    ben = fam == "Benign"
    rows = []
    designs = {f"two-stage, {k}": (p >= t) & flagged for k, t in THRESHOLDS.items()}
    designs["single-stage"] = flagged
    for side in ("host_a", "host_b"):
        base = pd.DataFrame({"capture": m["capture"][ben], "host": m[side][ben], "t0": m["t0"][ben]})
        for name, att in designs.items():
            d = base.assign(attack=att[ben].astype(int)).sort_values(["capture", "t0"], kind="stable")
            d["host"] = d.capture + "|" + d.host
            r = replay(d)
            rows.append({"source": side, "design": name, "benign_records": int(ben.sum()),
                         "benign_records_flagged": float(att[ben].mean()), "hosts": len(r),
                         "hosts_blocked": int(r.blocked.sum()), "hosts_blocked_share": float(r.blocked.mean()),
                         "records_after_block_share": float(r.records_after_block.sum() / ben.sum())})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/packet/benign_block.csv", index=False)


if __name__ == "__main__":
    main()
