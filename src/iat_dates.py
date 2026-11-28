"""What the IAT column of the CIC CSVs really holds.

CIC averages per-packet values over a window of packets. For the first packet of a flow, the inter-arrival
time is taken against 0, so it equals the packet's Unix timestamp. A window that holds such a packet gets an
IAT of about timestamp / window size (10 packets; 20 when the mean in-flow index 'Number' is 9.5).
This script undoes the division and turns the value back into a date:
  1. CICIoMT2024 ping-sweep test capture: the recovered times must fall inside the PCAP's own time range;
  2. CICIoT2023 merged test split: recovered date per attack type (each type was recorded on its own days).
Floods use larger windows, so their min/max dates are off; the medians are what matters.

Usage: .venv/bin/python src/iat_dates.py  -> results/iat/{pcap_check,capture_dates}.csv
"""
import os

import dpkt
import numpy as np
import pandas as pd

OUT = "results/iat"
PCAP = "data/CICIoMT2024/PCAP_full/test/Recon-Ping_Sweep_test.pcap"
CSV = "data/CICIoMT2024/WiFi_and_MQTT/test/Recon-Ping_Sweep_test.pcap.csv"


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(PCAP, "rb") as f:
        ts = np.array([t for t, _ in dpkt.pcap.Reader(f)])
    d = pd.read_csv(CSV)
    big = d.IAT[d.IAT > 1e6] * 10
    check = pd.DataFrame([{"capture": os.path.basename(PCAP), "packets": len(ts), "csv_rows": len(d),
                           "pcap_first": ts[0], "pcap_last": ts[-1], "rows_with_big_iat": len(big),
                           "iat_x10_min": big.min(), "iat_x10_max": big.max(),
                           "inside_capture": bool(big.min() >= ts[0] - 1 and big.max() <= ts[-1] + 1)}])
    check.to_csv(f"{OUT}/pcap_check.csv", index=False)
    print(check.T.to_string())

    t = pd.read_csv("data/CICIoT2023/test/test.csv", usecols=["IAT", "Number", "label"])
    share = (t.IAT > 1e6).groupby(t.label).mean()
    t = t[t.IAT > 1e6]
    t["ts"] = np.where(t.Number.round(1) == 9.5, t.IAT * 20, t.IAT * 10)
    g = t.groupby("label").ts.agg(["median", "count"]).rename(columns={"count": "rows_with_big_iat"})
    g["share_of_rows"] = share
    g["date"] = pd.to_datetime(g["median"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    g = g.sort_values("median")
    g.to_csv(f"{OUT}/capture_dates.csv")
    print(g.to_string())


if __name__ == "__main__":
    main()
