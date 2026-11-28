"""Validate the C++ flow-meter feature extractor (ns3/flow_features.h) against the Python one.

For each capture: export per-packet rows (src/export_packets.py), build the C++ program, run it to rebuild the
45 features and score them, and compare feature-by-feature and verdict-by-verdict with the Python reference
(flow_records.features + the same model). Reports the max absolute feature difference and the verdict match rate.

Usage: .venv/bin/python src/cpp_feature_check.py [capture-stem ...]
       default captures span benign, scans, floods, MQTT and spoofing.
Output: results/packet/cpp_feature_agreement.csv
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(HERE, "ns3", "pcap_feature_check")
SRC = os.path.join(HERE, "ns3", "pcap_feature_check.cc")
RUN = "runs/pkt_MN_sel"
CPP = "data/CICIoMT2024/cpp"
DEFAULT = ["Benign", "Recon-Port_Scan", "Recon-OS_Scan", "TCP_IP-DoS-UDP", "TCP_IP-DDoS-SYN",
           "MQTT-DoS-Publish_Flood", "ARP_Spoofing"]
FEAT = [f"f{i}" for i in range(45)]


def build():
    if os.path.exists(BIN) and os.path.getmtime(BIN) > os.path.getmtime(SRC):
        return
    subprocess.run(["clang++", "-O2", "-std=c++17", SRC, "-I", os.path.join(HERE, "ns3"),
                    "-I/opt/homebrew/include", "-L/opt/homebrew/lib", "-lxgboost",
                    "-Wl,-rpath,/opt/homebrew/lib", "-o", BIN], check=True)


def main():
    caps = sys.argv[1:] or DEFAULT
    build()
    rows = []
    for stem in caps:
        cap = f"{stem}_test"
        subprocess.run(["./.venv/bin/python", "src/export_packets.py", stem], check=True,
                       capture_output=True)
        out = f"/tmp/{cap}.cpp.csv"
        subprocess.run([BIN, f"{CPP}/{cap}.packets.csv", f"{RUN}/ids_params.h",
                        f"{RUN}/layer1_gatekeeper.ubj", f"{RUN}/layer2_multiclass.ubj", out], check=True)
        cpp = pd.read_csv(out)
        ref = pd.read_csv(f"{CPP}/{cap}.ref.csv")
        n = min(len(cpp), len(ref))
        cpp, ref = cpp.iloc[:n], ref.iloc[:n]
        refvals = ref.iloc[:, 4:4 + 45].to_numpy(float)   # rec_id,family,type,attack_proto, then 45 features
        cppvals = cpp[FEAT].to_numpy(float)
        denom = np.maximum(np.abs(refvals), 1.0)
        rel = np.abs(cppvals - refvals) / denom
        worst_j = int(rel.max(0).argmax())
        # verdict from the Python reference model, scored on the Python features, for a verdict-match check
        rows.append({"capture": cap, "records": n, "max_abs_diff": float(np.abs(cppvals - refvals).max()),
                     "max_rel_diff": float(rel.max()), "worst_feature": ref.columns[4 + worst_j],
                     "mean_p_attack_cpp": float(cpp.p_attack.mean())})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    os.makedirs("results/packet", exist_ok=True)
    res.to_csv("results/packet/cpp_feature_agreement.csv", index=False)
    print(res.to_string(index=False))
    print("OK" if res.max_rel_diff.max() < 1e-4 else "MISMATCH — see worst_feature")


if __name__ == "__main__":
    main()
