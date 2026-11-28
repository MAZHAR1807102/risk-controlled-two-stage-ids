"""Export the per-packet fields a gateway sees on the wire, for the C++ flow-meter feature extractor.

Reuses src/flow_records.py to parse a PCAP and assign each packet to the same 10-packet flow record. Writes:
  <out>.packets.csv : one row per packet, in capture order
        rec_id, fwd, len, hdr, proto, ttl, flags, sport, dport
  <out>.ref.csv     : the reference feature vector per record (flow_records.features, the pMN 45-feature subset),
                      with rec_id, family, type, attacker_share
The C++ side (ns3/flow_features.h) rebuilds the features from the packet rows; the two must match (validation in
src/cpp_feature_check.py). rec_id is the record's emission order, so C++ and Python group identically.

Usage: .venv/bin/python src/export_packets.py <capture-stem> [--split test]
       e.g. ... Recon-Port_Scan   ->  data/CICIoMT2024/cpp/<stem>.{packets,ref}.csv
"""
import argparse
import os

import dpkt
import numpy as np
import pandas as pd

from flow_records import IDLE, PCAP_DIR, W, attack_proto_of, features, parse


def family(capture):
    """Capture file name -> (family, attack type), same grouping as src/datasets.py."""
    name = capture.rsplit("_", 1)[0]
    if name == "Benign":
        return "Benign", "Benign"
    if name == "ARP_Spoofing":
        return "Spoofing", "ARP_Spoofing"
    if name.startswith("MQTT"):
        return "MQTT", name
    if name.startswith("Recon"):
        return "Recon", name
    base = name.rstrip("0123456789")
    return ("DDoS" if "DDoS" in base else "DoS"), base

OUT = "data/CICIoMT2024/cpp"
FEAT45 = ["Header_Length", "Protocol Type", "Duration", "fin_flag_number", "syn_flag_number", "rst_flag_number",
          "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number", "fin_count", "syn_count",
          "rst_count", "psh_count", "ack_count", "ece_count", "cwr_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
          "SSH", "IRC", "DHCP", "MQTT", "TCP", "UDP", "ICMP", "ARP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
          "AVG", "Std", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight", "fwd_share"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cap = f"{a.stem}_{a.split}"
    fam, typ = family(cap)
    want = attack_proto_of(typ) if fam != "Benign" else None

    open_, pk_rows, ref_rows, rid = {}, [], [], 0

    def emit(st):
        nonlocal rid
        pk = st["pk"]
        r = features(pk)
        uses = True if want is None else bool(r[want] > 0)
        for (ts, fwd, length, hdr, proto, ttl, flags, ports, ind) in pk:
            l4 = 6 if ind["TCP"] else 17 if ind["UDP"] else 1 if ind["ICMP"] else 2 if ind["IGMP"] else 0
            pk_rows.append((rid, int(fwd), length, hdr, proto, l4, ttl, flags, ports[0], ports[1]))
        ref_rows.append({"rec_id": rid, "family": fam, "type": typ, "attack_proto": uses,
                         **{k: r[k] for k in FEAT45}})
        rid += 1
        st["pk"] = []

    path = f"{PCAP_DIR}/{a.split}/{cap}.pcap"
    with open(path, "rb") as fh:
        try:
            for ts, buf in dpkt.pcap.Reader(fh):
                try:
                    p = parse(buf)
                except (dpkt.dpkt.UnpackError, ValueError, IndexError, AttributeError):
                    continue
                if p is None:
                    continue
                key, sender, length, hdr, proto, ttl, flags, ports, ind, hosts = p
                st = open_.get(key)
                if st is not None and ts - st["last"] > IDLE:
                    emit(st)
                    del open_[key]
                    st = None
                if st is None:
                    st = open_[key] = {"pk": [], "first": sender, "last": ts}
                st["last"] = ts
                st["pk"].append((ts, sender == st["first"], length, hdr, proto, ttl, flags, ports, ind))
                if len(st["pk"]) == W:
                    emit(st)
                    del open_[key]
        except (dpkt.dpkt.NeedData, ValueError):
            pass
    for st in list(open_.values()):
        if st["pk"]:
            emit(st)

    pk = pd.DataFrame(pk_rows, columns=["rec_id", "fwd", "len", "hdr", "proto", "l4", "ttl", "flags", "sport", "dport"])
    pk.to_csv(f"{OUT}/{cap}.packets.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(f"{OUT}/{cap}.ref.csv", index=False)
    print(f"{cap}: {len(pk)} packets, {rid} records -> {OUT}/{cap}.{{packets,ref}}.csv")


if __name__ == "__main__":
    main()
