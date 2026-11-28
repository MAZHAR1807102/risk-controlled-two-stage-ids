"""Clean flow records from packet captures (the features used by the rebuilt paper).

A record holds up to W = 10 consecutive packets of one flow; the same W for every kind of traffic (CIC used 100
packets for floods, which leaks the label through Number/Weight). A flow is bidirectional: (protocol, the two IP
endpoints and ports), or (ARP, sender MAC, target IP). A record is closed after W packets, after IDLE seconds
without a packet of its flow, or at the end of the capture. This is what a gateway flow meter can compute.

Features keep CIC's names where the meaning is the same, but every value is computed from the record's own
packets only: no capture position, no absolute time, no state carried over from other flows.
  Header_Length  mean IP + transport header bytes      Protocol Type  IP protocol number (0 = ARP/other)
  Duration       mean TTL (CIC's name)                 Rate / Srate / Drate   packets/s, forward, backward
  *_flag_number  share of packets with the TCP flag    *_count        packets with the flag in the record
  HTTP ... MQTT  server-port indicators                TCP ... LLC    protocol indicators
  Tot sum, Min, Max, AVG, Std   packet lengths (14 + IP length; frame length for non-IP)
  IAT, IAT_std, IAT_min, IAT_max  real gaps between the record's packets (0 for one packet)
  Number         packets in the record                 Span           time from first to last packet
  Magnitue, Radius, Covariance, Variance, Weight  forward/backward length statistics (CICIoT2023 definitions)
  fwd_share      share of packets sent by the flow's first sender
Gateway context (added 2026-11-21; counted over packets, not seconds, so they do not depend on a recording's pace):
  arp_reply_share, arp_grat_share   ARP replies / gratuitous ARP (sender IP = target IP) in the record
  arp_ip_macs    most MACs seen claiming one sender IP of the record so far (as arpwatch), capped at 8
  arp_mac_ips    most sender IPs claimed by the record's MAC so far, capped at 64
  src_peers, src_ports   distinct destinations / destination ports in the last 50 packets the first sender sent
  dst_peers, dst_ports   distinct sources / source ports in the last 50 packets the other side received
Ground truth (not a feature): attacker_share = share of the record's packets sent to or from one of the four attacker
devices (Raspberry Pi MACs, found in the PCAPs: they carry 98-100% of the flood/scan packets and ~0 of the benign
captures). The dataset builder labels a record of an attack capture as attack when attacker_share > 0.
Metadata (not features): capture, family, type, flow, rec (index in the flow), t0, t1, src/dst host, attack_proto
(whether the record uses the protocol of its capture's attack; background traffic in attack captures is dropped
by the dataset builder).

Usage: .venv/bin/python src/flow_records.py [train|test] [--only NAME]
       -> data/CICIoMT2024/records_ctx/<split>/<capture>.parquet  (records/ = the version without context)
"""
import argparse
import glob
import os
import socket
from collections import deque

import dpkt
import numpy as np
import pandas as pd


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


W = 10
IDLE = 10.0
PCAP_DIR = "data/CICIoMT2024/PCAP_full"
OUT_DIR = "data/CICIoMT2024/records_ctx"
HOST_WINDOW = 50
ATTACKERS = {bytes.fromhex(m) for m in ("dca632dc27d5", "dca632c9e5a4", "dca632c9e4ab", "dca632c9e4d5")}
FLAGS = [("fin", 0x01), ("syn", 0x02), ("rst", 0x04), ("psh", 0x08), ("ack", 0x10), ("ece", 0x40), ("cwr", 0x80)]
PORTS = {"HTTP": {80, 8080}, "HTTPS": {443}, "DNS": {53}, "Telnet": {23}, "SMTP": {25}, "SSH": {22},
         "IRC": {194, 6667}, "DHCP": {67, 68}, "MQTT": {1883, 8883}}
PROTO_IND = ["TCP", "UDP", "ICMP", "ARP", "IGMP", "IPv", "LLC"]
FEATURES = (["Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate"]
            + [f"{f}_flag_number" for f, _ in FLAGS] + [f"{f}_count" for f, _ in FLAGS]
            + list(PORTS) + PROTO_IND
            + ["Tot sum", "Min", "Max", "AVG", "Std", "IAT", "IAT_std", "IAT_min", "IAT_max", "Number", "Span",
               "Magnitue", "Radius", "Covariance", "Variance", "Weight", "fwd_share"])
CONTEXT = ["arp_reply_share", "arp_grat_share", "arp_ip_macs", "arp_mac_ips", "src_peers", "src_ports", "dst_peers",
           "dst_ports"]
TIMING = ["Rate", "Srate", "Drate", "IAT", "IAT_std", "IAT_min", "IAT_max", "Span"]
FEATURES = FEATURES + CONTEXT
META = ["capture", "family", "type", "flow", "rec", "t0", "t1", "host_a", "host_b", "attack_proto", "attacker_share"]


def attack_proto_of(attack_type):
    """Which packets an attack uses (None = all, e.g. scans)."""
    if attack_type == "ARP_Spoofing":
        return "ARP"
    if attack_type.startswith("MQTT"):
        return "MQTT"
    name = attack_type.replace("TCP_IP-", "")      # the prefix would match "TCP" for every flood
    for key, col in (("ICMP", "ICMP"), ("SYN", "TCP"), ("TCP", "TCP"), ("UDP", "UDP")):
        if key in name:
            return col
    return None


def parse(buf):
    """Frame -> (key, sender, length, header, proto number, ttl, flags, ports, indicators, hosts) or None."""
    eth = dpkt.ethernet.Ethernet(buf)
    p = eth.data
    ind = dict.fromkeys(PROTO_IND, 0)
    if isinstance(p, dpkt.arp.ARP):
        ind["ARP"] = 1
        smac = eth.src.hex()
        tpa = socket.inet_ntoa(p.tpa) if len(p.tpa) == 4 else ""
        return ("arp", smac, tpa), smac, len(buf), 28, 0, 0, 0, (0, 0), ind, (smac, tpa)
    if not isinstance(p, dpkt.ip.IP):
        return None
    ind["IPv"] = ind["LLC"] = 1
    s, d = socket.inet_ntoa(p.src), socket.inet_ntoa(p.dst)
    l4, hdr, flags, sp, dp = p.data, p.hl * 4, 0, 0, 0
    if isinstance(l4, dpkt.tcp.TCP):
        ind["TCP"] = 1
        sp, dp, flags, hdr = l4.sport, l4.dport, l4.flags, hdr + l4.off * 4
    elif isinstance(l4, dpkt.udp.UDP):
        ind["UDP"] = 1
        sp, dp, hdr = l4.sport, l4.dport, hdr + 8
    elif isinstance(l4, dpkt.icmp.ICMP):
        ind["ICMP"] = 1
    elif p.p == 2:
        ind["IGMP"] = 1
    a, b = (s, sp), (d, dp)
    key = (p.p,) + ((a, b) if a <= b else (b, a))
    return key, s, p.len + 14, hdr, p.p, p.ttl, flags, (sp, dp), ind, (s, d)


def arp_fields(buf):
    """(op, sender IP, sender MAC, target IP) of an ARP frame, else None."""
    eth = dpkt.ethernet.Ethernet(buf)
    a = eth.data
    if not isinstance(a, dpkt.arp.ARP):
        return None
    return a.op, bytes(a.spa), bytes(a.sha), bytes(a.tpa)


def features(pk):
    """pk: list of per-packet tuples (t, fwd, length, header, proto, ttl, flags, ports, indicators)."""
    n = len(pk)
    t = np.array([x[0] for x in pk])
    fwd = np.array([x[1] for x in pk], bool)
    L = np.array([x[2] for x in pk], float)
    span = t[-1] - t[0]
    gaps = np.diff(t) if n > 1 else np.zeros(1)
    r = {"Header_Length": np.mean([x[3] for x in pk]), "Protocol Type": pk[0][4],
         "Duration": np.mean([x[5] for x in pk]),
         "Rate": n / span if span > 0 else 0.0, "Srate": fwd.sum() / span if span > 0 else 0.0,
         "Drate": (~fwd).sum() / span if span > 0 else 0.0}
    fl = np.array([x[6] for x in pk])
    for name, bit in FLAGS:
        c = int(((fl & bit) > 0).sum())
        r[f"{name}_flag_number"], r[f"{name}_count"] = c / n, c
    sp, dp = pk[0][7]
    for name, ports in PORTS.items():
        r[name] = float(sp in ports or dp in ports)
    for name in PROTO_IND:
        r[name] = np.mean([x[8][name] for x in pk])
    r.update({"Tot sum": L.sum(), "Min": L.min(), "Max": L.max(), "AVG": L.mean(), "Std": L.std(),
              "IAT": gaps.mean(), "IAT_std": gaps.std(), "IAT_min": gaps.min(), "IAT_max": gaps.max(),
              "Number": n, "Span": span, "fwd_share": fwd.mean()})
    li, lo = L[~fwd], L[fwd]          # incoming = sent by the other side, outgoing = sent by the first sender
    mi, mo = (li.mean() if len(li) else 0.0), (lo.mean() if len(lo) else 0.0)
    vi, vo = (li.var() if len(li) else 0.0), (lo.var() if len(lo) else 0.0)
    k = min(len(li), len(lo))
    r.update({"Magnitue": np.sqrt(mi + mo), "Radius": np.sqrt(vi + vo),
              "Covariance": float(np.cov(li[:k], lo[:k])[0, 1]) if k > 1 else 0.0,
              "Variance": vi / vo if vo > 0 else 0.0, "Weight": len(li) * len(lo)})
    return r


def extract(path, cap, transform=None):
    """transform(packets) -> packets is applied to every attack record before its features (src/packet_shaping.py)."""
    fam, typ = family(cap)
    want = attack_proto_of(typ) if fam != "Benign" else None
    open_, rows, nflow, last_sweep = {}, [], 0, None
    ip_macs, mac_ips = {}, {}                      # ARP bindings seen so far in this capture
    sent, recv = {}, {}                            # host -> last HOST_WINDOW (peer, port) sent / received

    def close(key):
        st = open_.pop(key)
        if st["pk"]:
            emit(st)

    def emit(st):
        r = features(st["pk"])
        uses = True if want is None else bool(r[want] > 0)
        if transform is not None and fam != "Benign" and st["ctx"]["att"] > 0:
            r = features(transform(st["pk"]))
        c = st["ctx"]
        n = len(st["pk"])
        a, b = st["hosts"]
        ws, wr = sent.get(a, ()), recv.get(b, ())
        r.update({"arp_reply_share": c["reply"] / n, "arp_grat_share": c["grat"] / n,
                  "arp_ip_macs": c["ip_macs"], "arp_mac_ips": c["mac_ips"],
                  "src_peers": len({x[0] for x in ws}), "src_ports": len({x[1] for x in ws}),
                  "dst_peers": len({x[0] for x in wr}), "dst_ports": len({x[1] for x in wr}),
                  "attacker_share": c["att"] / n})
        rows.append({**r, "capture": cap, "family": fam, "type": typ, "flow": st["id"], "rec": st["rec"],
                     "t0": st["pk"][0][0], "t1": st["pk"][-1][0], "host_a": st["hosts"][0],
                     "host_b": st["hosts"][1], "attack_proto": uses})
        st["rec"] += 1
        st["pk"] = []
        st["ctx"] = {"reply": 0, "grat": 0, "ip_macs": 0, "mac_ips": 0, "att": 0}

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
                    close(key)
                    st = None
                if st is None:
                    st = open_[key] = {"id": nflow, "rec": 0, "pk": [], "first": sender, "hosts": hosts,
                                       "last": ts, "ctx": {"reply": 0, "grat": 0, "ip_macs": 0, "mac_ips": 0, "att": 0}}
                    nflow += 1
                st["last"] = ts
                src, dst = (hosts[0], hosts[1]) if sender == hosts[0] else (hosts[1], hosts[0])
                port = ports[1] if sender == hosts[0] else ports[0]
                sent.setdefault(src, deque(maxlen=HOST_WINDOW)).append((dst, port))
                recv.setdefault(dst, deque(maxlen=HOST_WINDOW)).append((src, port))
                st["ctx"]["att"] += buf[0:6] in ATTACKERS or buf[6:12] in ATTACKERS
                if ind["ARP"]:
                    op, spa, sha, tpa = arp_fields(buf)
                    c = st["ctx"]
                    c["reply"] += op == 2
                    c["grat"] += spa == tpa
                    ms = ip_macs.setdefault(spa, set())
                    if len(ms) < 8:
                        ms.add(sha)
                    ips = mac_ips.setdefault(sha, set())
                    if len(ips) < 64:
                        ips.add(spa)
                    c["ip_macs"] = max(c["ip_macs"], len(ms))
                    c["mac_ips"] = max(c["mac_ips"], len(ips))
                st["pk"].append((ts, sender == st["first"], length, hdr, proto, ttl, flags, ports, ind))
                if len(st["pk"]) == W:
                    emit(st)
                if last_sweep is None:
                    last_sweep = ts
                elif ts - last_sweep > IDLE:
                    for k in [k for k, s in open_.items() if ts - s["last"] > IDLE]:
                        close(k)
                    last_sweep = ts
        except (dpkt.dpkt.NeedData, ValueError):
            pass                       # captures cut at 100 MB end in a partial packet
    for k in list(open_):
        close(k)
    df = pd.DataFrame(rows, columns=FEATURES + META)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df


def attack_mask(df, rule="mac"):
    """Records kept for training/evaluation: benign captures without attacker packets, and attack records:
    rule 'mac' = record has attacker packets (ground truth); 'proto' = record uses the attack's protocol (old)."""
    if rule == "proto":
        return df.attack_proto.to_numpy(bool)
    ben = (df.family == "Benign").to_numpy()
    att = (df.attacker_share > 0).to_numpy()
    return np.where(ben, ~att, att)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(f"{OUT_DIR}/{a.split}", exist_ok=True)
    for path in sorted(glob.glob(f"{PCAP_DIR}/{a.split}/*.pcap")):
        cap = os.path.basename(path)[:-5]
        out = f"{OUT_DIR}/{a.split}/{cap}.parquet"
        if (a.only and a.only not in cap) or os.path.exists(out):
            continue
        df = extract(path, cap)
        df.to_parquet(out + ".part", index=False)
        os.replace(out + ".part", out)
        print(f"{a.split}/{cap}: {len(df)} records, {df.flow.nunique()} flows, "
              f"attack-protocol share {df.attack_proto.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
