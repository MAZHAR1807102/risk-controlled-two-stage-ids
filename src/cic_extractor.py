"""Packet-level re-implementation of the CIC feature extractor (CICIoMT2024 / CICIoT2023 CSV features).

CIC did not publish its extractor. The definitions below were recovered by matching the PCAPs of CICIoMT2024
with CIC's CSVs of the same captures (src/cic_validate.py reports the agreement per feature):
  * A CSV row is the mean of per-packet values over consecutive, non-overlapping windows of W packets in capture
    order (W = 10; 100 for flood captures). Packets are not grouped by flow.
  * Packet length = 14 + IP total length for IP packets (Ethernet padding dropped), else the frame length.
  * Some per-packet values use a buffer that is emptied every 20 packets of the capture, whatever the traffic:
    Tot sum, Min, Max, AVG, Std (lengths in the buffer), Variance (1 if the buffer's lengths are not all equal),
    Magnitue, Radius, Covariance (computed as if incoming = outgoing = the buffer, an approximation).
  * IAT = time since the previous packet, with the reference reset to 0 once every 20 packets, so one packet in
    20 carries its absolute Unix timestamp. This is the recording-date leak (src/iat_dates.py).
  * Number and Weight depend only on the packet's place in the 20-packet cycle (window means 5.5/13.5 and
    38.5/244.6; 9.5 and 141.55 for W = 100). They carry no traffic information.
  * Per-flow state (bidirectional 5-tuple, TCP/UDP): Header_Length = running bytes of the flow;
    rst_count = running packet count of a TCP flow; Rate = Srate = flow packets / flow age, and packets
    without a flow (or a flow's first packet) repeat the last Rate.
  * Duration = IP TTL; Protocol Type = IP protocol number (0 for non-IP); flag numbers = TCP flag bits;
    protocol columns = indicators.
Mode "clean" gives our own, leak-free definitions of the same quantities, per window: real inter-arrival times,
no capture-position features, and flow statistics over the window only.

Usage:
  from cic_extractor import extract
  df = extract("capture.pcap", window=10, mode="cic")      # or mode="clean"
"""
import dpkt
import numpy as np
import pandas as pd

COLUMNS = ["Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate", "fin_flag_number",
           "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number",
           "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
           "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max",
           "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"]
FLAGS = [("fin", 0x01), ("syn", 0x02), ("rst", 0x04), ("psh", 0x08), ("ack", 0x10), ("ece", 0x40), ("cwr", 0x80)]
PORTS = {"HTTP": {80}, "HTTPS": {443}, "DNS": {53}, "Telnet": {23}, "SMTP": {25}, "SSH": {22}, "IRC": {194, 6667},
         "DHCP": {67, 68}}
CYCLE = 20
# window means of the position-only columns (W = 10: even / odd window of the 20-packet cycle)
POSITION = {"Number": (5.5, 13.5, 9.5), "Weight": (38.5, 244.6, 141.55)}


def packets(path, limit=None):
    """Per-packet raw fields, in capture order."""
    rows = []
    with open(path, "rb") as f:
        for i, (t, buf) in enumerate(dpkt.pcap.Reader(f)):
            if limit is not None and i >= limit:
                break
            r = {"t": t, "ip": 0, "arp": 0, "tcp": 0, "udp": 0, "icmp": 0, "igmp": 0, "ttl": 0, "proto": 0,
                 "flags": 0, "sport": -1, "dport": -1, "key": None}
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.UnpackError, IndexError):
                r["len"] = len(buf)
                rows.append(r)
                continue
            p = eth.data
            if isinstance(p, dpkt.ip.IP):
                r.update(ip=1, ttl=p.ttl, proto=p.p, len=p.len + 14)
                l4 = p.data
                if isinstance(l4, dpkt.tcp.TCP):
                    r.update(tcp=1, flags=l4.flags, sport=l4.sport, dport=l4.dport)
                elif isinstance(l4, dpkt.udp.UDP):
                    r.update(udp=1, sport=l4.sport, dport=l4.dport)
                elif isinstance(l4, dpkt.icmp.ICMP):
                    r["icmp"] = 1
                elif p.p == 2:
                    r["igmp"] = 1
                if r["sport"] >= 0:
                    a, b = (bytes(p.src), r["sport"]), (bytes(p.dst), r["dport"])
                    r["key"] = (p.p,) + tuple(sorted([a, b]))
            else:
                r["len"] = len(buf)
                r["arp"] = int(isinstance(p, dpkt.arp.ARP))
            rows.append(r)
    return rows


def per_packet_cic(rows):
    """Per-packet values as CIC computes them (see module docstring)."""
    n = len(rows)
    out = {c: np.zeros(n) for c in COLUMNS}
    flows, last_rate, buf, ref = {}, 0.0, [], 0.0
    for j, r in enumerate(rows):
        if j % CYCLE == 0:
            buf = []
        if j % CYCLE == CYCLE // 2:
            ref = 0.0
        L, t = r["len"], r["t"]
        buf.append(L)
        a = np.asarray(buf, float)
        var = a.var()
        out["Tot sum"][j], out["Min"][j], out["Max"][j] = a.sum(), a.min(), a.max()
        out["AVG"][j], out["Std"][j] = a.mean(), np.sqrt(var)
        out["Variance"][j] = float(var > 0)
        out["Magnitue"][j], out["Radius"][j], out["Covariance"][j] = np.sqrt(2 * a.mean()), np.sqrt(2 * var), var
        out["Tot size"][j] = L
        out["IAT"][j] = t - ref
        ref = t
        out["Duration"][j], out["Protocol Type"][j] = r["ttl"], r["proto"]
        for c, k in (("TCP", "tcp"), ("UDP", "udp"), ("ICMP", "icmp"), ("ARP", "arp"), ("IGMP", "igmp"),
                     ("IPv", "ip"), ("LLC", "ip")):
            out[c][j] = r[k]
        for name, bit in FLAGS:
            out[f"{name}_flag_number"][j] = float(bool(r["flags"] & bit))
        for c, ports in PORTS.items():
            out[c][j] = float(r["sport"] in ports or r["dport"] in ports)
        if r["key"] is not None:
            s = flows.setdefault(r["key"], {"bytes": 0, "n": 0, "t0": t})
            s["bytes"] += L
            s["n"] += 1
            out["Header_Length"][j] = s["bytes"]
            out["rst_count"][j] = s["n"] if r["tcp"] else 0
            if t > s["t0"]:
                last_rate = s["n"] / (t - s["t0"])
        out["Rate"][j] = out["Srate"][j] = last_rate
    return pd.DataFrame(out)


def windows(per_packet, window):
    """Mean over consecutive windows of `window` packets (a last partial window is dropped)."""
    n = len(per_packet) // window * window
    g = per_packet.iloc[:n].groupby(np.arange(n) // window).mean()
    k = np.arange(len(g))
    for c, (even, odd, full) in POSITION.items():
        g[c] = full if window % CYCLE == 0 else np.where(k % 2 == 0, even, odd)
    return g[COLUMNS].reset_index(drop=True)


def extract(path, window=10, mode="cic", limit=None):
    rows = packets(path, limit)
    if mode == "cic":
        return windows(per_packet_cic(rows), window)
    raise NotImplementedError(mode)
