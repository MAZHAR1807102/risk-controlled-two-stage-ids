"""Dataset loaders for the two-stage IDS pipeline.

Each dataset returns raw (unscaled) float32 features, the macro class of every row,
the original fine-grained label, and the feature names. Class order is alphabetical
with "Benign" first only by coincidence of the names; code must use CLASSES.index().
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from eval_offline import CLASSES as CICIOT2023_CLASSES, macro_group

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class CICIoT2023:
    name = "ciciot2023"
    classes = CICIOT2023_CLASSES
    # Families that resemble benign traffic; reported separately as "stealthy" miss rate.
    stealthy = ["BruteForce", "Recon", "Spoofing", "Web"]

    def load(self, split, frac=1.0):
        df = pd.read_csv(os.path.join(ROOT, "CICIoT2023", split, f"{split}.csv"))
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=0)
        raw = df["label"].to_numpy()
        lut = {l: macro_group(l) for l in np.unique(raw)}
        g = np.array([lut[l] for l in raw])
        return df.iloc[:, :46].to_numpy(np.float32), g, raw, list(df.columns[:46])


class CICIoMT2024:
    """Wi-Fi/MQTT part of CICIoMT2024 (Dadkhah et al., Internet of Things, 2024).

    One CSV per attack; the label is the file name. There is no validation split, so the
    last VAL_FRAC of every training file (in file order, i.e. time order) is held out.
    A contiguous tail is used instead of a random split because neighbouring rows are
    overlapping windows of the same capture and would leak across a random split.
    """
    name = "ciciomt2024"
    classes = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
    stealthy = ["MQTT", "Recon", "Spoofing"]
    VAL_FRAC = 0.15
    DIR = os.path.join(ROOT, "CICIoMT2024", "WiFi_and_MQTT")
    FOLDERS = {"train": "train", "test": "test"}  # where the model / the evaluation captures come from

    @staticmethod
    def attack_name(path):
        stem = re.sub(r"_(train|test)\.pcap\.csv$", "", os.path.basename(path))
        return re.sub(r"\d+$", "", stem)  # TCP_IP-DDoS-ICMP3 -> TCP_IP-DDoS-ICMP

    @staticmethod
    def family(attack):
        if attack == "Benign":
            return "Benign"
        if attack == "ARP_Spoofing":
            return "Spoofing"
        if attack.startswith("MQTT-"):
            return "MQTT"
        if attack.startswith("Recon-"):
            return "Recon"
        if attack.startswith("TCP_IP-DDoS-"):
            return "DDoS"
        if attack.startswith("TCP_IP-DoS-"):
            return "DoS"
        raise ValueError(f"unmapped CICIoMT2024 file: {attack}")

    def load(self, split, frac=1.0):
        folder = self.FOLDERS["test"] if split == "test" else self.FOLDERS["train"]
        Xs, gs, raws, feats = [], [], [], None
        for path in sorted(glob.glob(os.path.join(self.DIR, folder, "*.csv"))):
            df = pd.read_csv(path)
            if feats is None:
                feats = list(df.columns)
            elif list(df.columns) != feats:
                raise ValueError(f"column mismatch in {path}")
            if split != "test":
                cut = int(round(len(df) * (1 - self.VAL_FRAC)))
                df = df.iloc[:cut] if split == "train" else df.iloc[cut:]
            if frac < 1.0:
                df = df.sample(frac=frac, random_state=0)
            attack = self.attack_name(path)
            Xs.append(df.to_numpy(np.float32))
            raws.append(np.full(len(df), attack, dtype=object))
            gs.append(np.full(len(df), self.family(attack), dtype=object))
        return np.concatenate(Xs), np.concatenate(gs).astype(str), np.concatenate(raws).astype(str), feats


class CICIoMT2024Swapped(CICIoMT2024):
    """Capture-level replication: the roles of the CIC train and test captures are swapped. The model is
    trained on the test captures (validation = their last 15%), and the separate train captures play the
    new site."""
    name = "ciciomt2024s"
    FOLDERS = {"train": "test", "test": "train"}


class CICIoT2023Temporal(CICIoT2023):
    """CICIoT2023 per-attack CSV files (39 features), cut in time order inside every file:
    first 70% train, next 15% validation, last 15% test. Built by src/build_ciciot_temporal.py."""
    name = "ciciot2023t"
    DIR = os.path.join(ROOT, "CICIoT2023", "temporal")

    def load(self, split, frac=1.0):
        z = np.load(os.path.join(self.DIR, f"{split}.npz"))
        X, raw = z["X"], z["label"]
        if frac < 1.0:
            keep = np.random.default_rng(0).random(len(raw)) < frac
            X, raw = X[keep], raw[keep]
        lut = {l: macro_group(l) for l in np.unique(raw)}
        g = np.array([lut[l] for l in raw])
        return X, g, raw, [str(f) for f in z["features"]]


class CICIoT2023Random(CICIoT2023Temporal):
    """Control for CICIoT2023Temporal: the same rows, pooled and split 70/15/15 at random (seed 0),
    so a difference between the two shows the effect of the time split alone."""
    name = "ciciot2023r"

    def load(self, split, frac=1.0):
        parts = [CICIoT2023Temporal.load(self, s) for s in ("train", "validation", "test")]
        X = np.concatenate([q[0] for q in parts])
        g = np.concatenate([q[1] for q in parts])
        raw = np.concatenate([q[2] for q in parts])
        perm = np.random.default_rng(0).permutation(len(g))
        a, b = int(round(0.70 * len(g))), int(round(0.85 * len(g)))
        idx = np.sort(perm[{"train": slice(0, a), "validation": slice(a, b), "test": slice(b, None)}[split]])
        if frac < 1.0:
            idx = idx[np.random.default_rng(1).random(len(idx)) < frac]
        return X[idx], g[idx], raw[idx], parts[0][3]


class CICIoMT2024Packet(CICIoMT2024):
    """Clean flow records computed from the CICIoMT2024 PCAPs (src/flow_records.py, src/build_packet_dataset.py).
    Split C: train/validation = the same train captures cut in time (85/15), test = the test captures."""
    name = "ciciomt2024p"
    SPLIT = "C"
    DROP = ()
    DIR = os.path.join(ROOT, "CICIoMT2024", "packet")

    def _z(self, split):
        return np.load(os.path.join(self.DIR, self.SPLIT, f"{split}.npz"))

    def load(self, split, frac=1.0):
        z = self._z(split)
        feats = [str(f) for f in z["features"]]
        keep = [i for i, f in enumerate(feats) if f not in self.DROP]
        X, fam, typ = z["X"][:, keep], z["fam"].astype(str), z["typ"].astype(str)
        if frac < 1.0:
            m = np.random.default_rng(0).random(len(fam)) < frac
            X, fam, typ = X[m], fam[m], typ[m]
        return X, fam, typ, [feats[i] for i in keep]

    def meta(self, split):
        z = self._z(split)
        return {k: z[k] for k in ("capture", "t0", "flow", "host_a", "host_b")}


class CICIoMT2024PacketNoTime(CICIoMT2024Packet):
    """Split C without any timing feature (what an attacker can set by delaying packets)."""
    name = "ciciomt2024pn"
    DROP = ("Rate", "Srate", "Drate", "IAT", "IAT_std", "IAT_min", "IAT_max", "Span")


class CICIoMT2024PacketRandom(CICIoMT2024Packet):
    """Control: the same records pooled and split 70/15/15 at random."""
    name = "ciciomt2024pr"
    SPLIT = "R"


class CICIoMT2024PacketContext(CICIoMT2024Packet):
    """Split C with gateway context features (ARP bindings, host peers/ports over the last 50 packets)."""
    name = "ciciomt2024pc"
    DIR = os.path.join(ROOT, "CICIoMT2024", "packet_ctx")


class CICIoMT2024PacketContextNoTime(CICIoMT2024PacketContext):
    """Split C with context features and without any timing feature."""
    name = "ciciomt2024pcn"
    DROP = CICIoMT2024PacketNoTime.DROP


class CICIoMT2024PacketMAC(CICIoMT2024Packet):
    """Split C, records labelled by attacker-MAC ground truth, without timing features (main setting)."""
    name = "ciciomt2024pmn"
    DIR = os.path.join(ROOT, "CICIoMT2024", "packet_mac")
    DROP = CICIoMT2024PacketNoTime.DROP + tuple(["arp_reply_share", "arp_grat_share", "arp_ip_macs", "arp_mac_ips",
                                                 "src_peers", "src_ports", "dst_peers", "dst_ports"])


class CICIoMT2024PacketMACContext(CICIoMT2024PacketMAC):
    """Attacker-MAC labels, context features, no timing."""
    name = "ciciomt2024pmxn"
    DROP = CICIoMT2024PacketNoTime.DROP


class CICIoMT2024PacketMACNoOS(CICIoMT2024PacketMAC):
    """Main MAC setting minus OS-fingerprint clues (TTL = Duration, Header_Length): does the model lean on the
    attacker devices' operating system rather than on the attack? A large drop here = fingerprinting."""
    name = "ciciomt2024pmno"
    DROP = CICIoMT2024PacketMAC.DROP + ("Duration", "Header_Length")


class CICIoMT2024PacketMACSwapArp(CICIoMT2024PacketMAC):
    """Main MAC setting on the dataset where the two ARP_Spoofing captures swap roles (train<->test), so the
    ARP test set is the larger capture (option C). Built into packet_mac_swaparp by build_packet_dataset."""
    name = "ciciomt2024pmns"
    DIR = os.path.join(ROOT, "CICIoMT2024", "packet_mac_swaparp")


class CICIoMT2024PacketMACTiming(CICIoMT2024PacketMAC):
    """Attacker-MAC labels, timing features, no context."""
    name = "ciciomt2024pm"
    DROP = ("arp_reply_share", "arp_grat_share", "arp_ip_macs", "arp_mac_ips", "src_peers", "src_ports", "dst_peers",
            "dst_ports")


class CICIoMT2024NoArtifacts(CICIoMT2024):
    """CIC's CSV features without the three artifact columns (IAT = timestamp, Number/Weight = window size)."""
    name = "ciciomt2024x"
    DROP = ("IAT", "Number", "Weight")

    def load(self, split, frac=1.0):
        X, g, raw, feats = super().load(split, frac)
        keep = [i for i, f in enumerate(feats) if f not in self.DROP]
        return X[:, keep], g, raw, [feats[i] for i in keep]


class CICIoT2023NoArtifacts(CICIoT2023):
    """The merged CICIoT2023 CSV without IAT (recording time) and Number/Weight (window size by attack type)."""
    name = "ciciot2023x"
    DROP = ("IAT", "Number", "Weight")

    def load(self, split, frac=1.0):
        X, g, raw, feats = super().load(split, frac)
        keep = [i for i, f in enumerate(feats) if f not in self.DROP]
        return X[:, keep], g, raw, [feats[i] for i in keep]


DATASETS = {d.name: d for d in (CICIoT2023(), CICIoMT2024(), CICIoT2023Temporal(), CICIoT2023Random(),
                                 CICIoMT2024Swapped(), CICIoMT2024Packet(), CICIoMT2024PacketNoTime(),
                                 CICIoMT2024PacketRandom(), CICIoMT2024NoArtifacts(), CICIoMT2024PacketContext(),
                                 CICIoMT2024PacketContextNoTime(), CICIoMT2024PacketMAC(), CICIoMT2024PacketMACContext(),
                                 CICIoMT2024PacketMACTiming(), CICIoMT2024PacketMACNoOS(),
                                 CICIoMT2024PacketMACSwapArp(),
                                 CICIoT2023NoArtifacts())}


def get_dataset(name):
    return DATASETS[name]
