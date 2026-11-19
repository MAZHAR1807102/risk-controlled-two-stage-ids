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
        folder = "test" if split == "test" else "train"
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


DATASETS = {d.name: d for d in (CICIoT2023(), CICIoMT2024(), CICIoT2023Temporal(), CICIoT2023Random())}


def get_dataset(name):
    return DATASETS[name]
