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


DATASETS = {d.name: d for d in (CICIoT2023(), CICIoMT2024())}


def get_dataset(name):
    return DATASETS[name]
