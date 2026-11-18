"""Shared definitions for CICIoT2023: the eight classes and the mapping of the
34 fine-grained labels to attack families.
"""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CICIoT2023")
DEFAULT_TEST = os.path.join(DATA_DIR, "test", "test.csv")

# Stage 2 class order. Assumed alphabetical (sklearn LabelEncoder default);
# verified at runtime by checking the confusion-matrix diagonal.
CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]


def macro_group(label: str) -> str:
    if label == "BenignTraffic":
        return "Benign"
    if label.startswith("DDoS-"):
        return "DDoS"
    if label.startswith("DoS-"):
        return "DoS"
    if label.startswith("Mirai-"):
        return "Mirai"
    if label.startswith("Recon-") or label == "VulnerabilityScan":
        return "Recon"
    if label == "DictionaryBruteForce":
        return "BruteForce"
    if label in ("MITM-ArpSpoofing", "DNS_Spoofing"):
        return "Spoofing"
    if label in ("SqlInjection", "XSS", "CommandInjection", "BrowserHijacking",
                 "Backdoor_Malware", "Uploading_Attack"):
        return "Web"
    raise ValueError(f"unmapped label: {label}")
