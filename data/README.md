# Datasets (not tracked)

The CIC datasets may not be redistributed. Download them from the Canadian Institute for
Cybersecurity, University of New Brunswick, and put them here:

```
data/
├── CICIoT2023/
│   ├── train/train.csv
│   ├── validation/validation.csv
│   ├── test/test.csv
│   └── per_attack_csv/<attack>/<attack>N.pcap.csv   # CSV/CSV folder of the CIC release (time-ordered test)
├── CICIoMT2024/
│   ├── WiFi_and_MQTT/
│   │   ├── train/*.csv      # one CSV per attack; the label is in the file name
│   │   └── test/*.csv
│   └── PCAP_full/
│       ├── train/*.pcap     # Wi-Fi/MQTT packet captures (flood captures: first 100 MB are enough)
│       └── test/*.pcap
└── kitsune/                 # optional cross-dataset test (Mirsky et al., NDSS 2018)
    ├── OS_Scan_pcap.pcapng, OS_Scan_labels.csv
    ├── SSDP_Flood_pcap.pcap, SSDP_Flood_labels.csv
    └── ARP_MitM_pcap.pcapng, ARP_MitM_labels.csv
```

- **CICIoT2023:** E. C. P. Neto et al., *Sensors* 23(13):5941, 2023.
  https://www.unb.ca/cic/datasets/iotdataset-2023.html
  The official train/validation/test split, 46 flow features. For the time-ordered test, also the
  per-attack files of the CSV/CSV folder (39 features, one folder per attack, rows in extraction order).
- **CICIoMT2024:** S. Dadkhah et al., *Internet of Things* 28:101351, 2024,
  doi:10.1016/j.iot.2024.101351. https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
  The Wi-Fi and MQTT CSV files (45 features) and the matching packet captures. From the captures,
  `src/flow_records.py` builds our own flow records (up to 10 packets per flow, labels from the four
  attacker MAC addresses); `src/build_packet_dataset.py` turns them into the train/validation/test sets.
- **Kitsune:** Y. Mirsky et al., NDSS 2018 (Kaggle "Kitsune Network Attack Dataset"); only for the
  cross-dataset check.

`src/datasets.py` reads these paths. For CICIoMT2024, the last 15% of every training file is used
for validation.
