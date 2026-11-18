# Datasets (not tracked)

The CIC datasets may not be redistributed. Download them from the Canadian Institute for
Cybersecurity, University of New Brunswick, and put them here:

```
data/
├── CICIoT2023/
│   ├── train/train.csv
│   ├── validation/validation.csv
│   └── test/test.csv
└── CICIoMT2024/
    └── WiFi_and_MQTT/
        ├── train/*.csv      # one CSV per attack; the label is in the file name
        └── test/*.csv
```

- **CICIoT2023:** E. C. P. Neto et al., *Sensors* 23(13):5941, 2023.
  https://www.unb.ca/cic/datasets/iotdataset-2023.html
  The official train/validation/test split, 46 flow features.
- **CICIoMT2024:** S. Dadkhah et al., *Internet of Things* 28:101351, 2024,
  doi:10.1016/j.iot.2024.101351. https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
  Only the Wi-Fi and MQTT CSV files are used (45 features).

`src/datasets.py` reads these paths. For CICIoMT2024, the last 15% of every training file is used
for validation.
