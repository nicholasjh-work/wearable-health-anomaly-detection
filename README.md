<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/nh-logo-dark.svg" width="80">
    <source media="(prefers-color-scheme: light)" srcset="assets/nh-logo-light.svg" width="80">
    <img alt="Hidalgo Systems Labs" src="assets/nh-logo-light.svg" width="80">
  </picture>
</p>

<h1 align="center">Wearable Health Anomaly Detection Pipeline</h1>
<p align="center"><b>Sensor data pipeline with activity classification, anomaly detection, and A/B experiment evaluation</b></p>

<p align="center">
  <a href="https://github.com/nicholasjh-work/wearable-health-anomaly-detection"><img src="https://img.shields.io/badge/Platform-Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white" alt="Databricks"></a>
  <a href="https://github.com/nicholasjh-work/wearable-health-anomaly-detection/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

---

### Overview

End-to-end wearable biometric data pipeline built on Databricks. Ingests real sensor data (accelerometer, gyroscope, ECG) from 10 subjects performing 12 physical activities, engineers health-relevant features (HRV proxy, strain scoring, recovery windows), and runs activity classification and anomaly detection with A/B model evaluation.

---

### Key Results

- **Activity classification: 99.3% accuracy, F1 0.993** across 12 activities from 1.2M sensor readings
- **Strain scoring (0-10):** Running 8.9, jogging 5.6, walking 1.9, sedentary <1.0
- **HRV proxy:** 5x variation across subjects (0.036 to 0.188 SDNN)
- **Anomaly detection:** Isolation Forest v1 vs v2 with A/B evaluation (paired t-test, Cohen's d, Ship/Iterate/Kill framework)
- **Recovery windows:** Automatic detection of rest periods with recovery quality scoring

---

### Architecture

```
UCI MHEALTH (10 subjects, 23 sensors, 50Hz, 1.2M rows)
    │
    ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Bronze  │ →  │  Silver  │ →  │   Gold   │
│ Raw Data │    │ Features │    │ Metrics  │
└──────────┘    └──────────┘    └──────────┘
                     │
              ┌──────┴──────┐
              │  ML Models  │
              │ RandomForest│
              │ IsolationF. │
              │ A/B Eval    │
              └─────────────┘
```

---

### Tech Stack

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)

---

### Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `ingest_and_features` | Download MHEALTH, build bronze + silver with rolling features |
| 02 | `activity_classification` | Random Forest classifier (19 vs 10 features) |
| 03 | `anomaly_detection` | Isolation Forest v1 vs v2 comparison |
| 04 | `ab_experiment_eval` | Paired t-test, Cohen's d, Ship/Iterate/Kill decision |
| 05 | `gold_layer_export` | Aggregate to gold tables, export CSVs |
| viz | `results_dashboard` | Charts: strain by activity, HRV, confusion matrix, A/B bars |

---

### Feature Engineering

**5-second rolling windows** over accelerometer, gyroscope, and ECG signals. **HRV proxy** (SDNN approximation from ECG inter-peak intervals). **Composite strain scoring** (0-10 scale from accel + gyro magnitudes). **Recovery window detection** during sedentary activities (labels 0-3).

---

### Dataset

[UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) — 10 subjects, 23 sensor columns, 12 activity labels, 50Hz sampling. CC BY 4.0 license.

---

<p align="center">
  <a href="https://linkedin.com/in/nicholashidalgo"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>&nbsp;
  <a href="https://nicholashidalgo.com"><img src="https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=About.me&logoColor=white" alt="Website"></a>&nbsp;
  <a href="mailto:analytics@nicholashidalgo.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"></a>
</p>
