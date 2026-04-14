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
  <a href="https://github.com/nicholasjh-work/wearable-health-anomaly-detection"><img src="https://img.shields.io/badge/Notebooks-6-16a34a?style=for-the-badge" alt="Notebooks"></a>
  <a href="https://github.com/nicholasjh-work/wearable-health-anomaly-detection"><img src="https://img.shields.io/badge/Accuracy-99.3%25-7c3aed?style=for-the-badge" alt="Accuracy"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white" alt="Databricks">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=flat" alt="matplotlib">
</p>

---

### What This Does

<table>
<tr>
<td>

**1.2M sensor readings** from **10 subjects** performing **12 physical activities** flow through a medallion architecture (Bronze/Silver/Gold) on Databricks.

The pipeline engineers health-relevant features from raw accelerometer, gyroscope, and ECG signals: **HRV proxy** (SDNN from inter-peak intervals), **strain scoring** (0-10 composite), and **recovery window detection**. A Random Forest classifier achieves **99.3% accuracy** (F1 0.993) across all 12 activities. Isolation Forest anomaly detection runs through an **A/B evaluation framework** with paired t-test, Cohen's d, and Ship/Iterate/Kill decision logic.

**Real sensor data. Real statistical evaluation. No toy datasets.**

</td>
</tr>
</table>

---

### Key Results

| Metric | Value |
|:-------|:------|
| ![Classification](https://img.shields.io/badge/Activity_Classification-7c3aed?style=flat-square) | **99.3% accuracy, F1 0.993** across 12 activities |
| ![Strain](https://img.shields.io/badge/Strain_Scoring-2563EB?style=flat-square) | Running 8.9, jogging 5.6, walking 1.9, sedentary <1.0 |
| ![HRV](https://img.shields.io/badge/HRV_Proxy-16a34a?style=flat-square) | 5x variation across subjects (0.036 to 0.188 SDNN) |
| ![Anomaly](https://img.shields.io/badge/Anomaly_Detection-d29922?style=flat-square) | Isolation Forest v1 vs v2 with A/B evaluation |
| ![Dataset](https://img.shields.io/badge/Dataset-8b949e?style=flat-square) | 1.2M rows, 23 sensor columns, 50Hz sampling rate |

---

### Results

![Strain by Activity](strain_by_activity.png)

![Activity Classification](confusion_matrix.png)

![HRV by Subject](hrv_by_subject.png)

![Strain vs HRV](strain_vs_hrv.png)

![Anomaly Detection A/B](anomaly_ab.png)

---

### Architecture

```
UCI MHEALTH (10 subjects, 23 sensors, 50Hz, 1.2M rows)
        │
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Bronze    │ →  │    Silver    │ →  │     Gold     │
│   Raw Data   │    │  Engineered  │    │  Aggregated  │
│  1.2M rows   │    │   Features   │    │   Metrics    │
└──────────────┘    └──────────────┘    └──────────────┘
                          │
                 ┌────────┴────────┐
                 │   ML Pipeline   │
                 │  RandomForest   │
                 │  IsolationForest│
                 │  A/B Evaluation │
                 └─────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Results Dashboard│
                 │   (matplotlib)   │
                 └─────────────────┘
```

---

### Notebooks

| # | Notebook | Purpose |
|:--|:---------|:--------|
| 01 | [![ingest_and_features](https://img.shields.io/badge/01__ingest__and__features-161b22?style=flat-square&logo=databricks&logoColor=FF3621)](01_ingest_and_features.py) | Download MHEALTH, build bronze + silver with rolling features |
| 02 | [![activity_classification](https://img.shields.io/badge/02__activity__classification-161b22?style=flat-square&logo=databricks&logoColor=FF3621)](02_activity_classification.py) | Random Forest classifier (19 vs 10 features) |
| 03 | [![anomaly_detection](https://img.shields.io/badge/03__anomaly__detection-161b22?style=flat-square&logo=databricks&logoColor=FF3621)](03_anomaly_detection.py) | Isolation Forest v1 vs v2 comparison |
| 04 | [![ab_experiment_eval](https://img.shields.io/badge/04__ab__experiment__eval-161b22?style=flat-square&logo=databricks&logoColor=FF3621)](04_ab_experiment_eval.py) | Paired t-test, Cohen's d, Ship/Iterate/Kill decision |
| 05 | [![gold_layer_export](https://img.shields.io/badge/05__gold__layer__export-161b22?style=flat-square&logo=databricks&logoColor=FF3621)](05_gold_layer_export.py) | Aggregate to gold tables, export CSVs |
| viz | [![results_dashboard](https://img.shields.io/badge/viz__results__dashboard-161b22?style=flat-square&logo=databricks&logoColor=FF3621)](viz_p1_results_dashboard.py) | Strain, HRV, confusion matrix, A/B comparison charts |

---

### Feature Engineering

| Feature | Method | Output |
|:--------|:-------|:-------|
| ![Rolling](https://img.shields.io/badge/Rolling_Windows-58a6ff?style=flat-square) | 5-second windows over accel, gyro, ECG | Mean, std, max, range per sensor per window |
| ![HRV](https://img.shields.io/badge/HRV_Proxy-16a34a?style=flat-square) | SDNN from ECG inter-peak intervals | Continuous HRV estimate per window |
| ![Strain](https://img.shields.io/badge/Strain_Score-d29922?style=flat-square) | Weighted composite (60% accel + 40% gyro) | 0-10 scale normalized to population range |
| ![Recovery](https://img.shields.io/badge/Recovery_Windows-7c3aed?style=flat-square) | Activity labels 0-3 flagged as rest | Binary recovery flag + recovery quality proxy |

---

### A/B Experiment Framework

> Anomaly detection models evaluated using statistical significance testing, not just accuracy metrics.

| Component | Description |
|:----------|:------------|
| ![v1](https://img.shields.io/badge/v1_default-58a6ff?style=flat-square) | Isolation Forest, contamination=0.05, n_estimators=200 |
| ![v2](https://img.shields.io/badge/v2_tuned-3fb950?style=flat-square) | Isolation Forest, adaptive contamination, n_estimators=300 |
| ![Stats](https://img.shields.io/badge/Statistics-d29922?style=flat-square) | Paired t-test, Cohen's d effect size |
| ![Decision](https://img.shields.io/badge/Decision-f85149?style=flat-square) | Ship / Iterate / Kill framework based on lift + significance |

---

### Tech Stack

| Component | Technology |
|:----------|:-----------|
| ![Compute](https://img.shields.io/badge/Compute-FF3621?style=flat-square) | Databricks Community Edition (Serverless) |
| ![ML](https://img.shields.io/badge/ML-F7931E?style=flat-square) | scikit-learn (RandomForest, IsolationForest) |
| ![Stats](https://img.shields.io/badge/Statistics-8CAAE6?style=flat-square) | SciPy (paired t-test, Cohen's d) |
| ![Data](https://img.shields.io/badge/Data-150458?style=flat-square) | pandas, NumPy |
| ![Viz](https://img.shields.io/badge/Visualization-11557C?style=flat-square) | matplotlib |
| ![Lang](https://img.shields.io/badge/Language-3776AB?style=flat-square) | Python 3.11 |

---

### Dataset

[UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) — 10 subjects, 23 sensor columns, 12 activity labels, 50Hz sampling. CC BY 4.0 license.

---

<p align="center">
  <a href="https://linkedin.com/in/nicholashidalgo"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>&nbsp;
  <a href="https://nicholashidalgo.com"><img src="https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=About.me&logoColor=white" alt="Website"></a>&nbsp;
  <a href="mailto:analytics@nicholashidalgo.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"></a>
</p>
