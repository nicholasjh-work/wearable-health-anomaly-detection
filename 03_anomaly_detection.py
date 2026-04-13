# Databricks notebook source
# Wearable Health Anomaly Detection Pipeline\n# Nicholas Hidalgo | github.com/nicholasjh-work\n# Notebook 03: Anomaly detection with Isolation Forest

# COMMAND ----------

import pandas as pd
import numpy as np
import os, zipfile, urllib.request
from pathlib import Path
from scipy.signal import find_peaks

SILVER_PATH = "/tmp/mhealth_silver"
BRONZE_PATH = "/tmp/mhealth_bronze"
SAMPLE_RATE_HZ = 50
WINDOW_SEC = 5
WINDOW_SAMPLES = WINDOW_SEC * SAMPLE_RATE_HZ

if not os.path.exists(f"{SILVER_PATH}.parquet"):
    print("Building bronze + silver from scratch...")
    DATA_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
    LOCAL_DIR = "/tmp/mhealth"
    ZIP_PATH = f"{LOCAL_DIR}/mhealth.zip"
    COLUMN_NAMES = ["chest_accel_x","chest_accel_y","chest_accel_z","ecg_lead_1","ecg_lead_2","ankle_accel_x","ankle_accel_y","ankle_accel_z","ankle_gyro_x","ankle_gyro_y","ankle_gyro_z","ankle_mag_x","ankle_mag_y","ankle_mag_z","arm_accel_x","arm_accel_y","arm_accel_z","arm_gyro_x","arm_gyro_y","arm_gyro_z","arm_mag_x","arm_mag_y","arm_mag_z","activity_label"]
    ACTIVITY_MAP = {0:"null_class",1:"standing_still",2:"sitting_relaxing",3:"lying_down",4:"walking",5:"climbing_stairs",6:"waist_bends_forward",7:"frontal_elevation_arms",8:"knees_bending",9:"cycling",10:"jogging",11:"running",12:"jump_front_back"}
    os.makedirs(LOCAL_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(LOCAL_DIR)
    frames = []
    for f in sorted(Path(LOCAL_DIR).rglob("mHealth_subject*.log")):
        sid = int(f.stem.replace("mHealth_subject",""))
        d = pd.read_csv(f, sep=r"\s+", header=None, names=COLUMN_NAMES)
        d["subject_id"] = sid
        d["sample_idx"] = range(len(d))
        d["timestamp_sec"] = d["sample_idx"] / SAMPLE_RATE_HZ
        frames.append(d)
    bronze = pd.concat(frames, ignore_index=True)
    bronze["activity_name"] = bronze["activity_label"].map(ACTIVITY_MAP)
    for loc in ["chest","ankle","arm"]:
        bronze[f"{loc}_accel_mag"] = np.sqrt(bronze[f"{loc}_accel_x"]**2 + bronze[f"{loc}_accel_y"]**2 + bronze[f"{loc}_accel_z"]**2)
    bronze.to_parquet(f"{BRONZE_PATH}.parquet", index=False)
    def _rolling(group):
        w = WINDOW_SAMPLES
        feats = pd.DataFrame(index=group.index)
        for loc in ["chest","ankle","arm"]:
            m = f"{loc}_accel_mag"
            feats[f"{loc}_accel_mag_mean"] = group[m].rolling(w, min_periods=1).mean()
            feats[f"{loc}_accel_mag_std"] = group[m].rolling(w, min_periods=1).std().fillna(0)
            feats[f"{loc}_accel_mag_max"] = group[m].rolling(w, min_periods=1).max()
            feats[f"{loc}_accel_mag_range"] = group[m].rolling(w, min_periods=1).max() - group[m].rolling(w, min_periods=1).min()
        for loc in ["ankle","arm"]:
            for ax in ["x","y","z"]:
                c = f"{loc}_gyro_{ax}"
                feats[f"{c}_mean"] = group[c].rolling(w, min_periods=1).mean()
                feats[f"{c}_std"] = group[c].rolling(w, min_periods=1).std().fillna(0)
        for loc in ["ankle","arm"]:
            gm = np.sqrt(group[f"{loc}_gyro_x"]**2+group[f"{loc}_gyro_y"]**2+group[f"{loc}_gyro_z"]**2)
            feats[f"{loc}_gyro_mag_mean"] = gm.rolling(w, min_periods=1).mean()
            feats[f"{loc}_gyro_mag_std"] = gm.rolling(w, min_periods=1).std().fillna(0)
        feats["ecg_lead1_mean"] = group["ecg_lead_1"].rolling(w, min_periods=1).mean()
        feats["ecg_lead1_std"] = group["ecg_lead_1"].rolling(w, min_periods=1).std().fillna(0)
        feats["ecg_lead1_range"] = group["ecg_lead_1"].rolling(w, min_periods=1).max() - group["ecg_lead_1"].rolling(w, min_periods=1).min()
        return feats
    def _hrv(ecg, window=250):
        hrv = pd.Series(np.nan, index=ecg.index)
        for s in range(0, len(ecg)-window+1, window):
            seg = ecg.iloc[s:s+window].values
            pk, _ = find_peaks(seg, distance=20, height=np.percentile(seg, 75))
            hrv.iloc[s:s+window] = np.std(np.diff(pk)/SAMPLE_RATE_HZ) if len(pk)>2 else 0.0
        return hrv.ffill().fillna(0)
    def _strain(row):
        a = (row.get("chest_accel_mag_mean",0)+row.get("ankle_accel_mag_mean",0)+row.get("arm_accel_mag_mean",0))/3
        g = (row.get("ankle_gyro_mag_mean",0)+row.get("arm_gyro_mag_mean",0))/2
        return a*0.6+g*0.4
    sfs = []
    for sid, grp in bronze.groupby("subject_id"):
        grp = grp.sort_values("sample_idx").reset_index(drop=True)
        rf = _rolling(grp)
        mg = pd.concat([grp, rf], axis=1)
        mg["hrv_proxy_sdnn"] = _hrv(grp["ecg_lead_1"])
        mg["strain_raw"] = mg.apply(_strain, axis=1)
        rest = mg["activity_label"].isin([0,1,2,3])
        mg["is_recovery_window"] = rest.astype(int)
        mg["recovery_proxy"] = np.where(rest, 10.0-mg["strain_raw"].clip(0,10), 0.0)
        sfs.append(mg)
    sf = pd.concat(sfs, ignore_index=True)
    lo, hi = sf["strain_raw"].quantile(0.01), sf["strain_raw"].quantile(0.99)
    sf["strain_score"] = ((sf["strain_raw"]-lo)/(hi-lo)*10).clip(0,10)
    silver = sf.iloc[::WINDOW_SAMPLES].reset_index(drop=True)
    silver.to_parquet(f"{SILVER_PATH}.parquet", index=False)
    print(f"Silver built: {silver.shape[0]:,} rows x {silver.shape[1]} cols")
else:
    silver = pd.read_parquet(f"{SILVER_PATH}.parquet")
    print(f"Silver loaded: {silver.shape[0]:,} rows x {silver.shape[1]} cols")

# COMMAND ----------

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

ANOMALY_FEATURES = [
    "chest_accel_mag_mean", "chest_accel_mag_std",
    "ankle_accel_mag_mean", "ankle_accel_mag_std",
    "arm_accel_mag_mean", "arm_accel_mag_std",
    "ankle_gyro_mag_mean", "ankle_gyro_mag_std",
    "arm_gyro_mag_mean", "arm_gyro_mag_std",
    "ecg_lead1_std", "ecg_lead1_range",
    "hrv_proxy_sdnn", "strain_score"
]

X_anomaly = silver[ANOMALY_FEATURES].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

def create_ground_truth(df):
    flags = pd.Series(0, index=df.index)
    for label in df["activity_label"].unique():
        mask = df["activity_label"] == label
        strain = df.loc[mask, "strain_score"]
        q1, q3 = strain.quantile(0.05), strain.quantile(0.95)
        iqr = q3 - q1
        lower, upper = q1 - 2.0*iqr, q3 + 2.0*iqr
        outlier = mask & ((df["strain_score"] < lower) | (df["strain_score"] > upper))
        flags[outlier] = 1
    return flags

silver["anomaly_ground_truth"] = create_ground_truth(silver)
gt = silver["anomaly_ground_truth"]
gt_rate = gt.mean()
print(f"Ground truth anomaly rate: {gt_rate:.4f} ({gt.sum()} samples)")

# COMMAND ----------

iso_v1 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
preds_v1 = iso_v1.fit_predict(X_scaled)
silver["anomaly_v1"] = np.where(preds_v1 == -1, 1, 0)
silver["anomaly_score_v1"] = -iso_v1.score_samples(X_scaled)

p1 = precision_score(gt, silver["anomaly_v1"], zero_division=0)
r1 = recall_score(gt, silver["anomaly_v1"], zero_division=0)
f1_v1 = f1_score(gt, silver["anomaly_v1"], zero_division=0)
print(f"v1 (contamination=0.05): precision={p1:.4f}, recall={r1:.4f}, f1={f1_v1:.4f}")

# COMMAND ----------

tuned_contamination = min(gt_rate * 1.5, 0.15)

iso_v2 = IsolationForest(n_estimators=300, contamination=tuned_contamination, random_state=42, n_jobs=-1)
preds_v2 = iso_v2.fit_predict(X_scaled)
silver["anomaly_v2"] = np.where(preds_v2 == -1, 1, 0)
silver["anomaly_score_v2"] = -iso_v2.score_samples(X_scaled)

p2 = precision_score(gt, silver["anomaly_v2"], zero_division=0)
r2 = recall_score(gt, silver["anomaly_v2"], zero_division=0)
f1_v2 = f1_score(gt, silver["anomaly_v2"], zero_division=0)
print(f"v2 (contamination={tuned_contamination:.4f}): precision={p2:.4f}, recall={r2:.4f}, f1={f1_v2:.4f}")
print(f"F1 improvement: {(f1_v2-f1_v1)*100:.2f}pp")

silver.to_parquet(f"{SILVER_PATH}.parquet", index=False)
print("Silver updated with anomaly scores.")
