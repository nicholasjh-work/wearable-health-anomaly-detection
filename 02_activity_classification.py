# Databricks notebook source
# Wearable Health Anomaly Detection Pipeline\n# Nicholas Hidalgo | github.com/nicholasjh-work\n# Notebook 02: Activity classification with Random Forest

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

silver_active = silver[silver["activity_label"] > 0].copy()
print(f"Active samples: {silver_active.shape[0]:,}")

FEATURE_COLS = [
    "chest_accel_mag_mean", "chest_accel_mag_std", "chest_accel_mag_range",
    "ankle_accel_mag_mean", "ankle_accel_mag_std", "ankle_accel_mag_range",
    "arm_accel_mag_mean", "arm_accel_mag_std", "arm_accel_mag_range",
    "ankle_gyro_mag_mean", "ankle_gyro_mag_std",
    "arm_gyro_mag_mean", "arm_gyro_mag_std",
    "ecg_lead1_mean", "ecg_lead1_std", "ecg_lead1_range",
    "hrv_proxy_sdnn", "strain_score", "recovery_proxy"
]

TOP_10_FEATURES = [
    "chest_accel_mag_mean", "chest_accel_mag_std",
    "ankle_accel_mag_mean", "ankle_accel_mag_std",
    "arm_accel_mag_mean", "arm_accel_mag_std",
    "ankle_gyro_mag_mean", "arm_gyro_mag_mean",
    "strain_score", "hrv_proxy_sdnn"
]

X = silver_active[FEATURE_COLS].fillna(0)
y = silver_active["activity_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

# COMMAND ----------

clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"Run 1 - All features ({len(FEATURE_COLS)}): accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")
print(classification_report(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("Feature importances:")
print(importances.to_string())

# COMMAND ----------

X_train_r = X_train[TOP_10_FEATURES]
X_test_r = X_test[TOP_10_FEATURES]

clf_r = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
clf_r.fit(X_train_r, y_train)
y_pred_r = clf_r.predict(X_test_r)

acc_r = accuracy_score(y_test, y_pred_r)
f1_r = f1_score(y_test, y_pred_r, average="macro")

print(f"Run 2 - Top 10 features: accuracy={acc_r:.4f}, f1_macro={f1_r:.4f}")
print(f"Accuracy delta: {(acc - acc_r)*100:.2f}pp")
print(f"Feature reduction: {len(FEATURE_COLS)} -> {len(TOP_10_FEATURES)} ({(1-len(TOP_10_FEATURES)/len(FEATURE_COLS))*100:.0f}% fewer)")
