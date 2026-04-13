# Databricks notebook source
# Wearable Health Anomaly Detection - Results Dashboard\n# Nicholas Hidalgo | github.com/nicholasjh-work

# COMMAND ----------

import pandas as pd
import numpy as np
import os, zipfile, urllib.request
from pathlib import Path
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

SILVER_PATH = "/tmp/mhealth_silver"
SAMPLE_RATE_HZ = 50
WINDOW_SAMPLES = 250

if not os.path.exists(f"{SILVER_PATH}.parquet"):
    print("Building data from scratch...")
    DATA_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
    LOCAL_DIR = "/tmp/mhealth"
    ZIP_PATH = f"{LOCAL_DIR}/mhealth.zip"
    CN = ["chest_accel_x","chest_accel_y","chest_accel_z","ecg_lead_1","ecg_lead_2","ankle_accel_x","ankle_accel_y","ankle_accel_z","ankle_gyro_x","ankle_gyro_y","ankle_gyro_z","ankle_mag_x","ankle_mag_y","ankle_mag_z","arm_accel_x","arm_accel_y","arm_accel_z","arm_gyro_x","arm_gyro_y","arm_gyro_z","arm_mag_x","arm_mag_y","arm_mag_z","activity_label"]
    AM = {0:"null_class",1:"standing_still",2:"sitting_relaxing",3:"lying_down",4:"walking",5:"climbing_stairs",6:"waist_bends_forward",7:"frontal_elevation_arms",8:"knees_bending",9:"cycling",10:"jogging",11:"running",12:"jump_front_back"}
    os.makedirs(LOCAL_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH,"r") as z: z.extractall(LOCAL_DIR)
    frames = []
    for f in sorted(Path(LOCAL_DIR).rglob("mHealth_subject*.log")):
        sid = int(f.stem.replace("mHealth_subject",""))
        d = pd.read_csv(f, sep=r"\s+", header=None, names=CN)
        d["subject_id"]=sid; d["sample_idx"]=range(len(d)); d["timestamp_sec"]=d["sample_idx"]/SAMPLE_RATE_HZ
        frames.append(d)
    bronze = pd.concat(frames, ignore_index=True)
    bronze["activity_name"] = bronze["activity_label"].map(AM)
    for loc in ["chest","ankle","arm"]:
        bronze[f"{loc}_accel_mag"] = np.sqrt(bronze[f"{loc}_accel_x"]**2+bronze[f"{loc}_accel_y"]**2+bronze[f"{loc}_accel_z"]**2)
    def _roll(g):
        w=WINDOW_SAMPLES; feats=pd.DataFrame(index=g.index)
        for loc in ["chest","ankle","arm"]:
            m=f"{loc}_accel_mag"
            feats[f"{loc}_accel_mag_mean"]=g[m].rolling(w,min_periods=1).mean()
            feats[f"{loc}_accel_mag_std"]=g[m].rolling(w,min_periods=1).std().fillna(0)
            feats[f"{loc}_accel_mag_max"]=g[m].rolling(w,min_periods=1).max()
            feats[f"{loc}_accel_mag_range"]=g[m].rolling(w,min_periods=1).max()-g[m].rolling(w,min_periods=1).min()
        for loc in ["ankle","arm"]:
            for ax in ["x","y","z"]:
                c=f"{loc}_gyro_{ax}"
                feats[f"{c}_mean"]=g[c].rolling(w,min_periods=1).mean()
                feats[f"{c}_std"]=g[c].rolling(w,min_periods=1).std().fillna(0)
        for loc in ["ankle","arm"]:
            gm=np.sqrt(g[f"{loc}_gyro_x"]**2+g[f"{loc}_gyro_y"]**2+g[f"{loc}_gyro_z"]**2)
            feats[f"{loc}_gyro_mag_mean"]=gm.rolling(w,min_periods=1).mean()
            feats[f"{loc}_gyro_mag_std"]=gm.rolling(w,min_periods=1).std().fillna(0)
        feats["ecg_lead1_mean"]=g["ecg_lead_1"].rolling(w,min_periods=1).mean()
        feats["ecg_lead1_std"]=g["ecg_lead_1"].rolling(w,min_periods=1).std().fillna(0)
        feats["ecg_lead1_range"]=g["ecg_lead_1"].rolling(w,min_periods=1).max()-g["ecg_lead_1"].rolling(w,min_periods=1).min()
        return feats
    def _hrv(ecg,window=250):
        hrv=pd.Series(np.nan,index=ecg.index)
        for s in range(0,len(ecg)-window+1,window):
            seg=ecg.iloc[s:s+window].values; pk,_=find_peaks(seg,distance=20,height=np.percentile(seg,75))
            hrv.iloc[s:s+window]=np.std(np.diff(pk)/SAMPLE_RATE_HZ) if len(pk)>2 else 0.0
        return hrv.ffill().fillna(0)
    def _strain(r):
        a=(r.get("chest_accel_mag_mean",0)+r.get("ankle_accel_mag_mean",0)+r.get("arm_accel_mag_mean",0))/3
        g=(r.get("ankle_gyro_mag_mean",0)+r.get("arm_gyro_mag_mean",0))/2
        return a*0.6+g*0.4
    sfs=[]
    for sid,grp in bronze.groupby("subject_id"):
        grp=grp.sort_values("sample_idx").reset_index(drop=True)
        rf=_roll(grp); mg=pd.concat([grp,rf],axis=1)
        mg["hrv_proxy_sdnn"]=_hrv(grp["ecg_lead_1"])
        mg["strain_raw"]=mg.apply(_strain,axis=1)
        rest=mg["activity_label"].isin([0,1,2,3])
        mg["is_recovery_window"]=rest.astype(int)
        mg["recovery_proxy"]=np.where(rest,10.0-mg["strain_raw"].clip(0,10),0.0)
        sfs.append(mg)
    sf=pd.concat(sfs,ignore_index=True)
    lo,hi=sf["strain_raw"].quantile(0.01),sf["strain_raw"].quantile(0.99)
    sf["strain_score"]=((sf["strain_raw"]-lo)/(hi-lo)*10).clip(0,10)
    silver=sf.iloc[::WINDOW_SAMPLES].reset_index(drop=True)
    silver.to_parquet(f"{SILVER_PATH}.parquet",index=False)
    print(f"Built: {silver.shape[0]:,} rows x {silver.shape[1]} cols")
else:
    silver=pd.read_parquet(f"{SILVER_PATH}.parquet")
    print(f"Loaded: {silver.shape[0]:,} rows x {silver.shape[1]} cols")

# COMMAND ----------

# --- STRAIN BY ACTIVITY ---
fig, ax = plt.subplots(figsize=(12, 5))
activity_strain = silver[silver["activity_label"] > 0].groupby("activity_name")["strain_score"].mean().sort_values(ascending=False)
bars = ax.bar(range(len(activity_strain)), activity_strain.values, color="#3B82F6")
ax.set_xticks(range(len(activity_strain)))
ax.set_xticklabels(activity_strain.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean Strain Score (0-10)")
ax.set_title("Average Strain Score by Activity")
for i, v in enumerate(activity_strain.values):
    ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# --- HRV BY SUBJECT ---
fig, ax = plt.subplots(figsize=(10, 5))
hrv_sub = silver.groupby("subject_id")["hrv_proxy_sdnn"].mean().sort_values(ascending=False)
ax.bar(hrv_sub.index.astype(str), hrv_sub.values, color="#10B981")
ax.set_xlabel("Subject ID"); ax.set_ylabel("Mean HRV (SDNN proxy)")
ax.set_title("HRV Proxy by Subject")
for i, v in enumerate(hrv_sub.values):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.show()

# COMMAND ----------

# --- STRAIN vs HRV SCATTER ---
fig, ax = plt.subplots(figsize=(10, 6))
rec = silver[silver["is_recovery_window"]==1]; act = silver[silver["is_recovery_window"]==0]
ax.scatter(act["strain_score"], act["hrv_proxy_sdnn"], alpha=0.3, s=10, color="#EF4444", label="Active")
ax.scatter(rec["strain_score"], rec["hrv_proxy_sdnn"], alpha=0.3, s=10, color="#10B981", label="Recovery")
ax.set_xlabel("Strain Score"); ax.set_ylabel("HRV Proxy")
ax.set_title("Strain vs HRV: Active vs Recovery"); ax.legend()
plt.tight_layout(); plt.show()

# COMMAND ----------

# --- MEMBER SUMMARY ---
summary = silver.groupby("subject_id").agg(
    avg_strain=("strain_score","mean"), max_strain=("strain_score","max"),
    avg_hrv=("hrv_proxy_sdnn","mean"), recovery_pct=("is_recovery_window","mean"),
    total_windows=("strain_score","count")
).round(3).reset_index()
summary["recovery_pct"] = (summary["recovery_pct"]*100).round(1)
print("=== Member Health Summary ===")
print(summary.to_string(index=False))

# COMMAND ----------

# --- ACTIVITY CLASSIFICATION ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

FC = ["chest_accel_mag_mean","chest_accel_mag_std","chest_accel_mag_range","ankle_accel_mag_mean","ankle_accel_mag_std","ankle_accel_mag_range","arm_accel_mag_mean","arm_accel_mag_std","arm_accel_mag_range","ankle_gyro_mag_mean","ankle_gyro_mag_std","arm_gyro_mag_mean","arm_gyro_mag_std","ecg_lead1_mean","ecg_lead1_std","ecg_lead1_range","hrv_proxy_sdnn","strain_score","recovery_proxy"]
sa = silver[silver["activity_label"]>0].copy()
X=sa[FC].fillna(0); y=sa["activity_label"]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
clf = RandomForestClassifier(n_estimators=200,max_depth=20,random_state=42,n_jobs=-1)
clf.fit(Xtr,ytr); yp=clf.predict(Xte)
acc=accuracy_score(yte,yp); f1=f1_score(yte,yp,average="macro")
AM2 = {1:"standing",2:"sitting",3:"lying",4:"walking",5:"stairs",6:"waist_bend",7:"arm_elev",8:"knees_bend",9:"cycling",10:"jogging",11:"running",12:"jumping"}
labels=sorted(yte.unique()); names=[AM2.get(l,str(l)) for l in labels]
fig,ax=plt.subplots(figsize=(10,8))
cm=confusion_matrix(yte,yp,labels=labels)
ConfusionMatrixDisplay(cm,display_labels=names).plot(ax=ax,cmap="Blues",values_format="d")
ax.set_title(f"Activity Classification: accuracy={acc:.3f}, F1={f1:.3f}")
plt.xticks(rotation=45,ha="right"); plt.tight_layout(); plt.show()
print(f"Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")

# COMMAND ----------

# --- ANOMALY DETECTION A/B ---
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score as f1fn

AF=["chest_accel_mag_mean","chest_accel_mag_std","ankle_accel_mag_mean","ankle_accel_mag_std","arm_accel_mag_mean","arm_accel_mag_std","ankle_gyro_mag_mean","ankle_gyro_mag_std","arm_gyro_mag_mean","arm_gyro_mag_std","ecg_lead1_std","ecg_lead1_range","hrv_proxy_sdnn","strain_score"]
Xa=silver[AF].fillna(0); Xs=StandardScaler().fit_transform(Xa)
def _gt(df):
    fl=pd.Series(0,index=df.index)
    for l in df["activity_label"].unique():
        m=df["activity_label"]==l; s=df.loc[m,"strain_score"]
        q1,q3=s.quantile(0.05),s.quantile(0.95); iqr=q3-q1
        fl[m&((df["strain_score"]<q1-2*iqr)|(df["strain_score"]>q3+2*iqr))]=1
    return fl
gt=_gt(silver); gr=gt.mean()
i1=IsolationForest(n_estimators=200,contamination=0.05,random_state=42,n_jobs=-1)
silver["av1"]=np.where(i1.fit_predict(Xs)==-1,1,0)
tc=min(gr*1.5,0.15)
i2=IsolationForest(n_estimators=300,contamination=tc,random_state=42,n_jobs=-1)
silver["av2"]=np.where(i2.fit_predict(Xs)==-1,1,0)
res=pd.DataFrame({"Model":["v1 (default)","v2 (tuned)"],
    "Precision":[precision_score(gt,silver["av1"],zero_division=0),precision_score(gt,silver["av2"],zero_division=0)],
    "Recall":[recall_score(gt,silver["av1"],zero_division=0),recall_score(gt,silver["av2"],zero_division=0)],
    "F1":[f1fn(gt,silver["av1"],zero_division=0),f1fn(gt,silver["av2"],zero_division=0)]}).round(4)
print("=== Anomaly Detection A/B ===")
print(res.to_string(index=False))
print(f"F1 improvement: {(res.iloc[1]['F1']-res.iloc[0]['F1'])*100:.2f}pp")
fig,ax=plt.subplots(figsize=(8,4))
x=range(2); w=0.25
ax.bar([i-w for i in x],res["Precision"],w,label="Precision",color="#3B82F6")
ax.bar(x,res["Recall"],w,label="Recall",color="#10B981")
ax.bar([i+w for i in x],res["F1"],w,label="F1",color="#F59E0B")
ax.set_xticks(x); ax.set_xticklabels(res["Model"])
ax.set_ylabel("Score"); ax.set_title("Anomaly Detection: v1 vs v2"); ax.legend(); ax.set_ylim(0,1)
plt.tight_layout(); plt.show()
