import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = r"E:\Download\SAML-D.xlsx"
OUT_DIR = r"E:\Download\Avinash\anomaly_results_5_5"
RANDOM_STATE = 42

# If you want pop-up windows in addition to saving PNGs
SHOW_PLOTS = True

# Isolation Forest params
# contamination: expected anomaly rate; set to your dataset's laundering ratio if known
CONTAMINATION = "auto"   # or e.g. 0.002 if laundering is ~0.2%

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# -----------------------------
# FEATURE ENGINEERING (similar to your XGBoost setup)
# -----------------------------
# Time -> seconds
df["Time_seconds"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour * 3600 + \
                     pd.to_datetime(df["Time"], format="%H:%M:%S").dt.minute * 60 + \
                     pd.to_datetime(df["Time"], format="%H:%M:%S").dt.second

# Date features
if not np.issubdtype(df["Date"].dtype, np.datetime64):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df["Day_of_week"] = df["Date"].dt.dayofweek
df["Day_of_month"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Encode categorical variables
categorical_columns = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Create account IDs (optional but matches your earlier pipeline)
all_accounts = pd.concat([df["Sender_account"], df["Receiver_account"]]).unique()
account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
df["Sender_id"] = df["Sender_account"].map(account_to_id)
df["Receiver_id"] = df["Receiver_account"].map(account_to_id)

# Choose features for anomaly detection
# (Keep label out of features)
ad_features = [
    "Amount",
    "Time_seconds",
    "Day_of_week",
    "Day_of_month",
    "Month",
    "Year",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Sender_id",
    "Receiver_id",
]

X = df[ad_features].copy()
y = df["Is_laundering"].astype(int).copy()

# Scale numeric features (recommended for distance/partition-based methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# FIT ANOMALY DETECTOR
# -----------------------------
iso = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

iso.fit(X_train)

# IsolationForest.decision_function: higher = more normal
# Convert to anomaly score where higher = more anomalous
# anomaly_score = -decision_function
test_normality = iso.decision_function(X_test)
anomaly_score = -test_normality  # higher => more anomalous

# -----------------------------
# OUTPUT DIR
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# (1) Distribution plot: anomaly scores by class
# -----------------------------
scores_normal = anomaly_score[y_test.values == 0]
scores_laund = anomaly_score[y_test.values == 1]

plt.figure(figsize=(8, 5))

# Histogram (density) — avoids extra dependencies
bins = 60
plt.hist(scores_normal, bins=bins, density=True, alpha=0.6, label="Normal (0)")
plt.hist(scores_laund, bins=bins, density=True, alpha=0.6, label="Laundering (1)")

plt.title("Anomaly Score Distribution (Isolation Forest)")
plt.xlabel("Anomaly score (higher = more anomalous)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()

dist_path = os.path.join(OUT_DIR, "anomaly_score_distribution.png")
plt.savefig(dist_path, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close()

# -----------------------------
# (2) Precision–Recall curve
# -----------------------------
precision, recall, thresholds = precision_recall_curve(y_test, anomaly_score)
ap = average_precision_score(y_test, anomaly_score)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, linewidth=2)
plt.title(f"Precision–Recall Curve (Anomaly Detection) | AP = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True, alpha=0.3)
plt.tight_layout()

pr_path = os.path.join(OUT_DIR, "anomaly_precision_recall_curve.png")
plt.savefig(pr_path, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close()

# -----------------------------
# Optional: Threshold vs Precision/Recall/F1
# -----------------------------
# precision_recall_curve returns:
# - precision, recall arrays of length (len(thresholds)+1)
# - thresholds length = len(precision)-1
prec_t = precision[:-1]
rec_t = recall[:-1]
thr = thresholds

f1 = (2 * prec_t * rec_t) / (prec_t + rec_t + 1e-12)

plt.figure(figsize=(8, 5))
plt.plot(thr, prec_t, label="Precision")
plt.plot(thr, rec_t, label="Recall")
plt.plot(thr, f1, label="F1", linewidth=2)
plt.title("Threshold vs Precision / Recall / F1 (Anomaly Detection)")
plt.xlabel("Anomaly score threshold (predict laundering if score >= threshold)")
plt.ylabel("Metric value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

thr_path = os.path.join(OUT_DIR, "anomaly_threshold_vs_metrics.png")
plt.savefig(thr_path, dpi=300, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close()

# -----------------------------
# OPTIONAL: pick a threshold (max F1) and print report
# -----------------------------
best_idx = int(np.nanargmax(f1))
best_thr = float(thr[best_idx])

y_pred = (anomaly_score >= best_thr).astype(int)

print("\n=== Anomaly Detection (Isolation Forest) ===")
print(f"Average Precision (PR-AUC): {ap:.4f}")
print(f"Best F1 threshold: {best_thr:.6f}")
print(classification_report(y_test, y_pred, digits=4))

print("\nSaved figures to:")
print(" -", dist_path)
print(" -", pr_path)
print(" -", thr_path)