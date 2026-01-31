from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.models.spike_risk import train_eval, FEATURES, add_time_features

PROCESSED_DIR = Path("data/processed")

def pick_longest_stress_v2() -> Path:
    pat = re.compile(r"^stress_v2_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_v2_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))
    if not parsed:
        raise FileNotFoundError("No stress_v2 file found.")
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

def build_dataset(df: pd.DataFrame, horizon_steps: int = 12):
    data = add_time_features(df).copy()
    data["future_stress"] = data["stress_index_v2"].shift(-horizon_steps)
    data = data.dropna(subset=FEATURES + ["future_stress"]).reset_index(drop=True)
    return data

def make_labels(data: pd.DataFrame, q: float):
    thr = data["future_stress"].quantile(q)
    y = (data["future_stress"] >= thr).astype(int)
    return y, float(thr)

if __name__ == "__main__":
    path = pick_longest_stress_v2()
    df = pd.read_parquet(path)

    # Train model on full data (already time-safe inside train_eval for CV)
    model, aucs, thr_train, used_folds = train_eval(df, q=0.90, n_splits=4)

    data = build_dataset(df, horizon_steps=12)
    y_true, thr = make_labels(data, q=0.90)
    X = data[FEATURES]

    proba = model.predict_proba(X)[:, 1]

    # Decision rule
    alert_threshold = 0.60
    y_pred = (proba >= alert_threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    alert_rate = y_pred.mean()
    spike_rate = y_true.mean()

    print("File:", path.name)
    print("Spike quantile q:", 0.90, "| spike threshold future_stress:", round(thr, 2))
    print("Model CV AUCs:", [round(float(a), 3) for a in aucs], "| mean:", round(float(np.mean(aucs)), 3))
    print("Decision rule: alert if P(spike) >=", alert_threshold)
    print("Spike rate:", round(float(spike_rate), 4), "| Alert rate:", round(float(alert_rate), 4))
    print("Precision:", round(float(precision), 3))
    print("Recall:", round(float(recall), 3))
    print("F1:", round(float(f1), 3))
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)
