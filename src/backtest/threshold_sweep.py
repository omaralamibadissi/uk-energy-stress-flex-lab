from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

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
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

def build_dataset(df: pd.DataFrame, horizon_steps: int = 12):
    data = add_time_features(df).copy()
    data["future_stress"] = data["stress_index_v2"].shift(-horizon_steps)
    data = data.dropna(subset=FEATURES + ["future_stress"]).reset_index(drop=True)
    return data

if __name__ == "__main__":
    path = pick_longest_stress_v2()
    df = pd.read_parquet(path)

    model, aucs, thr_train, used_folds = train_eval(df, q=0.90, n_splits=4)

    data = build_dataset(df, horizon_steps=12)
    thr = data["future_stress"].quantile(0.90)
    y_true = (data["future_stress"] >= thr).astype(int).values

    X = data[FEATURES]
    proba = model.predict_proba(X)[:, 1]

    thresholds = np.arange(0.40, 0.91, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        alert_rate = float(y_pred.mean())
        rows.append((t, alert_rate, precision, recall, f1))

    out = pd.DataFrame(rows, columns=["threshold", "alert_rate", "precision", "recall", "f1"])
    print("File:", path.name)
    print("AUC mean:", round(float(np.mean(aucs)), 3))
    print(out.to_string(index=False, justify="left", float_format=lambda x: f"{x:.3f}"))
