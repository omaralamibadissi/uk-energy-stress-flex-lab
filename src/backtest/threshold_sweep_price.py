from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.price_spike_risk import train_eval_price, FEATURES_PRICE, pick_price_column
from src.models.spike_risk import add_time_features

PROCESSED_DIR = Path("data/processed")

def pick_longest_stress_v2_prices() -> Path:
    pat = re.compile(r"^stress_v2_prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_v2_prices_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

def build_dataset(df: pd.DataFrame, horizon_steps: int = 12):
    data = add_time_features(df).copy()
    price_col = pick_price_column(data)
    data["future_price"] = data[price_col].shift(-horizon_steps)
    data = data.dropna(subset=FEATURES_PRICE + ["future_price"]).reset_index(drop=True)
    return data, price_col

if __name__ == "__main__":
    q = 0.90
    horizon_steps = 12

    path = pick_longest_stress_v2_prices()
    df = pd.read_parquet(path)

    model, aucs, thr_train, used_folds, price_col, feats = train_eval_price(
        df, q=q, n_splits=4, horizon_steps=horizon_steps
    )

    data, price_col = build_dataset(df, horizon_steps=horizon_steps)
    thr = float(data["future_price"].quantile(q))
    y_true = (data["future_price"] >= thr).astype(int).values

    X = data[FEATURES_PRICE]
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
    print(f"Price column: {price_col} | spike q={q} | price spike threshold={thr:.2f}")
    print("AUC mean:", round(float(np.mean(aucs)), 3))
    print(out.to_string(index=False, justify="left", float_format=lambda x: f"{x:.3f}"))
