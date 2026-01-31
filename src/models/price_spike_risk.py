from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.models.spike_risk import add_time_features  # reuse

FEATURES_PRICE = [
    # physical drivers + proxy
    "cold_score",
    "low_wind_score",
    "stress_index",
    "forecast",
    # calendar
    "hour",
    "dow",
    # short memory
    "stress_index_lag1",
    "cold_score_lag1",
    "low_wind_score_lag1",
]

def pick_price_column(df: pd.DataFrame) -> str:
    for c in ["systemSellPrice", "ssp", "systemBuyPrice", "sbp"]:
        if c in df.columns:
            return c
    raise ValueError("No known price column found (expected systemSellPrice/ssp/systemBuyPrice/sbp).")

def train_eval_price(df: pd.DataFrame, q: float = 0.90, n_splits: int = 4, horizon_steps: int = 12):
    data = add_time_features(df).copy()
    price_col = pick_price_column(data)

    data["future_price"] = data[price_col].shift(-horizon_steps)
    data = data.dropna(subset=FEATURES_PRICE + ["future_price"]).reset_index(drop=True)

    X_all = data[FEATURES_PRICE]
    s_all = data["future_price"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    aucs = []
    used = 0

    for tr, te in tscv.split(X_all):
        X_tr, X_te = X_all.iloc[tr], X_all.iloc[te]
        s_tr, s_te = s_all.iloc[tr], s_all.iloc[te]

        thr = s_tr.quantile(q)
        y_tr = (s_tr >= thr).astype(int)
        y_te = (s_te >= thr).astype(int)

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, proba))
        used += 1

    final_thr = float(s_all.quantile(q))
    y_full = (s_all >= final_thr).astype(int)
    if y_full.nunique() < 2:
        final_thr = float(s_all.quantile(0.85))
        y_full = (s_all >= final_thr).astype(int)

    model.fit(X_all, y_full)
    return model, aucs, final_thr, used, price_col, FEATURES_PRICE
