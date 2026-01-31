import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = [
    "cold_score",
    "low_wind_score",
    "stress_index",
    "forecast",
    "hour",
    "dow",
    "stress_index_lag1",
    "cold_score_lag1",
    "low_wind_score_lag1",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["from"] = pd.to_datetime(out["from"], utc=True)
    out["hour"] = out["from"].dt.hour
    out["dow"] = out["from"].dt.dayofweek

    # 1-step lag (30 minutes in your dataset)
    out["stress_index_lag1"] = out["stress_index"].shift(1)
    out["cold_score_lag1"] = out["cold_score"].shift(1)
    out["low_wind_score_lag1"] = out["low_wind_score"].shift(1)
    return out


def train_eval(df: pd.DataFrame, q: float = 0.90, n_splits: int = 4, horizon_steps: int = 12):

    """
    Train/evaluate a spike-risk classifier using time-series splits.

    Key points:
    - Threshold is computed on TRAIN only (no leakage).
    - If a fold has only one class, we skip it.
    """
    data = add_time_features(df).copy()

    # Target: will stress be in the top tail within the next 6 hours?
    data["future_stress"] = data["stress_index_v2"].shift(-horizon_steps)

    # Drop rows with missing features or target
    data = data.dropna(subset=FEATURES + ["future_stress"]).reset_index(drop=True)

    X_all = data[FEATURES].reset_index(drop=True)
    stress = data["future_stress"].reset_index(drop=True)



    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    aucs = []
    used_folds = 0

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        s_train, s_test = stress.iloc[train_idx], stress.iloc[test_idx]

        # Compute threshold on TRAIN only
        thr = s_train.quantile(q)

        y_train = (s_train >= thr).astype(int)
        y_test = (s_test >= thr).astype(int)

        # Need at least 2 classes in train
        if y_train.nunique() < 2:
            # skip this fold
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        # AUC needs both classes in test too; if not, skip metric
        if y_test.nunique() < 2:
            continue

        aucs.append(roc_auc_score(y_test, proba))
        used_folds += 1

    # Fit final model on full data with global threshold (for producing scores later)
    final_thr = stress.quantile(q)
    y_full = (stress >= final_thr).astype(int)

    # If the full set still has one class (unlikely with q=0.90), lower q automatically
    if y_full.nunique() < 2:
        final_thr = stress.quantile(0.80)
        y_full = (stress >= final_thr).astype(int)

    model.fit(X_all, y_full)

    return model, aucs, float(final_thr), used_folds
