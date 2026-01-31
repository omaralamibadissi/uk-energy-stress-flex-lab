import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.models.spike_risk import FEATURES, add_time_features, train_eval
from src.models.price_spike_risk import train_eval_price, FEATURES_PRICE, pick_price_column

PROCESSED_DIR = Path("data/processed")

def pick_longest(pattern: str, glob_pat: str) -> Path:
    pat = re.compile(pattern)
    parsed = []
    for p in PROCESSED_DIR.glob(glob_pat):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))
    if not parsed:
        raise FileNotFoundError(f"No matching files for {glob_pat} in data/processed")
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

def pick_longest_stress_v2() -> Path:
    return pick_longest(r"^stress_v2_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$", "stress_v2_*.parquet")

def pick_longest_stress_v2_prices() -> Path:
    return pick_longest(r"^stress_v2_prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$", "stress_v2_prices_*.parquet")

def format_time_axis():
    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.xticks(rotation=45)



def build_dataset_stress(df: pd.DataFrame, horizon_steps: int):
    data = add_time_features(df).copy()
    data["future_stress"] = data["stress_index_v2"].shift(-horizon_steps)
    data = data.dropna(subset=FEATURES + ["future_stress"]).reset_index(drop=True)
    return data

def build_dataset_price(df: pd.DataFrame, horizon_steps: int):
    data = add_time_features(df).copy()
    price_col = pick_price_column(data)
    data["future_price"] = data[price_col].shift(-horizon_steps)
    data = data.dropna(subset=FEATURES_PRICE + ["future_price"]).reset_index(drop=True)
    return data, price_col

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_resource(show_spinner=False)
def fit_stress_model(df: pd.DataFrame, q: float, horizon_steps: int):
    return train_eval(df, q=q, n_splits=4, horizon_steps=horizon_steps)

@st.cache_resource(show_spinner=False)
def fit_price_model(df: pd.DataFrame, q: float, horizon_steps: int):
    return train_eval_price(df, q=q, n_splits=4, horizon_steps=horizon_steps)

st.set_page_config(page_title="UK Spike Monitor", layout="wide")
st.title("UK Spike Monitor (Fundamentals → Market)")

st.write(
    "Two modes:\n"
    "- **Stress spike (proxy)**: carbon intensity + weather → predicts future stress regime\n"
    "- **Price spike (market)**: same fundamentals → predicts future UK system price spike\n"
)

mode = st.selectbox("Mode", ["Stress spike (proxy)", "Price spike (market)"], index=1)

col1, col2, col3, col4 = st.columns(4)
with col1:
    # default thresholds: stress 0.75, price 0.75
    alert_default = 0.75
    alert_threshold = st.slider("Alert threshold (P(spike) >=)", 0.40, 0.90, float(alert_default), 0.01)
with col2:
    q = st.slider("Spike definition quantile q", 0.80, 0.95, 0.90, 0.01)
with col3:
    horizon_hours = st.selectbox(
    "Forecast horizon",
    [3, 6, 12],
    index=1,
    format_func=lambda x: f"{x} hours"
)

with col4:
    retrain = st.button("Retrain model")

horizon_steps = int(horizon_hours * 2)  # 30-min steps

if mode.startswith("Stress"):
    path = pick_longest_stress_v2()
    df = load_data(str(path))

    if retrain:
        fit_stress_model.clear()

    model, aucs, thr_train, used_folds = fit_stress_model(df, q=q, horizon_steps=horizon_steps)

    data = build_dataset_stress(df, horizon_steps=horizon_steps)
    thr_spike = float(data["future_stress"].quantile(q))

    X = data[FEATURES]
    data["spike_proba"] = model.predict_proba(X)[:, 1]

    latest = data.iloc[-1]
    is_alert = float(latest["spike_proba"]) >= float(alert_threshold)

    st.caption(f"File: {path.name} | Used folds: {used_folds}/4 | Mean AUC: {float(np.mean(aucs)):.3f}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest stress_index_v2", f"{latest['stress_index_v2']:.2f}")
    k2.metric(f"P(stress spike in {horizon_hours}h)", f"{latest['spike_proba']:.3f}")
    k3.metric("Spike threshold (future_stress)", f"{thr_spike:.2f}")
    k4.metric("ALERT", "YES" if is_alert else "NO")

    left, right = st.columns(2)

    with left:
        st.subheader("Stress index v2")
        fig = plt.figure()
        plt.plot(data["from"], data["stress_index_v2"])
        plt.axhline(thr_spike, linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("stress_index_v2")
        format_time_axis()
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.subheader("Spike probability (stress)")
        fig2 = plt.figure()
        plt.plot(data["from"], data["spike_proba"])
        plt.axhline(alert_threshold, linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("P(spike)")
        format_time_axis()
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Last 20 rows")
    st.dataframe(data[["from", "stress_index_v2", "cold_score", "low_wind_score", "spike_proba"]].tail(20))

else:
    path = pick_longest_stress_v2_prices()
    df = load_data(str(path))

    if retrain:
        fit_price_model.clear()

    model, aucs, thr_train, used_folds, price_col, feats = fit_price_model(df, q=q, horizon_steps=horizon_steps)

    data, price_col = build_dataset_price(df, horizon_steps=horizon_steps)
    thr_spike = float(data["future_price"].quantile(q))

    X = data[FEATURES_PRICE]
    data["spike_proba"] = model.predict_proba(X)[:, 1]

    latest = data.iloc[-1]
    is_alert = float(latest["spike_proba"]) >= float(alert_threshold)

    st.caption(
        f"File: {path.name} | Price: {price_col} | Used folds: {used_folds}/4 | Mean AUC: {float(np.mean(aucs)):.3f}"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Latest {price_col}", f"{latest[price_col]:.2f}")
    k2.metric(f"P(price spike in {horizon_hours}h)", f"{latest['spike_proba']:.3f}")
    k3.metric("Spike threshold (future_price)", f"{thr_spike:.2f}")
    k4.metric("ALERT", "YES" if is_alert else "NO")

    left, right = st.columns(2)

    with left:
        st.subheader(f"{price_col} (market)")
        fig = plt.figure()
        plt.plot(data["from"], data[price_col])
        plt.axhline(thr_spike, linestyle="--")
        plt.xlabel("Time")
        format_time_axis()
        plt.ylabel(price_col)
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.subheader("Spike probability (price)")
        fig2 = plt.figure()
        plt.plot(data["from"], data["spike_proba"])
        plt.axhline(alert_threshold, linestyle="--")
        plt.xlabel("Time")
        format_time_axis()
        plt.ylabel("P(spike)")
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Last 20 rows")
    cols = ["from", price_col, "stress_index_v2", "cold_score", "low_wind_score", "spike_proba"]
    cols = [c for c in cols if c in data.columns]
    st.dataframe(data[cols].tail(20))
