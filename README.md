# UK Energy Spike Monitor (Proxy + Weather)

A compact, end-to-end project inspired by UK power & gas trading workflows: build an interpretable **UK system stress proxy** from public data (carbon intensity + weather), then train a **spike-risk model** to estimate the probability of an extreme regime within a chosen horizon (3h/6h/12h). A Streamlit dashboard turns it into a practical **early-warning monitor**.


---

## What I built

### Data sources (public)
- **UK Carbon Intensity API**: half-hourly national carbon intensity (proxy for generation mix / system tightness).
- **Open-Meteo Archive API**: hourly temperature and wind (UK proxy from London + Manchester + Edinburgh).

### Pipeline outputs
- `stress_index` (v1): 0–100 proxy from carbon intensity + normalization
- `stress_index_v2` (v2): v1 + physical drivers
  - `cold_score` (0–100): colder → higher stress
  - `low_wind_score` (0–100): lower wind → higher stress
- `spike risk model`: predicts **P(spike in H hours)** where spike is defined as top quantile of **future stress** (default `q=0.90`)
- `monitor rule`: converts probability into **ALERT / OK** using a threshold (recommended `0.75`)

---

## Repository structure
```
.
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── src/
    ├── backtest/
    │   ├── monitor_rule.py
    │   └── threshold_sweep.py
    ├── data/
    │   ├── carbon_intensity.py
    │   ├── ingest.py
    │   ├── ingest_weather.py
    │   ├── merge_weather.py
    │   ├── process.py
    │   ├── process_v2.py
    │   └── weather_openmeteo.py
    ├── features/
    │   ├── stress_index.py
    │   └── stress_index_v2.py
    └── models/
        ├── regimes.py
        ├── run_spike_model.py
        └── spike_risk.py
```

---

## Setup

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> If you update dependencies, regenerate the file:
```bash
pip freeze > requirements.txt
```

---

## Run the full pipeline (long range)

This downloads a long date range by chunking requests (Carbon Intensity API limits).

### 1) Ingest carbon intensity (chunked)
```bash
python -m src.data.ingest
```

Outputs to `data/raw/`:

* `carbon_intensity_YYYY-MM-DD_YYYY-MM-DD.parquet`

### 2) Ingest UK weather proxy
```bash
python -m src.data.ingest_weather
```

Outputs to `data/raw/`:

* `weather_uk_proxy_YYYY-MM-DD_YYYY-MM-DD.parquet`

### 3) Build Stress Index v1 (proxy)
```bash
python -m src.data.process
```

Outputs to `data/processed/`:

* `stress_YYYY-MM-DD_YYYY-MM-DD.parquet`
* `stress_index.png`

### 4) Merge weather into stress dataset
```bash
python -m src.data.merge_weather
```

Outputs:

* `stress_weather_YYYY-MM-DD_YYYY-MM-DD.parquet`

### 5) Build Stress Index v2 (+ weather drivers)
```bash
python -m src.data.process_v2
```

Outputs:

* `stress_v2_YYYY-MM-DD_YYYY-MM-DD.parquet`
* `stress_v2.png`

---

## Model & evaluation

### Spike-risk model (AUC)
```bash
python -m src.models.run_spike_model
```

What it does:

* Defines spike as **future stress** in the top quantile (default `q=0.90`)
* Predicts spike probability using:

  * weather drivers (`cold_score`, `low_wind_score`)
  * proxy system tightness (`stress_index`, `forecast`)
  * time features (`hour`, `dow`)
  * lag features (`*_lag1`)
* Uses **TimeSeriesSplit** to avoid look-ahead bias

### Monitoring rule (precision/recall tradeoff)
```bash
python -m src.backtest.monitor_rule
```

This applies an alert decision:

* `ALERT = 1` if `P(spike) >= alert_threshold`

It prints:

* spike rate, alert rate
* precision / recall / F1
* confusion matrix

### Threshold sweep (choose alert threshold)
```bash
python -m src.backtest.threshold_sweep
```

This prints a table for thresholds `[0.40 ... 0.90]` showing:

* alert_rate, precision, recall, F1

**Recommended threshold (balanced): `0.75`**
It yields fewer alerts with ~50% precision while keeping recall relatively high.

---

## Dashboard (Streamlit)

Run:
```bash
streamlit run app.py
```

The dashboard:

* displays `stress_index_v2` and predicted `P(spike)` over time
* lets you choose:

  * spike definition quantile (`q`)
  * forecast horizon (3h / 6h / 12h)
  * alert threshold
* shows a live **ALERT/OK** badge for the latest point

---

## How to interpret the signals (simple)

* **Cold score ↑** → more heating demand → system tightness risk ↑
* **Low wind score ↑** → less wind generation → reliance on gas/thermal ↑ → tightness risk ↑
* **Stress v2 ↑** → combined indicator that the system may be entering a constrained regime
* **P(spike in H hours)** → model's estimate of extreme regime probability within the horizon

---

## Notes on methodology (why it's credible)

* **No look-ahead bias**: time-series splits + target computed from **future stress** only
* **Interpretable features**: temperature/wind are physical drivers; lags reflect inertia
* **Actionable output**: probability → threshold → alert → risk decision

---


## License

MIT 
