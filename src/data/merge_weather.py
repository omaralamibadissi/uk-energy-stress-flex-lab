# src/data/merge_weather.py
from pathlib import Path
import re
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def pick_longest_base_stress() -> Path:
    """
    Pick base stress files like:
      stress_YYYY-MM-DD_YYYY-MM-DD.parquet
    Prefer:
      - latest end date
      - if tie, earliest start date (longest span)
    """
    pat = re.compile(r"^stress_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))

    if not parsed:
        raise FileNotFoundError("No base stress parquet found. Run: python -m src.data.process")

    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    chosen = min(candidates, key=lambda x: x[0])[2]  # earliest start
    return chosen

def pick_longest_weather() -> Path:
    """
    Pick weather files like:
      weather_uk_proxy_YYYY-MM-DD_YYYY-MM-DD.parquet
    Prefer:
      - latest end date
      - if tie, earliest start date (longest span)
    """
    pat = re.compile(r"^weather_uk_proxy_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in RAW_DIR.glob("weather_uk_proxy_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))

    if not parsed:
        raise FileNotFoundError("No weather parquet found. Run: python -m src.data.ingest_weather")

    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    chosen = min(candidates, key=lambda x: x[0])[2]  # earliest start
    return chosen

def run() -> Path:
    stress_path = pick_longest_base_stress()
    weather_path = pick_longest_weather()

    stress = pd.read_parquet(stress_path).copy()
    weather = pd.read_parquet(weather_path).copy()

    # Ensure datetimes
    stress["from"] = pd.to_datetime(stress["from"], utc=True)
    weather["time"] = pd.to_datetime(weather["time"], utc=True)

    # Weather is hourly; stress is ~30min. Upsample weather to 30min via forward-fill.
    weather = weather.set_index("time").sort_index()
    idx_30m = pd.date_range(weather.index.min(), weather.index.max(), freq="30min", tz="UTC")
    weather_30m = (
        weather.reindex(idx_30m)
        .ffill()
        .reset_index()
        .rename(columns={"index": "from"})
    )

    merged = stress.merge(weather_30m, on="from", how="left")

    out = PROCESSED_DIR / stress_path.name.replace("stress_", "stress_weather_")
    merged.to_parquet(out, index=False)

    print("Stress:", stress_path.name)
    print("Weather:", weather_path.name)
    print("Saved:", out.name)
    return out

if __name__ == "__main__":
    run()
