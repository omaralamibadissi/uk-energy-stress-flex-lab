from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def latest_stress() -> Path:
    files = sorted(PROCESSED_DIR.glob("stress_*.parquet"))
    if not files:
        raise FileNotFoundError("No stress parquet found in data/processed. Run Phase 2 first.")
    return files[-1]

def latest_weather() -> Path:
    files = sorted(RAW_DIR.glob("weather_uk_proxy_*.parquet"))
    if not files:
        raise FileNotFoundError("No weather parquet found in data/raw. Run ingest_weather first.")
    return files[-1]

def run() -> Path:
    stress_path = latest_stress()
    weather_path = latest_weather()

    stress = pd.read_parquet(stress_path).copy()
    weather = pd.read_parquet(weather_path).copy()

    # Ensure datetime
    stress["from"] = pd.to_datetime(stress["from"], utc=True)
    weather["time"] = pd.to_datetime(weather["time"], utc=True)

    # Make weather 30-min frequency by forward-filling
    weather = weather.set_index("time").sort_index()
    idx_30m = pd.date_range(weather.index.min(), weather.index.max(), freq="30min", tz="UTC")
    weather_30m = weather.reindex(idx_30m).ffill().reset_index().rename(columns={"index": "from"})

    merged = stress.merge(weather_30m, on="from", how="left")

    out = PROCESSED_DIR / stress_path.name.replace("stress_", "stress_weather_")
    merged.to_parquet(out, index=False)

    print("Stress:", stress_path.name)
    print("Weather:", weather_path.name)
    print("Saved:", out.name)
    return out

if __name__ == "__main__":
    run()
