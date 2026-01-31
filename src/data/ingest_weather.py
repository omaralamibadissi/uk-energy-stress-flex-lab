from pathlib import Path
from src.data.weather_openmeteo import fetch_uk_proxy_hourly

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def run(start_date: str, end_date: str) -> Path:
    df = fetch_uk_proxy_hourly(start_date, end_date)
    out = RAW_DIR / f"weather_uk_proxy_{start_date}_{end_date}.parquet"
    df.to_parquet(out, index=False)
    return out

if __name__ == "__main__":
    path = run("2026-01-01", "2026-01-15")
    print("Saved:", path)
