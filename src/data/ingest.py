from pathlib import Path
import pandas as pd
from src.data.carbon_intensity import fetch_carbon_intensity

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def run(start_iso: str, end_iso: str) -> Path:
    df = fetch_carbon_intensity(start_iso, end_iso)
    out = RAW_DIR / f"carbon_intensity_{start_iso[:10]}_{end_iso[:10]}.parquet"
    df.to_parquet(out, index=False)
    return out

if __name__ == "__main__":
    path = run("2026-01-01T00:00Z", "2026-01-15T00:00Z")
    print("Saved:", path)
