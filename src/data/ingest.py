from pathlib import Path
import time
import pandas as pd

from src.data.carbon_intensity import fetch_carbon_intensity

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _iso(dt: pd.Timestamp) -> str:
    # Carbon Intensity expects ISO with Z
    return dt.strftime("%Y-%m-%dT%H:%MZ")

def run_chunked(start_iso: str, end_iso: str, chunk_days: int = 7) -> Path:
    """
    Download carbon intensity over a long period by chunking the request.
    """
    start = pd.to_datetime(start_iso, utc=True)
    end = pd.to_datetime(end_iso, utc=True)

    # Build chunk boundaries
    boundaries = pd.date_range(start=start, end=end, freq=f"{chunk_days}D", tz="UTC").tolist()
    if boundaries[-1] != end:
        boundaries.append(end)

    parts = []
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        a_iso = _iso(a)
        b_iso = _iso(b)

        print(f"Fetching chunk {i+1}/{len(boundaries)-1}: {a_iso} -> {b_iso}")
        df = fetch_carbon_intensity(a_iso, b_iso)
        parts.append(df)

        # Small sleep to be polite with API
        time.sleep(0.2)

    out_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["from", "to"]).sort_values("from")

    out = RAW_DIR / f"carbon_intensity_{start_iso[:10]}_{end_iso[:10]}.parquet"
    out_df.to_parquet(out, index=False)
    return out

if __name__ == "__main__":
    path = run_chunked("2025-10-01T00:00Z", "2026-01-15T00:00Z", chunk_days=7)
    print("Saved:", path)
