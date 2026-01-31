# src/data/elexon_system_prices.py
from __future__ import annotations

from pathlib import Path
import time
import requests
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT: use the data host (Insights Solution APIs)
BASE = "https://data.elexon.co.uk/bmrs/api/v1"

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "uk-energy-stress-flex-lab/1.0"
})

def _get_json(url: str, retries: int = 5, backoff: float = 1.0) -> dict:
    """
    Robust GET -> JSON with retries.
    If response is not JSON, prints a small snippet to help debugging.
    """
    last_err = None
    for k in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            # Sometimes APIs return HTML error pages. Check early.
            ctype = (r.headers.get("Content-Type") or "").lower()

            if r.status_code >= 400:
                # show a snippet for diagnosis
                snippet = (r.text or "")[:200].replace("\n", " ")
                raise RuntimeError(f"HTTP {r.status_code} | {url} | content-type={ctype} | body[:200]={snippet}")

            if "application/json" not in ctype:
                snippet = (r.text or "")[:200].replace("\n", " ")
                raise RuntimeError(f"Non-JSON response | {url} | content-type={ctype} | body[:200]={snippet}")

            return r.json()

        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** k))

    raise RuntimeError(f"Failed after {retries} retries. Last error: {last_err}")

def fetch_system_prices_day(settlement_date: str) -> pd.DataFrame:
    """
    Fetch ALL system prices for a given settlement date (YYYY-MM-DD) in one request:
      GET /balancing/settlement/system-prices/{settlementDate}
    """
    url = f"{BASE}/balancing/settlement/system-prices/{settlement_date}"
    js = _get_json(url)

    # Typical shape: {"data":[...]}
    data = js.get("data", js)
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Ensure these exist for later merge
    if "settlementDate" not in df.columns:
        df["settlementDate"] = settlement_date

    # Some APIs use "settlementPeriod", some use "settlementPeriodId" etc.
    # Keep it flexible but ensure settlementPeriod is present if possible.
    if "settlementPeriod" not in df.columns:
        for alt in ["settlementPeriodId", "settlementperiod", "period"]:
            if alt in df.columns:
                df["settlementPeriod"] = df[alt]
                break

    return df

def run(start_date: str, end_date: str) -> Path:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = pd.date_range(start, end, freq="D")

    parts = []
    for i, d in enumerate(days, 1):
        ds = d.strftime("%Y-%m-%d")
        print(f"Fetching {i}/{len(days)}: {ds}")
        df = fetch_system_prices_day(ds)

        # small pause to be polite
        time.sleep(0.15)

        if df is not None and not df.empty:
            parts.append(df)

    if not parts:
        raise RuntimeError("No system price data downloaded (all days empty).")

    out_df = pd.concat(parts, ignore_index=True)

    out = RAW_DIR / f"system_prices_{start_date}_{end_date}.parquet"
    out_df.to_parquet(out, index=False)
    print("Saved:", out)
    return out

if __name__ == "__main__":
    run("2025-10-01", "2026-01-15")
