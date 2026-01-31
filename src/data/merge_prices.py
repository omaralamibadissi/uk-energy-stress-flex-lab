from __future__ import annotations

import re
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

UK = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")

def pick_longest_stress_v2() -> Path:
    pat = re.compile(r"^stress_v2_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_v2_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

def pick_longest_system_prices() -> Path:
    pat = re.compile(r"^system_prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in RAW_DIR.glob("system_prices_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))
    if not parsed:
        raise FileNotFoundError("No system_prices parquet in data/raw. Run elexon ingestion first.")
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

def settlement_to_utc(settlement_date: str, settlement_period: int) -> pd.Timestamp:
    """
    settlement_period 1..48 corresponds to half-hour blocks:
    SP=1 -> 00:00 local
    SP=2 -> 00:30 local
    ...
    SP=48 -> 23:30 local
    """
    base_local = pd.Timestamp(settlement_date).tz_localize(UK)
    minutes = (settlement_period - 1) * 30
    dt_local = base_local + pd.Timedelta(minutes=minutes)
    return dt_local.tz_convert(UTC)

def run() -> Path:
    stress_path = pick_longest_stress_v2()
    prices_path = pick_longest_system_prices()

    stress = pd.read_parquet(stress_path).copy()
    prices = pd.read_parquet(prices_path).copy()

    stress["from"] = pd.to_datetime(stress["from"], utc=True)

    prices["settlementDate"] = prices["settlementDate"].astype(str)
    prices["settlementPeriod"] = prices["settlementPeriod"].astype(int)

    prices["from"] = [
        settlement_to_utc(d, sp) for d, sp in zip(prices["settlementDate"], prices["settlementPeriod"])
    ]
    prices["from"] = pd.to_datetime(prices["from"], utc=True)

    # Pick columns (names can differ slightly by endpoint; keep what exists)
    cols = ["from"]
    for c in ["systemBuyPrice", "systemSellPrice", "sbp", "ssp"]:
        if c in prices.columns:
            cols.append(c)

    prices2 = prices[cols].drop_duplicates(subset=["from"]).sort_values("from")

    merged = stress.merge(prices2, on="from", how="left")

    out = PROCESSED_DIR / stress_path.name.replace("stress_v2_", "stress_v2_prices_")
    merged.to_parquet(out, index=False)
    print("Stress:", stress_path.name)
    print("Prices:", prices_path.name)
    print("Saved:", out.name)
    return out

if __name__ == "__main__":
    run()
