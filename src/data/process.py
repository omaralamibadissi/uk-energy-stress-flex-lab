# src/data/process.py
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

from src.features.stress_index import compute_stress_index
from src.models.regimes import label_regime

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def pick_longest_carbon_intensity_file() -> Path:
    """
    Pick the carbon_intensity file with the LONGEST time span.
    Example filenames:
      carbon_intensity_2026-01-01_2026-01-15.parquet
      carbon_intensity_2025-10-01_2026-01-15.parquet

    Rule:
    - Prefer the latest end date.
    - If same end date, choose the earliest start date (longest span).
    """
    files = list(RAW_DIR.glob("carbon_intensity_*.parquet"))
    if not files:
        raise FileNotFoundError("No carbon_intensity parquet found in data/raw. Run ingestion first.")

    pat = re.compile(r"carbon_intensity_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")

    parsed = []
    for f in files:
        m = pat.search(f.name)
        if not m:
            continue
        start_str, end_str = m.group(1), m.group(2)
        parsed.append((start_str, end_str, f))

    if not parsed:
        raise FileNotFoundError("No carbon_intensity files matched expected naming pattern.")

    # 1) choose latest end date
    max_end = max(end_str for _, end_str, _ in parsed)
    candidates = [(s, e, f) for (s, e, f) in parsed if e == max_end]

    # 2) among those, choose earliest start date (longest range)
    chosen = min(candidates, key=lambda x: x[0])[2]
    return chosen

def run() -> Path:
    raw_path = pick_longest_carbon_intensity_file()
    df = pd.read_parquet(raw_path)

    df2 = compute_stress_index(df)
    df2["regime"] = label_regime(df2["stress_index"])

    out_path = PROCESSED_DIR / raw_path.name.replace("carbon_intensity_", "stress_")
    df2.to_parquet(out_path, index=False)

    # Plot
    plt.figure()
    plt.plot(df2["from"], df2["stress_index"])
    plt.title("UK Energy Stress Index (proxy)")
    plt.xlabel("Time")
    plt.ylabel("Stress (0-100)")
    plt.tight_layout()

    plot_path = PROCESSED_DIR / "stress_index.png"
    plt.savefig(plot_path, dpi=160)

    print("Raw:", raw_path.name)
    print("Saved:", out_path.name)
    print("Plot:", plot_path.name)
    return out_path

if __name__ == "__main__":
    run()
