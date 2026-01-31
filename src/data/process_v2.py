# src/data/process_v2.py
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

from src.features.stress_index_v2 import compute_stress_v2
from src.models.regimes import label_regime

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def pick_longest_base_stress_weather() -> Path:
    """
    Pick base stress_weather files like:
      stress_weather_YYYY-MM-DD_YYYY-MM-DD.parquet
    Prefer:
      - latest end date
      - if tie, earliest start date (longest span)
    """
    pat = re.compile(r"^stress_weather_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_weather_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))

    if not parsed:
        raise FileNotFoundError("No base stress_weather parquet found. Run: python -m src.data.merge_weather")

    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    chosen = min(candidates, key=lambda x: x[0])[2]  # earliest start
    return chosen

def run() -> Path:
    path = pick_longest_base_stress_weather()
    df = pd.read_parquet(path)

    df2 = compute_stress_v2(df)
    df2["regime_v2"] = label_regime(df2["stress_index_v2"])

    out = PROCESSED_DIR / path.name.replace("stress_weather_", "stress_v2_")
    df2.to_parquet(out, index=False)

    plt.figure()
    plt.plot(df2["from"], df2["stress_index"], label="v1 proxy")
    plt.plot(df2["from"], df2["stress_index_v2"], label="v2 + weather")
    plt.title("UK Energy Stress Index: v1 vs v2")
    plt.xlabel("Time")
    plt.ylabel("Stress (0-100)")
    plt.legend()
    plt.tight_layout()

    plot_path = PROCESSED_DIR / "stress_v2.png"
    plt.savefig(plot_path, dpi=160)

    print("Input:", path.name)
    print("Saved:", out.name)
    print("Plot:", plot_path.name)
    return out

if __name__ == "__main__":
    run()
