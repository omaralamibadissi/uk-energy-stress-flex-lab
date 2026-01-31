from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.features.stress_index_v2 import compute_stress_v2
from src.models.regimes import label_regime

PROCESSED_DIR = Path("data/processed")

def latest_stress_weather() -> Path:
    files = sorted(PROCESSED_DIR.glob("stress_weather_*.parquet"))
    if not files:
        raise FileNotFoundError("No stress_weather parquet found. Run merge_weather first.")
    return files[-1]

def run() -> Path:
    path = latest_stress_weather()
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

    print("Saved:", out.name)
    print("Plot:", plot_path.name)
    return out

if __name__ == "__main__":
    run()
