from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.features.stress_index import compute_stress_index
from src.models.regimes import label_regime

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def latest_raw_file() -> Path:
    files = sorted(RAW_DIR.glob("carbon_intensity_*.parquet"))
    if not files:
        raise FileNotFoundError("No raw parquet found in data/raw. Run ingestion first.")
    return files[-1]

def run() -> Path:
    raw_path = latest_raw_file()
    df = pd.read_parquet(raw_path)

    df2 = compute_stress_index(df)
    df2["regime"] = label_regime(df2["stress_index"])

    out_path = PROCESSED_DIR / raw_path.name.replace("carbon_intensity_", "stress_")
    df2.to_parquet(out_path, index=False)

    plt.figure()
    plt.plot(df2["from"], df2["stress_index"])
    plt.title("UK Energy Stress Index (proxy)")
    plt.xlabel("Time")
    plt.ylabel("Stress (0-100)")
    plt.tight_layout()

    plot_path = PROCESSED_DIR / "stress_index.png"
    plt.savefig(plot_path, dpi=160)

    print("Raw:", raw_path)
    print("Saved:", out_path)
    print("Plot:", plot_path)
    return out_path

if __name__ == "__main__":
    run()
