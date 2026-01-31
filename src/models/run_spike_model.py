# src/models/run_spike_model.py
from pathlib import Path
import re
import pandas as pd

from src.models.spike_risk import train_eval, FEATURES

PROCESSED_DIR = Path("data/processed")

def pick_longest_stress_v2() -> Path:
    """
    Pick stress_v2 files like:
      stress_v2_YYYY-MM-DD_YYYY-MM-DD.parquet
    Prefer:
      - latest end date
      - if tie, earliest start date (longest span)
    """
    pat = re.compile(r"^stress_v2_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_v2_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))

    if not parsed:
        raise FileNotFoundError("No stress_v2 file found in data/processed.")

    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    chosen = min(candidates, key=lambda x: x[0])[2]
    return chosen

if __name__ == "__main__":
    path = pick_longest_stress_v2()
    df = pd.read_parquet(path)

    model, aucs, thr, used_folds = train_eval(df, q=0.90, n_splits=4)

    print("File:", path.name)
    print("Spike definition: future stress_index_v2 >= train-quantile(q=0.90)")
    print("Final threshold used:", round(thr, 2))
    print("Features:", FEATURES)
    print("Used folds:", used_folds, "/ 4")
    print("AUC per fold:", [round(float(a), 3) for a in aucs])
    print("Mean AUC:", round(sum(aucs) / len(aucs), 3) if aucs else "N/A")
