from pathlib import Path
import re
import pandas as pd
import numpy as np

from src.models.price_spike_risk import train_eval_price

PROCESSED_DIR = Path("data/processed")

def pick_longest_stress_v2_prices() -> Path:
    pat = re.compile(r"^stress_v2_prices_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
    parsed = []
    for p in PROCESSED_DIR.glob("stress_v2_prices_*.parquet"):
        m = pat.match(p.name)
        if m:
            start, end = m.group(1), m.group(2)
            parsed.append((start, end, p))
    if not parsed:
        raise FileNotFoundError("No stress_v2_prices file found. Run merge_prices first.")
    max_end = max(end for _, end, _ in parsed)
    candidates = [x for x in parsed if x[1] == max_end]
    return min(candidates, key=lambda x: x[0])[2]

if __name__ == "__main__":
    path = pick_longest_stress_v2_prices()
    df = pd.read_parquet(path)

    model, aucs, thr, used, price_col, feats = train_eval_price(df, q=0.90, n_splits=4, horizon_steps=12)

    print("File:", path.name)
    print("Target: future price spike in 6h (q=0.90)")
    print("Price column:", price_col)
    print("Final threshold used:", round(thr, 2))
    print("Features:", feats)
    print("Used folds:", used, "/ 4")
    print("AUC per fold:", [round(float(a), 3) for a in aucs])
    print("Mean AUC:", round(float(np.mean(aucs)), 3) if aucs else "N/A")
