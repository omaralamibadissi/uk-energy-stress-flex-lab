import pandas as pd

INDEX_MAP = {
    "low": 10,
    "moderate": 40,
    "high": 70,
    "very high": 90,
}

def compute_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an interpretable 0-100 stress index from carbon intensity signals.
    """
    out = df.copy()

    # Category-based score (interpretable)
    out["stress_cat"] = out["index"].astype(str).str.lower().map(INDEX_MAP).fillna(50)

    # Value-based score (robust normalization using quantiles)
    q05 = out["forecast"].quantile(0.05)
    q95 = out["forecast"].quantile(0.95)
    denom = (q95 - q05) if (q95 - q05) != 0 else 1.0
    out["stress_val"] = ((out["forecast"] - q05) / denom).clip(0, 1) * 100

    # Weighted blend
    out["stress_index"] = (0.6 * out["stress_val"] + 0.4 * out["stress_cat"]).round(2)
    return out
