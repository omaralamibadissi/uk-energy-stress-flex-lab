import pandas as pd

def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    denom = (hi - lo) if (hi - lo) != 0 else 1.0
    return ((s - lo) / denom).clip(0, 1)

def compute_stress_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a more 'physical' stress index by combining:
    - existing stress_index (proxy)
    - cold score (colder => higher stress)
    - low wind score (lower wind => higher stress)
    """
    out = df.copy()

    # cold: use negative temperature so that colder -> bigger
    cold_norm = _minmax(-out["temp_c_uk"])
    out["cold_score"] = (cold_norm * 100).round(2)

    # low wind: use negative wind so that low wind -> bigger
    lowwind_norm = _minmax(-out["wind_ms_uk"])
    out["low_wind_score"] = (lowwind_norm * 100).round(2)

    # Combine (weights can be tuned later)
    out["stress_index_v2"] = (
        0.5 * out["stress_index"] +
        0.3 * out["cold_score"] +
        0.2 * out["low_wind_score"]
    ).round(2)

    return out
