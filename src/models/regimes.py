import pandas as pd

def label_regime(stress_index: pd.Series) -> pd.Series:
    """
    Convert stress index into 3 regimes.
    """
    return pd.cut(
        stress_index,
        bins=[-1, 35, 65, 101],
        labels=["calm", "normal", "stressed"],
    )
