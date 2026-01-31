import requests
import pandas as pd

CITIES = {
    "london": (51.5072, -0.1276),
    "manchester": (53.4808, -2.2426),
    "edinburgh": (55.9533, -3.1883),
}

def _fetch_city_hourly(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    hourly = j["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"], utc=True),
        "temp_c": hourly["temperature_2m"],
        "wind_ms": hourly["wind_speed_10m"],
    })
    return df

def fetch_uk_proxy_hourly(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build a simple UK proxy by averaging 3 UK cities.
    Returns hourly time series in UTC.
    """
    dfs = []
    for name, (lat, lon) in CITIES.items():
        df = _fetch_city_hourly(lat, lon, start_date, end_date)
        df = df.rename(columns={"temp_c": f"temp_c_{name}", "wind_ms": f"wind_ms_{name}"})
        dfs.append(df)

    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="time", how="inner")

    temp_cols = [c for c in out.columns if c.startswith("temp_c_")]
    wind_cols = [c for c in out.columns if c.startswith("wind_ms_")]

    out["temp_c_uk"] = out[temp_cols].mean(axis=1)
    out["wind_ms_uk"] = out[wind_cols].mean(axis=1)

    return out[["time", "temp_c_uk", "wind_ms_uk"]]
