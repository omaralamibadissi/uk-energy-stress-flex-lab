import requests
import pandas as pd

BASE_URL = "https://api.carbonintensity.org.uk"

def fetch_carbon_intensity(start_iso: str, end_iso: str) -> pd.DataFrame:

    url = f"{BASE_URL}/intensity/{start_iso}/{end_iso}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    payload = r.json()["data"]
    rows = []
    for item in payload:
        rows.append({
            "from": item["from"],
            "to": item["to"],
            "forecast": item["intensity"]["forecast"],
            "actual": item["intensity"].get("actual", None),
            "index": item["intensity"]["index"],
        })

    df = pd.DataFrame(rows)
    df["from"] = pd.to_datetime(df["from"], utc=True)
    df["to"] = pd.to_datetime(df["to"], utc=True)
    return df

if __name__ == "__main__":
    df = fetch_carbon_intensity("2026-01-01T00:00Z", "2026-01-08T00:00Z")
    print(df.head())
