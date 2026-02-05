# scripts/fetch_weather_openmeteo.py
from pathlib import Path
import pandas as pd
import requests

# ------------------------------------------------------------
# Config: Montgomery County, MD (approx centroid)
# You can tweak later; this is good enough for a baseline.
# ------------------------------------------------------------
LAT = 39.1547
LON = -77.2405

START_DATE = "2017-04-01"
END_DATE   = None  # if None, we'll auto-use "today" based on your counts later if you want

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "weather_daily.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def fetch_openmeteo_daily(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ]),
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise RuntimeError(f"Unexpected response format. Keys: {list(data.keys())}")

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "tmax_f": daily.get("temperature_2m_max"),
        "tmin_f": daily.get("temperature_2m_min"),
        "precip_in": daily.get("precipitation_sum"),
        "rain_in": daily.get("rain_sum"),
        "snow_cm": daily.get("snowfall_sum"),  # NOTE: Open-Meteo returns snowfall_sum in cm even if precip is inch
        "precip_hours": daily.get("precipitation_hours"),
        "wind_max_mph": daily.get("wind_speed_10m_max"),
        "gust_max_mph": daily.get("wind_gusts_10m_max"),
    })

    # Convenience: convert snow from cm to inches (1 inch = 2.54 cm)
    df["snow_in"] = df["snow_cm"].astype(float) / 2.54

    # Optional: temp mean
    df["tmean_f"] = (df["tmax_f"].astype(float) + df["tmin_f"].astype(float)) / 2.0

    # Keep tidy
    df = df.drop(columns=["snow_cm"])
    return df

def main():
    # If you want to auto-align to your crime data’s end date, we can read it here.
    # For now, set END_DATE explicitly to avoid confusion.
    end_date = END_DATE or pd.Timestamp.today(tz="America/New_York").date().isoformat()

    print("Fetching Open-Meteo daily weather...")
    print("Location:", LAT, LON)
    print("Date range:", START_DATE, "→", end_date)

    dfw = fetch_openmeteo_daily(START_DATE, end_date)

    # Remove any duplicate dates (rare, but safe)
    dfw = dfw.sort_values("date").drop_duplicates("date")

    dfw.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print("Rows:", len(dfw))
    print("Sample:")
    print(dfw.head())

if __name__ == "__main__":
    main()