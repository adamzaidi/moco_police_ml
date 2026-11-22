# scripts/fetch_weather_forecast.py
"""
Day-ahead weather forecasts, as they were issued.

Uses Open-Meteo's Previous Runs API: `<var>_previous_day1` is the value for each
hour from the model run issued about 24 hours earlier (day2 = about 48 hours).
For a given day, every hourly value comes from a run issued on an earlier day,
so it is information a forecaster had before that day began (day1: from runs
issued during the previous day; day2: two days before).

Coverage for precipitation, snow and gusts starts 2024-01-19 (temperature goes
back further), so only the last two rolling-origin test windows can be scored.
Rain must match the archive's definition, where rain = precipitation on days
without snow. Forecast models report convective rain separately as `showers`
(the 2-day-lead blend does, from Dec 2024), so rain here is rain + showers.
Without this, forecast rain ran below precipitation on 26% of dry-snow days, and
ridge, which weights rain and precipitation about +400 and -355 per log-inch,
turned that gap into large prediction errors.
`rain` itself only starts 2024-12-06; before that it is derived as precipitation
in hours with no snowfall. Where both exist, the derived daily rain correlates
0.99 with the real value and flags the same days at >= 1 in.

Hourly values are aggregated to the same daily columns as weather_daily.csv
(America/New_York days). Days missing any hour are dropped.

Output: data/weather_forecast_daily.csv  (columns: lead_days, date, raw weather)
Run from the repo root:  .venv/bin/python scripts/fetch_weather_forecast.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from fetch_weather_openmeteo import LAT, LON, END_DATE

START_DATE = "2024-01-19"
LEADS = (1, 2)
HOURLY = ["temperature_2m", "precipitation", "rain", "showers", "snowfall", "wind_speed_10m", "wind_gusts_10m"]
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "weather_forecast_daily.csv"


def fetch_lead(lead: int) -> pd.DataFrame:
    names = [f"{v}_previous_day{lead}" for v in HOURLY]
    r = requests.get(URL, timeout=120, params={
        "latitude": LAT, "longitude": LON, "start_date": START_DATE, "end_date": END_DATE,
        "hourly": ",".join(names), "timezone": "America/New_York",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
    })
    r.raise_for_status()
    data = r.json()
    units = data["hourly_units"]
    assert units[f"snowfall_previous_day{lead}"] == "inch", units  # see snow-units note in the archive fetcher
    h = pd.DataFrame(data["hourly"]).rename(columns=dict(zip(names, HOURLY)))
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.floor("D")
    h["rain"] = (h["rain"] + h["showers"].fillna(0)).fillna(h["precipitation"].where(h["snowfall"] == 0, 0.0))

    complete = h.dropna(subset=[c for c in HOURLY if c != "showers"]).groupby("date").size()
    g = h.groupby("date")
    daily = pd.DataFrame({
        "tmax_f": g["temperature_2m"].max(),
        "tmin_f": g["temperature_2m"].min(),
        "precip_in": g["precipitation"].sum(),
        "rain_in": g["rain"].sum(),
        "snow_in": g["snowfall"].sum(),
        "precip_hours": g["precipitation"].apply(lambda s: (s > 0).sum()).astype(float),
        "wind_max_mph": g["wind_speed_10m"].max(),
        "gust_max_mph": g["wind_gusts_10m"].max(),
    })
    daily["tmean_f"] = (daily["tmax_f"] + daily["tmin_f"]) / 2
    daily = daily[complete.reindex(daily.index).fillna(0) >= 23]  # DST days have 23 hours
    daily.insert(0, "lead_days", lead)
    return daily.reset_index()


def main() -> None:
    out = pd.concat([fetch_lead(k) for k in LEADS], ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    for k, d in out.groupby("lead_days"):
        print(f"lead {k}: {len(d)} days, {d['date'].min():%Y-%m-%d} to {d['date'].max():%Y-%m-%d}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
