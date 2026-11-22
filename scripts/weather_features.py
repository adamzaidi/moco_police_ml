# scripts/weather_features.py
"""
Same-day weather features, defined once.

build_features.py applies these to observed weather; forecast_weather.py
applies the same function to day-ahead forecasts so both feature sets use
identical thresholds. Past-only features (30-day normals, 3-day streaks) stay
in build_features.py because they only use days already observed.
"""
from __future__ import annotations

import pandas as pd

# Raw daily weather columns, as written by fetch_weather_openmeteo.py.
RAW_WEATHER = [
    "tmax_f", "tmin_f", "tmean_f",
    "precip_in", "rain_in", "snow_in",
    "precip_hours", "wind_max_mph", "gust_max_mph",
]


def add_same_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flags, severity tiers, compound flags and anomalies from same-day raw weather.

    Anomalies need `tmean_roll30` and `precip_roll30` (past-only 30-day means)
    already on the frame.
    """
    # --- Basic flags ---
    df["is_rainy"] = (df["precip_in"] >= 0.10).astype(int)
    df["is_heavy_rain"] = (df["precip_in"] >= 0.75).astype(int)
    df["is_snow_day"] = (df["snow_in"] > 0.0).astype(int)
    # Wind flags use gusts: Open-Meteo daily sustained wind is smoothed and never
    # exceeds ~30 mph here, so sustained-wind thresholds almost never fire.
    df["is_windy"] = (df["gust_max_mph"] >= 40).astype(int)
    df["is_hot_day"] = (df["tmax_f"] >= 90).astype(int)
    df["is_freeze_night"] = (df["tmin_f"] <= 32).astype(int)

    # --- Severity tiers ---
    df["snow_light"] = ((df["snow_in"] > 0) & (df["snow_in"] < 1)).astype(int)
    df["snow_mod"] = ((df["snow_in"] >= 1) & (df["snow_in"] < 3)).astype(int)
    df["snow_heavy"] = (df["snow_in"] >= 3).astype(int)

    # Rain tiers use rain_in, not precip_in
    df["rain_light"] = ((df["rain_in"] > 0) & (df["rain_in"] < 0.25)).astype(int)
    df["rain_mod"] = ((df["rain_in"] >= 0.25) & (df["rain_in"] < 1.0)).astype(int)
    df["rain_heavy"] = (df["rain_in"] >= 1.0).astype(int)

    # Wind tiers (gusts); top tier = NWS wind advisory gust threshold (46 mph)
    df["wind_breezy"] = ((df["gust_max_mph"] >= 30) & (df["gust_max_mph"] < 40)).astype(int)
    df["wind_windy"] = ((df["gust_max_mph"] >= 40) & (df["gust_max_mph"] < 46)).astype(int)
    df["wind_gale"] = (df["gust_max_mph"] >= 46).astype(int)

    # --- Compound danger proxies ---
    df["is_ice_risk"] = ((df["tmin_f"] <= 32) & (df["precip_in"] > 0)).astype(int)
    df["is_blowing_snow"] = ((df["snow_in"] >= 1) & (df["gust_max_mph"] >= 35)).astype(int)
    df["is_stormy"] = ((df["rain_in"] >= 1.0) & (df["gust_max_mph"] >= 35)).astype(int)

    # --- Anomalies vs past 30 days ---
    df["temp_anom_30"] = df["tmean_f"] - df["tmean_roll30"]
    df["precip_anom_30"] = df["precip_in"] - df["precip_roll30"]
    return df
