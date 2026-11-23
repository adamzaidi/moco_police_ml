# scripts/astro.py
"""
Sunset times and moon phase, computed (no data download).

- sunset_local: NOAA's low-precision solar formula (fractional-year form);
  accurate to about 1-2 minutes, plenty for hour-level analysis.
- full_moon_instants: low-precision lunar and solar longitudes (Meeus,
  Astronomical Algorithms ch. 25/47, main terms only); full moon is when their
  difference crosses 180 degrees. Accurate to well under an hour, checked against
  published full-moon times in __main__.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Rockville, roughly the county's center (same point as the weather data).
LAT, LON = 39.08, -77.15
TZ = "America/New_York"


def sunset_local(dates: pd.DatetimeIndex, lat: float = LAT, lon: float = LON, tz: str = TZ) -> pd.Series:
    """Local sunset (tz-naive local clock time) for each date."""
    d = pd.DatetimeIndex(dates).normalize()
    doy = d.day_of_year.values
    g = 2 * np.pi / 365 * (doy - 1 + 0.5)  # fractional year at local noon
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    phi = np.radians(lat)
    ha = np.degrees(np.arccos(np.cos(np.radians(90.833)) / (np.cos(phi) * np.cos(decl)) - np.tan(phi) * np.tan(decl)))
    utc_min = 720 - 4 * (lon - ha) - eqtime  # minutes after UTC midnight
    utc = pd.DatetimeIndex(d + pd.to_timedelta(utc_min, unit="min")).tz_localize("UTC")
    return pd.Series(utc.tz_convert(tz).tz_localize(None), index=d)


def _elongation(t_utc: pd.DatetimeIndex) -> np.ndarray:
    """Moon minus sun ecliptic longitude, degrees in [0, 360)."""
    jd = t_utc.to_julian_date().values
    T = (jd - 2451545.0) / 36525
    # sun
    L0 = 280.46646 + 36000.76983 * T
    M = np.radians(357.52911 + 35999.05029 * T)
    sun = L0 + (1.914602 - 0.004817 * T) * np.sin(M) + 0.019993 * np.sin(2 * M) + 0.000289 * np.sin(3 * M)
    # moon
    Lp = 218.3164477 + 481267.88123421 * T
    D = np.radians(297.8501921 + 445267.1114034 * T)
    Mp = np.radians(134.9633964 + 477198.8675055 * T)
    F = np.radians(93.2720950 + 483202.0175233 * T)
    moon = (Lp + 6.288774 * np.sin(Mp) + 1.274027 * np.sin(2 * D - Mp) + 0.658314 * np.sin(2 * D)
            + 0.213618 * np.sin(2 * Mp) - 0.185116 * np.sin(M) - 0.114332 * np.sin(2 * F)
            + 0.058793 * np.sin(2 * D - 2 * Mp) + 0.057066 * np.sin(2 * D - M - Mp)
            + 0.053322 * np.sin(2 * D + Mp) + 0.045758 * np.sin(2 * D - M) - 0.040923 * np.sin(M - Mp)
            - 0.034720 * np.sin(D) - 0.030383 * np.sin(M + Mp))
    return np.mod(moon - sun, 360)


def phase_instants(start: str, end: str, target: float) -> pd.DatetimeIndex:
    """UTC instants when the moon-sun elongation crosses `target` degrees (180 = full, 0 = new)."""
    t = pd.date_range(pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=2),
                      pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2), freq="h")
    e = np.mod(_elongation(t) - target + 180, 360) - 180  # signed distance from target, in (-180, 180]
    idx = np.where((e[:-1] < 0) & (e[1:] >= 0) & (e[1:] - e[:-1] < 90))[0]
    out = []
    for i in idx:  # linear interpolation within the hour
        frac = -e[i] / (e[i + 1] - e[i])
        out.append(t[i] + pd.Timedelta(hours=float(frac)))
    return pd.DatetimeIndex(out)


def full_moon_instants(start: str, end: str) -> pd.DatetimeIndex:
    return phase_instants(start, end, 180)


def new_moon_instants(start: str, end: str) -> pd.DatetimeIndex:
    return phase_instants(start, end, 0)


if __name__ == "__main__":
    s = sunset_local(pd.to_datetime(["2024-03-09", "2024-03-10", "2024-11-02", "2024-11-03", "2024-06-21"]))
    print("Sunsets (expect about 18:09, 19:10, 18:08, 17:07, 20:37):")
    print(s.dt.strftime("%H:%M").to_string())
    fm = full_moon_instants("2019-01-01", "2025-01-31")
    print("Full moons (expect 2019-01-21 05:16, 2024-04-23 23:49, 2025-01-13 22:27 UTC):")
    for ref in ["2019-01-21 05:16", "2024-04-23 23:49", "2025-01-13 22:27"]:
        r = pd.Timestamp(ref, tz="UTC")
        print(" ", fm[np.argmin(np.abs(fm - r))].strftime("%Y-%m-%d %H:%M"))
    nm = new_moon_instants("2024-04-01", "2024-04-30")
    print("New moon (expect 2024-04-08 18:21 UTC, the eclipse):", nm[0].strftime("%Y-%m-%d %H:%M"))
