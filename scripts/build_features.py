# scripts/build_features.py
import pandas as pd
from pathlib import Path

from pandas.tseries.holiday import USFederalHolidayCalendar

# -----------------------
# 0) Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "daily_counts.csv"
WEATHER_PATH = BASE_DIR / "data" / "weather_daily.csv"
OUT_PATH = BASE_DIR / "data" / "features_daily.csv"

# -----------------------
# 1) Load daily incidents
# -----------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# -----------------------
# 2) Add US federal holidays (creates holiday_name)
# -----------------------
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df["date"].min(), end=df["date"].max())
holidays = pd.to_datetime(holidays).normalize()

hset = set(holidays)

def holiday_label(d: pd.Timestamp) -> str:
    if d.normalize() not in hset:
        return ""

    m = d.month
    day = d.day
    dow = d.weekday()  # Mon=0

    if m == 1 and day in (1, 2, 3) and dow in (0, 4, 5, 6):
        return "New Year's Day"

    if m == 7 and day in (3, 4, 5):
        return "Independence Day"

    if m == 12 and day in (24, 25, 26):
        return "Christmas Day"

    if m == 11 and dow == 3 and 22 <= day <= 28:
        return "Thanksgiving Day"

    if m == 5 and dow == 0 and day >= 25:
        return "Memorial Day"

    if m == 9 and dow == 0 and day <= 7:
        return "Labor Day"

    return "US Federal Holiday"

df["holiday_name"] = df["date"].apply(holiday_label)

df["is_holiday"] = (df["holiday_name"] != "").astype(int)
df["is_day_before_holiday"] = df["is_holiday"].shift(-1).fillna(0).astype(int)
df["is_day_after_holiday"] = df["is_holiday"].shift(1).fillna(0).astype(int)

# -----------------------
# 3) Named holiday windows (date-derived, no leakage)
# -----------------------
dates = df["date"]

is_thanksgiving = (dates.dt.month == 11) & (dates.dt.weekday == 3) & (dates.dt.day.between(22, 28))
thanksgiving_dates = df["date"].where(is_thanksgiving)

def in_window(dates: pd.Series, centers: pd.Series, days_before: int, days_after: int) -> pd.Series:
    prev_center = centers.ffill()
    next_center = centers.bfill()

    in_prev = (
        (prev_center.notna())
        & (dates >= (prev_center - pd.Timedelta(days=days_before)))
        & (dates <= (prev_center + pd.Timedelta(days=days_after)))
    )
    in_next = (
        (next_center.notna())
        & (dates >= (next_center - pd.Timedelta(days=days_before)))
        & (dates <= (next_center + pd.Timedelta(days=days_after)))
    )
    return (in_prev | in_next)

df["is_thanksgiving_window"] = in_window(df["date"], thanksgiving_dates, 3, 3).astype(int)

df["is_christmas_window"] = (
    ((dates.dt.month == 12) & (dates.dt.day >= 18))
    | ((dates.dt.month == 1) & (dates.dt.day <= 2))
).astype(int)

df["is_newyear_window"] = (
    ((dates.dt.month == 12) & (dates.dt.day >= 29))
    | ((dates.dt.month == 1) & (dates.dt.day <= 3))
).astype(int)

df["is_july4_window"] = ((dates.dt.month == 7) & (dates.dt.day.between(2, 6))).astype(int)

is_memorial = (dates.dt.month == 5) & (dates.dt.weekday == 0) & (dates.dt.day >= 25)
memorial_dates = df["date"].where(is_memorial)
df["is_memorialday_window"] = in_window(df["date"], memorial_dates, 2, 2).astype(int)

is_labor = (dates.dt.month == 9) & (dates.dt.weekday == 0) & (dates.dt.day <= 7)
labor_dates = df["date"].where(is_labor)
df["is_laborday_window"] = in_window(df["date"], labor_dates, 2, 2).astype(int)

def holiday_cat(row) -> str:
    if row["is_thanksgiving_window"] == 1:
        return "thanksgiving"
    if row["is_christmas_window"] == 1 or row["is_newyear_window"] == 1:
        return "christmas_newyear"
    if row["is_july4_window"] == 1 or row["is_memorialday_window"] == 1 or row["is_laborday_window"] == 1:
        return "summer"
    return "none"

df["holiday_cat"] = df.apply(holiday_cat, axis=1)
df = pd.get_dummies(df, columns=["holiday_cat"], prefix="holiday_cat", drop_first=False)

# -----------------------
# 4) Merge weather (left join keeps your incident timeline)
# -----------------------
weather_cols = [
    "tmax_f", "tmin_f", "tmean_f",
    "precip_in", "rain_in", "snow_in",
    "precip_hours", "wind_max_mph", "gust_max_mph",
]
has_weather = False

if WEATHER_PATH.exists():
    w = pd.read_csv(WEATHER_PATH, parse_dates=["date"])
    w = w.sort_values("date").drop_duplicates("date")

    missing = [c for c in weather_cols if c not in w.columns]
    if missing:
        print(f"WARNING: weather file is missing columns: {missing}. Proceeding without weather features.")
    else:
        w = w[["date"] + weather_cols].copy()
        df = df.merge(w, on="date", how="left")
        has_weather = True
else:
    print(f"WARNING: weather file not found at {WEATHER_PATH}. Proceeding without weather.")

# -----------------------
# 5) Weather-derived features (interpretable, no leakage)
#    UPGRADE: severity tiers + compound danger + short streaks
# -----------------------
if has_weather:
    for c in weather_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Basic flags (kept for continuity) ---
    df["is_rainy"] = (df["precip_in"] >= 0.10).astype(int)
    df["is_heavy_rain"] = (df["precip_in"] >= 0.75).astype(int)
    df["is_snow_day"] = (df["snow_in"] > 0.0).astype(int)
    df["is_windy"] = (df["wind_max_mph"] >= 25).astype(int)
    df["is_hot_day"] = (df["tmax_f"] >= 90).astype(int)
    df["is_freeze_night"] = (df["tmin_f"] <= 32).astype(int)

    # --- Severity tiers (more separation / "danger levels") ---
    # Snow inches: light/mod/heavy
    df["snow_light"] = ((df["snow_in"] > 0) & (df["snow_in"] < 1)).astype(int)
    df["snow_mod"]   = ((df["snow_in"] >= 1) & (df["snow_in"] < 3)).astype(int)
    df["snow_heavy"] = (df["snow_in"] >= 3).astype(int)

    # Rain inches: light/mod/heavy (use rain_in, not precip_in)
    df["rain_light"] = ((df["rain_in"] > 0) & (df["rain_in"] < 0.25)).astype(int)
    df["rain_mod"]   = ((df["rain_in"] >= 0.25) & (df["rain_in"] < 1.0)).astype(int)
    df["rain_heavy"] = (df["rain_in"] >= 1.0).astype(int)

    # Wind tiers
    df["wind_breezy"] = ((df["wind_max_mph"] >= 15) & (df["wind_max_mph"] < 25)).astype(int)
    df["wind_windy"]  = ((df["wind_max_mph"] >= 25) & (df["wind_max_mph"] < 35)).astype(int)
    df["wind_gale"]   = (df["wind_max_mph"] >= 35).astype(int)

    # --- Compound danger proxies ---
    # Ice risk: freezing overnight + any precip
    df["is_ice_risk"] = ((df["tmin_f"] <= 32) & (df["precip_in"] > 0)).astype(int)

    # Blowing snow proxy: meaningful snow + windy
    df["is_blowing_snow"] = ((df["snow_in"] >= 1) & (df["wind_max_mph"] >= 25)).astype(int)

    # Heavy rain + windy: storm proxy
    df["is_stormy"] = ((df["rain_in"] >= 1.0) & (df["wind_max_mph"] >= 25)).astype(int)

    # --- Rolling 30-day normals (past-only) + anomalies ---
    df["tmean_roll30"] = df["tmean_f"].shift(1).rolling(30, min_periods=30).mean()
    df["precip_roll30"] = df["precip_in"].shift(1).rolling(30, min_periods=30).mean()

    df["temp_anom_30"] = df["tmean_f"] - df["tmean_roll30"]
    df["precip_anom_30"] = df["precip_in"] - df["precip_roll30"]

    # --- Short streak / recent-bad-weather counts (PAST ONLY) ---
    # “In the last 3 days (excluding today), how many days were X?”
    df["snow_days_last3"] = df["snow_in"].shift(1).rolling(3, min_periods=3).apply(lambda x: (x > 0).sum())
    df["heavy_snow_last3"] = df["snow_in"].shift(1).rolling(3, min_periods=3).apply(lambda x: (x >= 3).sum())
    df["heavy_rain_last3"] = df["rain_in"].shift(1).rolling(3, min_periods=3).apply(lambda x: (x >= 1.0).sum())
    df["ice_risk_last3"] = df["is_ice_risk"].shift(1).rolling(3, min_periods=3).sum()

# -----------------------
# 6) Create lag features
# -----------------------
lags = [1, 7, 14, 28]
for lag in lags:
    df[f"lag_{lag}"] = df["incident_count"].shift(lag)

# -----------------------
# 7) Rolling incident means (no peeking)
# -----------------------
df["roll_mean_7"] = df["incident_count"].shift(1).rolling(window=7).mean()
df["roll_mean_28"] = df["incident_count"].shift(1).rolling(window=28).mean()

# -----------------------
# 8) Calendar features
# -----------------------
df["dow"] = df["date"].dt.weekday
df = pd.get_dummies(df, columns=["dow"], prefix="dow", drop_first=True)

# -----------------------
# 9) Select feature columns
# -----------------------
feature_cols = [
    "incident_count",
    *[f"lag_{l}" for l in lags],
    "roll_mean_7",
    "roll_mean_28",
] + [c for c in df.columns if c.startswith("dow_")]

holiday_feature_cols = [
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_thanksgiving_window",
    "is_christmas_window",
    "is_newyear_window",
    "is_july4_window",
    "is_memorialday_window",
    "is_laborday_window",
] + [c for c in df.columns if c.startswith("holiday_cat_")]

feature_cols += holiday_feature_cols

if has_weather:
    feature_cols += [
        # raw weather
        "tmax_f", "tmin_f", "tmean_f",
        "precip_in", "rain_in", "snow_in",
        "precip_hours", "wind_max_mph", "gust_max_mph",

        # basic flags
        "is_rainy", "is_heavy_rain", "is_snow_day", "is_windy", "is_hot_day", "is_freeze_night",

        # tiers
        "snow_light", "snow_mod", "snow_heavy",
        "rain_light", "rain_mod", "rain_heavy",
        "wind_breezy", "wind_windy", "wind_gale",

        # compound danger
        "is_ice_risk", "is_blowing_snow", "is_stormy",

        # anomalies
        "temp_anom_30", "precip_anom_30",

        # streaks
        "snow_days_last3", "heavy_snow_last3", "heavy_rain_last3", "ice_risk_last3",
    ]

df_features = df[["date"] + feature_cols].dropna().reset_index(drop=True)

# -----------------------
# 10) Save
# -----------------------
df_features.to_csv(OUT_PATH, index=False)

print(f"Saved feature table to: {OUT_PATH}")
print(f"Rows: {len(df_features)}")
print("\nHas weather:", has_weather)
print("\nColumns:")
for c in df_features.columns:
    print(" -", c)
print("\nSample:")
print(df_features.head())