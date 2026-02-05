import pandas as pd

df = pd.read_csv("data/weather_daily.csv", parse_dates=["date"])

weather_cols = [
    "tmax_f", "tmin_f", "tmean_f",
    "precip_in", "rain_in", "snow_in",
    "wind_max_mph", "gust_max_mph",
]

print(df[weather_cols].describe())
print(df[weather_cols].isna().sum())

features = pd.read_csv("data/features_daily.csv", parse_dates=["date"])
weather  = pd.read_csv("data/weather_daily.csv", parse_dates=["date"])

df = features.merge(weather, on="date", how="left")

print(df[["incident_count", "tmean_f", "precip_in", "wind_max_mph"]].head())
print(df.isna().sum())