import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw_incidents.csv"
OUT_PATH = BASE_DIR / "data" / "daily_counts.csv"

# Read only the columns we need (fast + memory-friendly)
usecols = ["Start Time"]

# If your CSV is huge, parsing dates on read is worth it
df = pd.read_csv(CSV_PATH, usecols=usecols)

# Parse datetime safely
dt = pd.to_datetime(df["Start Time"], errors="coerce")

# Drop rows where Start Time couldn't be parsed
dt = dt.dropna()

# Convert to date (daily bucket)
daily = (
    dt.dt.floor("D")
      .value_counts()
      .sort_index()
      .rename_axis("date")
      .reset_index(name="incident_count")
)

# Fill missing dates (important!)
all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
daily = daily.set_index("date").reindex(all_days, fill_value=0).rename_axis("date").reset_index()

daily.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(daily):,} rows to {OUT_PATH}")
print(daily.head())
print(daily.tail())