import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "daily_counts.csv"
PLOTS_DIR = BASE_DIR / "plots"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)

# Load data
daily = pd.read_csv(DATA_PATH, parse_dates=["date"])

# -----------------------
# Plot 1: Daily time series
# -----------------------
plt.figure(figsize=(12, 4))
plt.plot(daily["date"], daily["incident_count"])
plt.title("Montgomery County Dispatch Incidents — Daily Count")
plt.xlabel("Date")
plt.ylabel("Incidents")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "daily_incident_timeseries.png", dpi=150)
plt.close()

# -----------------------
# Plot 2: Average by weekday
# -----------------------
daily["dow"] = daily["date"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

dow = (
    daily.groupby("dow")["incident_count"]
         .mean()
         .reindex(dow_order)
)

plt.figure(figsize=(8, 4))
plt.bar(dow.index, dow.values)
plt.title("Average Dispatch Incidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Avg Incidents")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "avg_incidents_by_weekday.png", dpi=150)
plt.close()

print(f"Plots saved to {PLOTS_DIR}")