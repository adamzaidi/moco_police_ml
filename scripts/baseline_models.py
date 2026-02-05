import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# 0) Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "daily_counts.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# -----------------------
# 1) Load the daily dataset
# -----------------------
daily = pd.read_csv(DATA_PATH, parse_dates=["date"])
daily = daily.sort_values("date").reset_index(drop=True)

# Quick sanity check
print("Date range:", daily["date"].min().date(), "to", daily["date"].max().date())
print("Rows:", len(daily))

# -----------------------
# 2) Make baseline predictions
# -----------------------
# Baseline A: yesterday
daily["pred_yesterday"] = daily["incident_count"].shift(1)

# Baseline B: same day last week
daily["pred_last_week"] = daily["incident_count"].shift(7)

# -----------------------
# 3) Choose a test period (time-aware split)
# -----------------------
# We'll test on the last 365 days to simulate "future" prediction.
max_date = daily["date"].max()
cutoff = max_date - pd.Timedelta(days=365)

test = daily[daily["date"] > cutoff].copy()

# Some rows at the beginning of test may have NaNs (because shift needs prior days)
test = test.dropna(subset=["pred_yesterday", "pred_last_week"])

print("\nTest period:", test["date"].min().date(), "to", test["date"].max().date())
print("Test rows used (after dropping NaNs):", len(test))

# -----------------------
# 4) Evaluate with MAE (Mean Absolute Error)
# -----------------------
def mae(y_true, y_pred) -> float:
    return (y_true - y_pred).abs().mean()

mae_yesterday = mae(test["incident_count"], test["pred_yesterday"])
mae_last_week = mae(test["incident_count"], test["pred_last_week"])

print("\n=== BASELINE RESULTS (Lower MAE is better) ===")
print(f"MAE (Yesterday baseline):  {mae_yesterday:.2f}")
print(f"MAE (Last-week baseline):  {mae_last_week:.2f}")

# -----------------------
# 5) Save a plot (last 90 days) so it’s easy to visually inspect
# -----------------------
last_90 = test.tail(90)

plt.figure(figsize=(12, 4))
plt.plot(last_90["date"], last_90["incident_count"], label="Actual")
plt.plot(last_90["date"], last_90["pred_yesterday"], label="Pred: Yesterday")
plt.plot(last_90["date"], last_90["pred_last_week"], label="Pred: Last Week")
plt.title("Baselines vs Actual (Last 90 Days of Test)")
plt.xlabel("Date")
plt.ylabel("Incidents")
plt.legend()
plt.tight_layout()

out_path = PLOTS_DIR / "baselines_last_90_days.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"\nSaved plot: {out_path}")

# -----------------------
# 6) (Optional) Save test predictions to CSV for inspection
# -----------------------
out_csv = BASE_DIR / "data" / "baseline_test_predictions.csv"
test[["date", "incident_count", "pred_yesterday", "pred_last_week"]].to_csv(out_csv, index=False)
print(f"Saved predictions: {out_csv}")