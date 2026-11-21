# scripts/severe_weather.py
"""
Does severe weather move incident counts, and does knowing the weather help on
the days it happens?

A single test year has too few severe days (about 11 heavy-rain days), so this
uses rolling-origin evaluation: six consecutive 365-day test windows ending
2026-02-04. For each window, models are trained only on earlier data, with
hyperparameters chosen on the last 180 days of that training period (the same
train/validation rule as the main split; the most recent window IS the main
split). Every day in the six windows therefore gets an honest out-of-sample
prediction.

For each weather category it reports:
  - residual_no_weather: actual minus the calendar-and-history model's prediction,
    i.e. how many more (or fewer) incidents happened than a model that ignores
    weather expected. This is the estimated weather effect.
  - MAE with and without weather features, and the improvement from adding
    weather, with day-level bootstrap 95% intervals.

Outputs:
    data/rolling_origin_mae.csv      per-window MAE for each model (rolling validation)
    data/severe_weather_effects.csv  per-category effects
    data/oos_predictions.csv         every out-of-sample prediction (gitignored)
    plots/severe_weather_effect.png

Run from the repo root:  .venv/bin/python -m scripts.severe_weather
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from main import get_weather_cols  # noqa: E402
from scripts.compare_models import LEARNERS, fit_select, mae  # noqa: E402
from scripts.train_nn import DATA_PATH, VAL_DAYS, transform_weather  # noqa: E402

N_WINDOWS = 6
WINDOW_DAYS = 365
N_BOOT = 2000
OUT_ROLL = BASE_DIR / "data" / "rolling_origin_mae.csv"
OUT_EFFECTS = BASE_DIR / "data" / "severe_weather_effects.csv"
OUT_PREDS = BASE_DIR / "data" / "oos_predictions.csv"
OUT_PLOT = BASE_DIR / "plots" / "severe_weather_effect.png"

# Weather categories, defined from weather alone (thresholds live in build_features.py).
CATEGORIES = {
    "Any snow": lambda d: d["is_snow_day"] == 1,
    "Snow >= 1 in": lambda d: d["snow_in"] >= 1,
    "Snow >= 3 in": lambda d: d["snow_heavy"] == 1,
    "Day after snow >= 1 in": lambda d: d["snow_in"].shift(1).fillna(0) >= 1,
    "Ice risk (freeze + precip)": lambda d: d["is_ice_risk"] == 1,
    "Heavy rain (>= 0.75 in)": lambda d: d["is_heavy_rain"] == 1,
    "Storm (rain >= 1 in, gusts >= 35)": lambda d: d["is_stormy"] == 1,
    "Gusts >= 40 mph": lambda d: d["is_windy"] == 1,
    "Hot (>= 90 F)": lambda d: d["is_hot_day"] == 1,
}
SEVERE = ["Any snow", "Ice risk (freeze + precip)", "Heavy rain (>= 0.75 in)", "Gusts >= 40 mph"]


def rolling_predictions(df: pd.DataFrame, feature_sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df["date"].max()
    preds, roll_rows = [], []
    for k in range(N_WINDOWS - 1, -1, -1):  # oldest window first
        w_end = end - pd.Timedelta(days=WINDOW_DAYS * k)
        w_start = w_end - pd.Timedelta(days=WINDOW_DAYS)
        hist = df[df["date"] <= w_start]
        val_cut = w_start - pd.Timedelta(days=VAL_DAYS)
        tr, va = hist[hist["date"] <= val_cut], hist[hist["date"] > val_cut]
        te = df[(df["date"] > w_start) & (df["date"] <= w_end)]
        label = f"{te['date'].min():%Y-%m} to {te['date'].max():%Y-%m}"

        out = te[["date", "incident_count"]].copy()
        out["window"] = label
        out["pred_mean_7"] = te["roll_mean_7"].values
        roll_rows.append({"window": label, "model": "Mean of last 7 days", "features": "-",
                          "mae": mae(te["incident_count"], te["roll_mean_7"]), "train_days": len(tr)})

        for lname, (make_model, grid) in LEARNERS.items():
            for fname, cols in feature_sets.items():
                X = {n: transform_weather(d[cols]) for n, d in (("tr", tr), ("va", va), ("te", te))}
                _, params, _ = fit_select(make_model, grid, X["tr"], tr["incident_count"], X["va"], va["incident_count"])
                # Refit the chosen setting on all history before the window (train + validation).
                model = make_model(**params).fit(pd.concat([X["tr"], X["va"]]), hist["incident_count"])
                p = model.predict(X["te"])
                out[f"pred_{lname}_{fname}"] = p
                roll_rows.append({"window": label, "model": lname, "features": fname,
                                  "mae": mae(te["incident_count"], p), "train_days": len(hist),
                                  "chosen_params": ",".join(f"{a}={b}" for a, b in params.items())})
        preds.append(out)
        print(f"  window {label}: train {len(hist)} days, test {len(te)}", flush=True)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(roll_rows)


def boot_ci(x: np.ndarray, rng) -> tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def category_effects(oos: pd.DataFrame, feats: pd.DataFrame, learner: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = oos.merge(feats, on="date", how="left", suffixes=("", "_f"))
    y = d["incident_count"].values
    p_no, p_all = d[f"pred_{learner}_no weather"].values, d[f"pred_{learner}_all"].values
    resid = y - p_no
    gain = np.abs(y - p_no) - np.abs(y - p_all)  # > 0 means weather features helped that day

    masks = {name: f(d).values.astype(bool) for name, f in CATEGORIES.items()}
    severe_any = np.logical_or.reduce([masks[n] for n in SEVERE])
    masks["Any severe (snow, ice, heavy rain, gusts)"] = severe_any
    masks["Ordinary days (none of the above)"] = ~np.logical_or.reduce(list(masks.values()))
    masks["All days"] = np.ones(len(d), bool)

    rows = []
    for name, m in masks.items():
        r_lo, r_hi = boot_ci(resid[m], rng)
        g_lo, g_hi = boot_ci(gain[m], rng)
        rows.append({
            "learner": learner, "category": name, "n_days": int(m.sum()),
            "mean_actual": y[m].mean() if m.any() else np.nan,
            "residual_no_weather": resid[m].mean() if m.any() else np.nan,
            "residual_ci_low": r_lo, "residual_ci_high": r_hi,
            "residual_pct_of_expected": 100 * resid[m].mean() / p_no[m].mean() if m.any() else np.nan,
            "mae_no_weather": np.abs(y - p_no)[m].mean() if m.any() else np.nan,
            "mae_with_weather": np.abs(y - p_all)[m].mean() if m.any() else np.nan,
            "mae_gain_from_weather": gain[m].mean() if m.any() else np.nan,
            "gain_ci_low": g_lo, "gain_ci_high": g_hi,
        })
    return pd.DataFrame(rows)


def plot(eff: pd.DataFrame) -> None:
    d = eff[(eff["learner"] == "Ridge regression") & (eff["category"] != "All days")].iloc[::-1]
    d = d[d["n_days"] >= 5]
    labels = [f"{c}  (n={n})" for c, n in zip(d["category"], d["n_days"])]
    err = np.vstack([d["residual_no_weather"] - d["residual_ci_low"], d["residual_ci_high"] - d["residual_no_weather"]])
    sig = (d["residual_ci_low"] > 0) | (d["residual_ci_high"] < 0)
    colors = ["#b5523b" if (s and v < 0) else "#3b6ea5" if s else "#b8c0c8" for s, v in zip(sig, d["residual_no_weather"])]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(d) + 1.4))
    ax.barh(labels, d["residual_no_weather"], xerr=err, color=colors, capsize=2)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Incidents vs. what a weather-blind model expected (mean, 95% CI)")
    fig.suptitle("Weather effect on daily incidents (out-of-sample, 2020-2026)", x=0.02, ha="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#b5523b", label="fewer incidents"), Patch(color="#3b6ea5", label="more incidents"),
                       Patch(color="#b8c0c8", label="CI includes zero")], loc="lower left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    all_cols = [c for c in df.columns if c not in ("date", "incident_count") and pd.api.types.is_numeric_dtype(df[c])]
    weather = set(get_weather_cols(df))
    feature_sets = {"all": all_cols, "no weather": [c for c in all_cols if c not in weather]}

    oos, roll = rolling_predictions(df, feature_sets)
    oos.to_csv(OUT_PREDS, index=False)
    roll.to_csv(OUT_ROLL, index=False)

    summary = roll.pivot_table(index=["model", "features"], columns="window", values="mae")
    summary["mean"] = summary.mean(axis=1)
    print("\nRolling-origin MAE by test window:")
    print(summary.sort_values("mean").to_string(float_format="%.1f"))

    wx = df[["date"] + sorted(weather)]
    eff = pd.concat([category_effects(oos, wx, l) for l in LEARNERS], ignore_index=True)
    eff.to_csv(OUT_EFFECTS, index=False)
    plot(eff)
    cols = ["category", "n_days", "mean_actual", "residual_no_weather", "residual_ci_low", "residual_ci_high",
            "residual_pct_of_expected", "mae_gain_from_weather", "gain_ci_low", "gain_ci_high"]
    for l in LEARNERS:
        print(f"\n{l}: weather effects (out-of-sample, {len(oos)} days)")
        print(eff[eff["learner"] == l][cols].to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
