# scripts/forecast_weather.py
"""
Does the weather gain survive when the model only gets a forecast?

Every other script feeds the model the weather OBSERVED on the day it predicts,
which nobody has in advance. This rescores the last two rolling-origin test
windows (Feb 2024 - Feb 2026, the span the archived forecasts cover) three ways,
on the same days with the same fitted models:

  - no weather        calendar + incident history only
  - forecast weather  same-day weather replaced by the forecast issued 1 or 2
                      days ahead (fetch_weather_forecast.py)
  - observed weather  what the other scripts use (an upper bound)

Models are trained on observed weather exactly as in severe_weather.py (ridge
and gradient boosting, hyperparameters chosen on the last 180 days of history,
refit on all history). Only the test-time inputs change. Past-only weather
features (30-day normals, 3-day streaks) stay observed in every version, since
a forecaster knows yesterday's weather.

Weather categories (snow days, gusty days, ...) are always defined by what was
observed, so "severe days" means the same days in every version.

Outputs:
    data/forecast_vs_observed.csv          MAE and weather gain by version, day group, learner
    data/forecast_event_skill.csv          how often the forecast flagged the events that happened
    plots/forecast_vs_observed.png

Run from the repo root:  .venv/bin/python -m scripts.forecast_weather
"""
from __future__ import annotations

import os
import sys

# PyTorch (imported via main) and scikit-learn each ship an OpenMP runtime; with
# both loaded, gradient boosting's thread pool can spin at a barrier forever on
# macOS. Single-threaded is fast enough here and gives identical results.
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from main import get_weather_cols  # noqa: E402
from scripts.compare_models import LEARNERS, fit_select, mae  # noqa: E402
from scripts.severe_weather import CATEGORIES, N_BOOT, SEVERE, WINDOW_DAYS  # noqa: E402
from scripts.train_nn import DATA_PATH, VAL_DAYS, transform_weather  # noqa: E402
from scripts.weather_features import RAW_WEATHER, add_same_day_features  # noqa: E402

WEATHER_PATH = BASE_DIR / "data" / "weather_daily.csv"
FORECAST_PATH = BASE_DIR / "data" / "weather_forecast_daily.csv"
OUT_CSV = BASE_DIR / "data" / "forecast_vs_observed.csv"
OUT_SKILL = BASE_DIR / "data" / "forecast_event_skill.csv"
OUT_PLOT = BASE_DIR / "plots" / "forecast_vs_observed.png"
N_WINDOWS = 2  # the forecast archive starts 2024-01-20, so only the last two yearly windows

VERSIONS = ["Forecast, 2 days ahead", "Forecast, 1 day ahead", "Observed (upper bound)"]
DAY_GROUPS = ["All days", "Any severe", "Snow >= 1 in", "Gusts >= 40 mph", "Ordinary days"]

# Events checked for forecast skill: (label, rule on a frame of raw + flag columns)
EVENTS = {
    "Any snow": lambda d: d["snow_in"] > 0,
    "Snow >= 1 in": lambda d: d["snow_in"] >= 1,
    "Ice risk (freeze + precip)": lambda d: d["is_ice_risk"] == 1,
    "Heavy rain (>= 0.75 in)": lambda d: d["is_heavy_rain"] == 1,
    "Gusts >= 40 mph": lambda d: d["is_windy"] == 1,
    "Hot (>= 90 F)": lambda d: d["is_hot_day"] == 1,
}


def forecast_features(df: pd.DataFrame, lead: int) -> pd.DataFrame:
    """The feature table with same-day weather swapped for the forecast issued `lead` days ahead."""
    obs = pd.read_csv(WEATHER_PATH, parse_dates=["date"]).sort_values("date")
    obs["tmean_roll30"] = obs["tmean_f"].shift(1).rolling(30, min_periods=30).mean()  # past days: observed
    obs["precip_roll30"] = obs["precip_in"].shift(1).rolling(30, min_periods=30).mean()
    fc = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    fc = fc[fc["lead_days"] == lead].drop(columns="lead_days")

    d = df.drop(columns=RAW_WEATHER).merge(fc, on="date", how="inner")
    d = d.merge(obs[["date", "tmean_roll30", "precip_roll30"]], on="date", how="left")
    d = add_same_day_features(d)  # same thresholds as the observed features
    return d[df.columns]


def windowed_predictions(df: pd.DataFrame, test_inputs: dict[str, pd.DataFrame],
                         feature_sets: dict) -> pd.DataFrame:
    """Fit on observed history per window; predict the window from each version of the inputs."""
    end = df["date"].max()
    out = []
    for k in range(N_WINDOWS - 1, -1, -1):
        w_end = end - pd.Timedelta(days=WINDOW_DAYS * k)
        w_start = w_end - pd.Timedelta(days=WINDOW_DAYS)
        hist = df[df["date"] <= w_start]
        val_cut = w_start - pd.Timedelta(days=VAL_DAYS)
        tr, va = hist[hist["date"] <= val_cut], hist[hist["date"] > val_cut]
        te_dates = df[(df["date"] > w_start) & (df["date"] <= w_end)]["date"]
        tests = {v: t[t["date"].isin(te_dates)].set_index("date").loc[te_dates].reset_index()
                 for v, t in test_inputs.items()}
        rows = tests[VERSIONS[-1]][["date", "incident_count"]].copy()
        rows["window"] = f"{te_dates.min():%Y-%m} to {te_dates.max():%Y-%m}"

        for lname, (make_model, grid) in LEARNERS.items():
            for fname, cols in feature_sets.items():
                Xtr, Xva = transform_weather(tr[cols]), transform_weather(va[cols])
                _, params, _ = fit_select(make_model, grid, Xtr, tr["incident_count"], Xva, va["incident_count"])
                model = make_model(**params).fit(pd.concat([Xtr, Xva]), hist["incident_count"])
                if fname == "no weather":
                    rows[f"{lname}|No weather"] = model.predict(transform_weather(tests[VERSIONS[-1]][cols]))
                else:
                    for v, t in tests.items():
                        rows[f"{lname}|{v}"] = model.predict(transform_weather(t[cols]))
        out.append(rows)
        print(f"  window {rows['window'].iat[0]}: train {len(hist)} days, test {len(rows)}", flush=True)
    return pd.concat(out, ignore_index=True)


def day_masks(wx: pd.DataFrame) -> dict[str, np.ndarray]:
    m = {name: f(wx).values.astype(bool) for name, f in CATEGORIES.items()}
    severe = np.logical_or.reduce([m[n] for n in SEVERE])
    ordinary = ~np.logical_or.reduce(list(m.values()))
    return {"All days": np.ones(len(wx), bool), "Any severe": severe, "Snow >= 1 in": m["Snow >= 1 in"],
            "Gusts >= 40 mph": m["Gusts >= 40 mph"], "Ordinary days": ordinary}


def score(preds: pd.DataFrame, masks: dict) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = preds["incident_count"].values
    rows = []
    for lname in LEARNERS:
        err_no = np.abs(y - preds[f"{lname}|No weather"].values)
        for v in VERSIONS:
            err = np.abs(y - preds[f"{lname}|{v}"].values)
            gain = err_no - err  # > 0: weather inputs helped that day
            for g, m in masks.items():
                boots = rng.choice(gain[m], size=(N_BOOT, m.sum()), replace=True).mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({"learner": lname, "weather_input": v, "days": g, "n_days": int(m.sum()),
                             "mae_no_weather": err_no[m].mean(), "mae": err[m].mean(),
                             "gain_vs_no_weather": gain[m].mean(), "gain_ci_low": lo, "gain_ci_high": hi,
                             "share_of_observed_gain": np.nan})
    out = pd.DataFrame(rows)
    obs = out[out["weather_input"] == VERSIONS[-1]].set_index(["learner", "days"])["gain_vs_no_weather"]
    out["share_of_observed_gain"] = out["gain_vs_no_weather"] / out.set_index(["learner", "days"]).index.map(obs)
    return out


def event_skill(obs_feats: pd.DataFrame, fc_feats: dict[int, pd.DataFrame], dates: pd.Series) -> pd.DataFrame:
    o = obs_feats[obs_feats["date"].isin(dates)].set_index("date").sort_index()
    rows = []
    for lead, f in fc_feats.items():
        f = f[f["date"].isin(dates)].set_index("date").sort_index().loc[o.index]
        for name, rule in EVENTS.items():
            happened, flagged = rule(o).values, rule(f).values
            hits = int((happened & flagged).sum())
            rows.append({"lead_days": lead, "event": name, "days_observed": int(happened.sum()),
                         "days_forecast": int(flagged.sum()), "hits": hits,
                         "misses": int((happened & ~flagged).sum()), "false_alarms": int((~happened & flagged).sum()),
                         "hit_rate": hits / happened.sum() if happened.any() else np.nan,
                         "false_alarm_ratio": (flagged & ~happened).sum() / flagged.sum() if flagged.any() else np.nan})
        for c in ["tmax_f", "tmin_f", "precip_in", "snow_in", "gust_max_mph"]:
            diff = f[c] - o[c]
            rows.append({"lead_days": lead, "event": f"{c}: mean error (forecast - observed)", "value": diff.mean()})
            rows.append({"lead_days": lead, "event": f"{c}: mean absolute error", "value": diff.abs().mean()})
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame) -> None:
    # Snow >= 1 in (n=7) is left to the table: its intervals are ten times wider and flatten the rest.
    groups = ["All days", "Any severe", "Ordinary days"]
    colors = ["#86b6ef", "#2a78d6", "#104281"]  # one-hue ordinal ramp: less information -> more
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    width = 0.26
    for ax, learner in zip(axes, LEARNERS):
        d = res[res["learner"] == learner]
        for i, (v, c) in enumerate(zip(VERSIONS, colors)):
            s = d[d["weather_input"] == v].set_index("days").loc[groups]
            x = np.arange(len(groups)) + (i - 1) * width
            err = np.vstack([s["gain_vs_no_weather"] - s["gain_ci_low"], s["gain_ci_high"] - s["gain_vs_no_weather"]])
            ax.bar(x, s["gain_vs_no_weather"], width * 0.92, color=c, label=v, yerr=err,
                   error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#555"})
        n = d[d["weather_input"] == VERSIONS[0]].set_index("days").loc[groups, "n_days"]
        ax.set_xticks(range(len(groups)), [f"{g}\n(n={k})" for g, k in zip(groups, n)], fontsize=9)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(learner, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("MAE improvement over the same model\nwithout weather (incidents/day, 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3)
    fig.suptitle("Does weather still help when the model only gets a forecast? (Feb 2024 - Feb 2026)",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    all_cols = [c for c in df.columns if c not in ("date", "incident_count") and pd.api.types.is_numeric_dtype(df[c])]
    weather = set(get_weather_cols(df))
    feature_sets = {"all": all_cols, "no weather": [c for c in all_cols if c not in weather]}

    fc = {lead: forecast_features(df, lead) for lead in (2, 1)}
    inputs = {VERSIONS[0]: fc[2], VERSIONS[1]: fc[1], VERSIONS[2]: df}
    preds = windowed_predictions(df, inputs, feature_sets)

    wx = preds[["date"]].merge(df[["date"] + sorted(weather)], on="date", how="left")
    res = score(preds, day_masks(wx))
    res.to_csv(OUT_CSV, index=False)
    skill = event_skill(df, fc, preds["date"])
    skill.to_csv(OUT_SKILL, index=False)
    plot(res)

    pd.set_option("display.width", 200)
    print(f"\nWeather gain vs no-weather model ({len(preds)} test days):")
    print(res[["learner", "weather_input", "days", "n_days", "mae_no_weather", "mae", "gain_vs_no_weather",
               "gain_ci_low", "gain_ci_high", "share_of_observed_gain"]].to_string(index=False, float_format="%.2f"))
    print("\nForecast skill on observed events:")
    print(skill.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
