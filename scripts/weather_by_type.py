# scripts/weather_by_type.py
"""
Which kinds of incidents does weather move?

Repeats the severe-weather analysis separately for each dispatch category
(from make_category_series.py). For each category, a ridge model is given that
category's own lags and rolling means plus the shared calendar and holiday
features, and makes rolling-origin out-of-sample predictions with and without
weather (six yearly windows, Feb 2020 - Feb 2026, same protocol as
severe_weather.py). The weather effect for a category is how far its actual
count fell from the weather-blind prediction on days of each weather type.

Outputs:
    data/weather_effects_by_type.csv
    plots/weather_effects_by_type.png

Run from the repo root:  .venv/bin/python -m scripts.weather_by_type
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
from scripts.compare_models import LEARNERS  # noqa: E402
from scripts.severe_weather import category_effects, rolling_predictions  # noqa: E402
from scripts.train_nn import DATA_PATH  # noqa: E402

BY_TYPE_PATH = BASE_DIR / "data" / "daily_counts_by_type.csv"
OUT_CSV = BASE_DIR / "data" / "weather_effects_by_type.csv"
OUT_PLOT = BASE_DIR / "plots" / "weather_effects_by_type.png"
LEARNER = "Ridge regression"

# Weather columns of the heatmap (labels must match severe_weather.CATEGORIES / extra masks).
HEATMAP_WEATHER = [
    "Snow >= 1 in",
    "Day after snow >= 1 in",
    "Ice risk (freeze + precip)",
    "Heavy rain (>= 0.75 in)",
    "Gusts >= 40 mph",
    "Hot (>= 90 F)",
]
SHORT = {
    "Snow >= 1 in": "Snow\n>= 1 in",
    "Day after snow >= 1 in": "Day after\nsnow >= 1 in",
    "Ice risk (freeze + precip)": "Ice\nrisk",
    "Heavy rain (>= 0.75 in)": "Heavy\nrain",
    "Gusts >= 40 mph": "Gusts\n>= 40 mph",
    "Hot (>= 90 F)": "Hot\n>= 90 F",
}


def category_frame(base: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Swap in one category as the target and rebuild its own lag/rolling features (past only)."""
    s = counts.sort_index()
    own = pd.DataFrame({"incident_count": s})
    for lag in (1, 7, 14, 28):
        own[f"lag_{lag}"] = s.shift(lag)
    own["roll_mean_7"] = s.shift(1).rolling(7).mean()
    own["roll_mean_28"] = s.shift(1).rolling(28).mean()
    keep = [c for c in base.columns if c not in own.columns]
    return base[keep].merge(own.reset_index(), on="date", how="left")


def heatmap(eff: pd.DataFrame) -> None:
    order = (eff[eff["category"] == "Snow >= 1 in"].sort_values("residual_pct_of_expected")["dispatch_type"].tolist())
    rows = [t for t in order if t != "All incidents"] + ["All incidents"]
    pct = eff.pivot(index="dispatch_type", columns="category", values="residual_pct_of_expected").loc[rows, HEATMAP_WEATHER]
    lo = eff.pivot(index="dispatch_type", columns="category", values="residual_ci_low").loc[rows, HEATMAP_WEATHER]
    hi = eff.pivot(index="dispatch_type", columns="category", values="residual_ci_high").loc[rows, HEATMAP_WEATHER]
    sig = (lo > 0) | (hi < 0)

    lim = 40
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.8))
    im = ax.imshow(pct.clip(-lim, lim).values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(HEATMAP_WEATHER)):
            v = pct.iat[i, j]
            ax.text(j, i, f"{v:+.0f}%" if sig.iat[i, j] else f"{v:+.0f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if sig.iat[i, j] else "normal",
                    color="white" if abs(v) > 0.6 * lim else "#222" if sig.iat[i, j] else "#888")
    ax.set_xticks(range(len(HEATMAP_WEATHER)), [SHORT[w] for w in HEATMAP_WEATHER], fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.axhline(len(rows) - 1.5, color="#222", lw=1)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.7, label="% vs weather-blind expectation")
    fig.suptitle("Weather effect by dispatch type (out-of-sample, 2020-2026)", x=0.02, ha="left")
    fig.text(0.02, 0.01, "Bold with %: 95% CI excludes zero. Grey: not distinguishable from zero.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    base = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    by_type = pd.read_csv(BY_TYPE_PATH, parse_dates=["date"]).set_index("date")
    weather = sorted(get_weather_cols(base))
    learners = {LEARNER: LEARNERS[LEARNER]}

    series = {t: by_type[t] for t in by_type.columns}
    series["All incidents"] = by_type.drop(columns=["Crash report filed"]).sum(axis=1)

    effects = []
    for name, counts in series.items():
        df = category_frame(base, counts).dropna().reset_index(drop=True)
        all_cols = [c for c in df.columns if c not in ("date", "incident_count") and pd.api.types.is_numeric_dtype(df[c])]
        feature_sets = {"all": all_cols, "no weather": [c for c in all_cols if c not in set(weather)]}
        oos, roll = rolling_predictions(df, feature_sets, learners=learners, verbose=False)
        eff = category_effects(oos, df[["date"] + weather], LEARNER)
        eff.insert(0, "dispatch_type", name)
        eff.insert(1, "mean_daily_count", counts.mean())
        effects.append(eff)
        snow = eff[eff["category"] == "Snow >= 1 in"].iloc[0]
        gust = eff[eff["category"] == "Gusts >= 40 mph"].iloc[0]
        print(f"  {name:26s} mean/day {counts.mean():6.1f} | snow>=1in {snow['residual_pct_of_expected']:+6.1f}% "
              f"| gusts {gust['residual_pct_of_expected']:+6.1f}% | rolling MAE "
              f"{roll[roll['features'] == 'all']['mae'].mean():.1f}", flush=True)

    out = pd.concat(effects, ignore_index=True)
    # CI bounds as % of expected (expected = actual mean - residual)
    expected = out["mean_actual"] - out["residual_no_weather"]
    out["pct_ci_low"] = 100 * out["residual_ci_low"] / expected
    out["pct_ci_high"] = 100 * out["residual_ci_high"] / expected
    out.to_csv(OUT_CSV, index=False)

    heatmap(out)
    tbl = out[out["category"].isin(HEATMAP_WEATHER)].pivot(index="dispatch_type", columns="category",
                                                           values="residual_pct_of_expected")[HEATMAP_WEATHER]
    print("\n% effect vs weather-blind expectation:")
    print(tbl.round(1).to_string())


if __name__ == "__main__":
    main()
