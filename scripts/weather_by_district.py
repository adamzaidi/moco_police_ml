# scripts/weather_by_district.py
"""
Does weather move incidents differently across the county's six police districts?

Repeats the rolling-origin ridge weather analysis (same protocol as
severe_weather.py and weather_by_type.py) for each district's daily count, using
that district's own lags and rolling means.

Districts differ in what they handle (Bethesda's calls are 15% alarms against
6-9% elsewhere; violent/person calls range from 9% in Bethesda to 17% in
Gaithersburg). So a district can respond to weather differently just because
its call mix differs. The composition check
separates the two: for each district it predicts the effect from its call mix
and the countywide effect for each dispatch category
(data/weather_effects_by_type.csv), then compares that with the district's
actual effect.

Districts share the same days, so their effects are correlated and comparing
confidence intervals side by side is misleading. The heterogeneity test works on
within-day differences instead: each day, each district's residual minus the
average of the six districts' residuals (all as % of expected), compared with
the same district's average deviation on ordinary days (its baseline). Under
"weather affects every district alike" those differences average zero. A
centered day-level bootstrap gives the p-value for their sum of squares.

Weather is still measured at a single point near the county's center, so
differences between districts also reflect weather varying across the county
(upcounty gets more snow).

Inputs (from make_category_series.py and weather_by_type.py):
    data/daily_counts_by_district.csv, data/district_category_mix.csv, data/weather_effects_by_type.csv
Outputs:
    data/weather_effects_by_district.csv
    data/district_composition_check.csv
    data/district_heterogeneity.csv
    plots/weather_effects_by_district.png

Run from the repo root:  .venv/bin/python -m scripts.weather_by_district
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see forecast_weather.py

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from main import get_weather_cols  # noqa: E402
from scripts.severe_weather import CATEGORIES, N_BOOT, SEVERE  # noqa: E402
from scripts.train_nn import DATA_PATH  # noqa: E402
from scripts.weather_by_type import HEATMAP_WEATHER, LEARNER, run_effects  # noqa: E402

BY_DISTRICT_PATH = BASE_DIR / "data" / "daily_counts_by_district.csv"
MIX_PATH = BASE_DIR / "data" / "district_category_mix.csv"
BY_TYPE_EFFECTS = BASE_DIR / "data" / "weather_effects_by_type.csv"
OUT_CSV = BASE_DIR / "data" / "weather_effects_by_district.csv"
OUT_CHECK = BASE_DIR / "data" / "district_composition_check.csv"
OUT_HET = BASE_DIR / "data" / "district_heterogeneity.csv"
OUT_PLOT = BASE_DIR / "plots" / "weather_effects_by_district.png"


def composition_check(eff: pd.DataFrame) -> pd.DataFrame:
    """Effect each district would show if every category behaved as it does countywide."""
    mix = pd.read_csv(MIX_PATH, index_col=0)
    by_type = pd.read_csv(BY_TYPE_EFFECTS)
    cat_pct = by_type.pivot(index="dispatch_type", columns="category", values="residual_pct_of_expected")
    rows = []
    for district in mix.index.drop("Other", errors="ignore"):
        shares = mix.loc[district]
        for w in HEATMAP_WEATHER:
            predicted = float((shares * cat_pct.loc[shares.index, w]).sum())
            a = eff[(eff["district"] == district) & (eff["category"] == w)].iloc[0]
            rows.append({"district": district, "weather": w, "n_days": a["n_days"],
                         "actual_pct": a["residual_pct_of_expected"], "actual_ci_low": a["pct_ci_low"],
                         "actual_ci_high": a["pct_ci_high"], "predicted_from_mix_pct": predicted,
                         "unexplained_pct": a["residual_pct_of_expected"] - predicted})
    return pd.DataFrame(rows)


def heterogeneity(oos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Do districts' weather effects differ by more than chance? (within-day, centered bootstrap)"""
    rng = np.random.default_rng(0)
    base = pd.read_csv(DATA_PATH, parse_dates=["date"])
    wx = base[["date"] + get_weather_cols(base)]
    districts = [d for d in oos if d != "All incidents"]
    col = f"pred_{LEARNER}_no weather"
    frames = [oos[d][["date", "incident_count", col]].rename(columns={"incident_count": f"y|{d}", col: f"p|{d}"})
              for d in districts]
    m = frames[0]
    for f in frames[1:]:
        m = m.merge(f, on="date")
    m = m.merge(wx, on="date", how="left")

    masks = {name: f(m).values.astype(bool) for name, f in CATEGORIES.items()}
    masks["Any severe (snow, ice, heavy rain, gusts)"] = np.logical_or.reduce([masks[n] for n in SEVERE])
    masks["Ordinary days (none of the above)"] = ~np.logical_or.reduce(
        [v for k, v in masks.items() if not k.startswith("Any severe")])

    def deviations(mask: np.ndarray) -> np.ndarray:
        # per-day residual as % of the district's mean expectation on these days (its mean is the reported
        # effect), minus that day's average over the six districts
        r = np.column_stack([100 * (m.loc[mask, f"y|{d}"] - m.loc[mask, f"p|{d}"]) / m.loc[mask, f"p|{d}"].mean()
                             for d in districts])
        return r - r.mean(axis=1, keepdims=True)

    # Each district's deviation on ordinary days is its baseline (small model drift, not weather), so the
    # test compares weather-day deviations with ordinary-day ones; both sets of days are resampled.
    ordinary = masks["Ordinary days (none of the above)"]
    dev_o = deviations(ordinary)
    base = dev_o.mean(axis=0)
    idx_o = rng.integers(0, len(dev_o), size=(N_BOOT, len(dev_o)))
    boot_o = dev_o[idx_o].mean(axis=1)

    rows = [{"weather": "Ordinary days (baseline)", "n_days": int(ordinary.sum()), "sum_sq_dev": np.nan,
             "p_value": np.nan, "ci_halfwidth_pct": np.nan, **{f"dev|{d}": v for d, v in zip(districts, base)}}]
    for name, mask in masks.items():
        if mask.sum() < 5 or name.startswith("Ordinary"):
            continue
        dev = deviations(mask)
        obs = dev.mean(axis=0) - base
        stat = float((obs ** 2).sum())
        idx = rng.integers(0, len(dev), size=(N_BOOT, len(dev)))
        boot = (dev[idx].mean(axis=1) - boot_o) - obs  # centered: the null of no differences
        p = float(((boot ** 2).sum(axis=1) >= stat).mean())
        # typical 95% CI half-width of one district's deviation: differences much smaller than this are undetectable
        halfwidth = float(np.median(1.96 * boot.std(axis=0)))
        rows.append({"weather": name, "n_days": int(mask.sum()), "sum_sq_dev": stat, "p_value": p,
                     "ci_halfwidth_pct": halfwidth,
                     **{f"dev|{d}": v for d, v in zip(districts, obs)}})
    return pd.DataFrame(rows)


def main() -> None:
    by_d = pd.read_csv(BY_DISTRICT_PATH, parse_dates=["date"]).set_index("date")
    series = {d: by_d[d] for d in by_d.columns if d != "Other"}
    series["All incidents"] = by_d.sum(axis=1)
    eff, oos = run_effects(series, "district", OUT_CSV, OUT_PLOT,
                           "Weather effect by police district (out-of-sample, 2020-2026)", color_lim=15)

    check = composition_check(eff)
    check.to_csv(OUT_CHECK, index=False)
    pd.set_option("display.width", 200)
    print("\nActual vs call-mix-predicted effect (%):")
    print(check.to_string(index=False, float_format="%.1f"))

    het = heterogeneity(oos)
    het.to_csv(OUT_HET, index=False)
    print("\nDo districts differ? (deviation from the six-district average vs ordinary days, % points)")
    print(het.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
