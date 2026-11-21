# main.py
"""
Weather feature ablation, run without touching the test set.

Stage 1 (selection, VALIDATION only):
    Train the baseline model and one model per dropped feature / feature group,
    each over several seeds. Rank runs by validation MAE. The test set is never
    scored here, so nothing about it can influence which features are kept.

Stage 2 (one-time TEST evaluation):
    Score exactly two configurations on the test set: the all-features baseline
    and the single configuration with the best validation MAE. Report them next
    to the naive baselines (yesterday's count, same day last week).

Outputs:
    data/ablation_results.csv   validation-only ranking of every run
    data/final_test_results.csv test MAE for the two chosen configs + naive baselines

Caveat: validation is also used for early stopping, so validation MAE is
slightly optimistic. That is why selection is confirmed on the untouched test
year instead of trusted on its own.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features_daily.csv"
ABLATION_PATH = BASE_DIR / "data" / "ablation_results.csv"
FINAL_PATH = BASE_DIR / "data" / "final_test_results.csv"

SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 160
PATIENCE = 12
# A drop counts as a real change only if its mean paired difference from the
# baseline (same seeds) is larger than this many standard errors.
NOISE_SE = 2.0


def now() -> str:
    return time.strftime("%H:%M:%S")


def get_weather_cols(df: pd.DataFrame) -> List[str]:
    prefixes = ("tmax_", "tmin_", "tmean_", "precip_", "rain_", "snow_", "wind_", "gust_")
    exact = {
        "tmax_f", "tmin_f", "tmean_f",
        "precip_in", "rain_in", "snow_in",
        "precip_hours", "wind_max_mph", "gust_max_mph",
        "temp_anom_30", "precip_anom_30",
        "is_rainy", "is_heavy_rain", "is_snow_day", "is_windy", "is_hot_day", "is_freeze_night",
        # compound danger proxies + past-3-day streak counts
        "is_ice_risk", "is_blowing_snow", "is_stormy",
        "snow_days_last3", "heavy_snow_last3", "heavy_rain_last3", "ice_risk_last3",
    }
    out: List[str] = []
    for c in df.columns:
        if c in ("date", "incident_count") or c in out:
            continue
        if c in exact or c.startswith(prefixes):
            out.append(c)
    return out


def define_weather_groups(weather_cols: List[str]) -> Dict[str, List[str]]:
    def keep(cols: List[str]) -> List[str]:
        return [c for c in cols if c in weather_cols]

    groups = {
        # Every weather column at once: does weather help at all?
        "drop_weather_all": list(weather_cols),
        # Raw continuous measurements
        "drop_weather_temp_raw": keep(["tmax_f", "tmin_f", "tmean_f"]),
        "drop_weather_precip_raw": keep(["precip_in", "rain_in", "precip_hours"]),
        "drop_weather_snow_raw": keep(["snow_in"]),
        "drop_weather_wind_raw": keep(["wind_max_mph", "gust_max_mph"]),
        "drop_weather_anoms": keep(["temp_anom_30", "precip_anom_30"]),
        # Threshold flags
        "drop_weather_flags_all": keep(["is_rainy", "is_heavy_rain", "is_snow_day", "is_windy", "is_hot_day", "is_freeze_night"]),
        "drop_weather_flags_snow": keep(["is_snow_day", "is_freeze_night"]),
        "drop_weather_flags_rain": keep(["is_rainy", "is_heavy_rain"]),
        "drop_weather_flags_wind": keep(["is_windy"]),
        "drop_weather_flags_heat": keep(["is_hot_day"]),
        # Compound danger proxies and recent-weather streaks
        "drop_weather_compound": keep(["is_ice_risk", "is_blowing_snow", "is_stormy"]),
        "drop_weather_streaks": keep(["snow_days_last3", "heavy_snow_last3", "heavy_rain_last3", "ice_risk_last3"]),
        # Every severe-weather indicator: flags, heavy tiers, compound, streaks
        "drop_weather_severe_all": keep([
            "is_heavy_rain", "is_snow_day", "is_windy", "is_hot_day", "is_freeze_night",
            "snow_mod", "snow_heavy", "rain_heavy", "wind_windy", "wind_gale",
            "is_ice_risk", "is_blowing_snow", "is_stormy",
            "snow_days_last3", "heavy_snow_last3", "heavy_rain_last3", "ice_risk_last3",
        ]),
    }
    return {k: v for k, v in groups.items() if v}


def run_seeds(train_eval, drop_cols: Optional[List[str]], evaluate_test: bool) -> Dict[int, Dict[str, float]]:
    out = {}
    for s in SEEDS:
        r = train_eval(
            data_path=DATA_PATH,
            drop_cols=drop_cols,
            seed=s,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            make_plots=False,
            verbose=False,
            evaluate_test=evaluate_test,
            save_outputs=False,
        )
        out[s] = {"val": float(r["best_val_mae"]), "test": r["test_mae"]}
    return out


def summarize(name: str, drop_cols: Optional[List[str]], per_seed, base_per_seed) -> Dict[str, object]:
    vals = np.array([per_seed[s]["val"] for s in SEEDS])
    row: Dict[str, object] = {
        "run": name,
        "n_drop": len(drop_cols) if drop_cols else 0,
        "val_mae_mean": vals.mean(),
        "val_mae_std": vals.std(ddof=1),
    }
    if base_per_seed is not None:
        diffs = vals - np.array([base_per_seed[s]["val"] for s in SEEDS])
        se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        row["val_delta_vs_baseline"] = diffs.mean()
        row["val_delta_se"] = se
        row["beyond_noise"] = bool(se > 0 and abs(diffs.mean()) > NOISE_SE * se)
    row["drop_cols"] = ",".join(drop_cols) if drop_cols else ""
    return row


def naive_test_maes() -> Dict[str, float]:
    """Yesterday / same-day-last-week baselines on the same test window as train_nn."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    test = df[df["date"] > df["date"].max() - pd.Timedelta(days=365)]
    y = test["incident_count"]
    return {
        "naive_yesterday": float((y - test["lag_1"]).abs().mean()),
        "naive_last_week": float((y - test["lag_7"]).abs().mean()),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run: python scripts/build_features.py")

    from scripts.train_nn import train_eval

    weather_cols = get_weather_cols(pd.read_csv(DATA_PATH, nrows=1))
    groups = define_weather_groups(weather_cols)
    plan: List[tuple] = list(groups.items()) + [(f"drop_{c}", [c]) for c in weather_cols]
    print(f"[{now()}] {len(weather_cols)} weather cols, {len(plan) + 1} runs x {len(SEEDS)} seeds", flush=True)

    # ---------------- Stage 1: selection on validation ----------------
    t0 = time.time()
    base = run_seeds(train_eval, None, evaluate_test=False)
    rows = [summarize("baseline_all_features", None, base, None)]
    rows[0].update(val_delta_vs_baseline=0.0, val_delta_se=0.0, beyond_noise=False)
    print(f"[{now()}] baseline val MAE {rows[0]['val_mae_mean']:.2f}", flush=True)

    for i, (name, drop) in enumerate(plan, start=1):
        try:
            per_seed = run_seeds(train_eval, drop, evaluate_test=False)
        except Exception:
            print(f"ERROR in {name}; skipping", flush=True)
            traceback.print_exc()
            continue
        row = summarize(name, drop, per_seed, base)
        rows.append(row)
        flag = " *" if row["beyond_noise"] else ""
        print(f"[{now()}] {i}/{len(plan)} {name:32s} val {row['val_mae_mean']:.2f}  "
              f"delta {row['val_delta_vs_baseline']:+.2f} +/- {row['val_delta_se']:.2f}{flag}", flush=True)

    abl = pd.DataFrame(rows).sort_values("val_mae_mean").reset_index(drop=True)
    abl.to_csv(ABLATION_PATH, index=False)
    print(f"\n[{now()}] stage 1 done in {time.time() - t0:.0f}s -> {ABLATION_PATH}", flush=True)
    print(abl[["run", "val_mae_mean", "val_delta_vs_baseline", "val_delta_se", "beyond_noise"]]
          .to_string(index=False, float_format="%.2f"), flush=True)

    # ---------------- Stage 2: one-time test evaluation ----------------
    best = abl.iloc[0]
    finalists = [("baseline_all_features", None)]
    if best["run"] != "baseline_all_features":
        finalists.append((best["run"], best["drop_cols"].split(",")))

    final_rows = []
    for name, drop in finalists:
        per_seed = run_seeds(train_eval, drop, evaluate_test=True)
        tests = np.array([per_seed[s]["test"] for s in SEEDS])
        final_rows.append({
            "model": f"nn:{name}",
            "test_mae_mean": tests.mean(),
            "test_mae_std": tests.std(ddof=1),
            "val_mae_mean": np.mean([per_seed[s]["val"] for s in SEEDS]),
        })
    for name, mae in naive_test_maes().items():
        final_rows.append({"model": name, "test_mae_mean": mae, "test_mae_std": 0.0, "val_mae_mean": np.nan})

    final = pd.DataFrame(final_rows)
    final.to_csv(FINAL_PATH, index=False)
    print(f"\n=== ONE-TIME TEST EVALUATION ({len(SEEDS)} seeds) -> {FINAL_PATH} ===", flush=True)
    print(final.to_string(index=False, float_format="%.2f"), flush=True)


if __name__ == "__main__":
    main()
