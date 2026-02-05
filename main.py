# main.py
from __future__ import annotations

import os
import sys
import time
import traceback
import faulthandler
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

print("main.py started", flush=True)
faulthandler.enable()  # if it hard-hangs, you can still dump stack traces

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "features_daily.csv"
OUT_PATH = BASE_DIR / "data" / "ablation_results.csv"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_weather_cols(df: pd.DataFrame) -> List[str]:
    prefixes = ("tmax_", "tmin_", "tmean_", "precip_", "rain_", "snow_", "wind_", "gust_")
    exact = {
        "tmax_f", "tmin_f", "tmean_f",
        "precip_in", "rain_in", "snow_in",
        "precip_hours", "wind_max_mph", "gust_max_mph",
        "temp_anom_30", "precip_anom_30",
        "is_rainy", "is_heavy_rain", "is_snow_day", "is_windy", "is_hot_day", "is_freeze_night",
    }

    cols = []
    for c in df.columns:
        if c in ("date", "incident_count"):
            continue
        if c in exact:
            cols.append(c)
            continue
        if c.startswith(prefixes):
            cols.append(c)
            continue

    # de-dup, keep stable order
    seen = set()
    out = []
    for c in cols:
        if c not in seen and c in df.columns:
            out.append(c)
            seen.add(c)
    return out


def define_weather_groups(weather_cols: List[str]) -> Dict[str, List[str]]:
    def keep(existing: List[str]) -> List[str]:
        return [c for c in existing if c in weather_cols]

    groups = {
        # Raw continuous-ish
        "drop_weather_temp_raw": keep(["tmax_f", "tmin_f", "tmean_f"]),
        "drop_weather_precip_raw": keep(["precip_in", "rain_in", "precip_hours"]),
        "drop_weather_snow_raw": keep(["snow_in"]),
        "drop_weather_wind_raw": keep(["wind_max_mph", "gust_max_mph"]),
        "drop_weather_anoms": keep(["temp_anom_30", "precip_anom_30"]),

        # Flags
        "drop_weather_flags_all": keep(["is_rainy", "is_heavy_rain", "is_snow_day", "is_windy", "is_hot_day", "is_freeze_night"]),
        "drop_weather_flags_snow": keep(["is_snow_day", "is_freeze_night"]),
        "drop_weather_flags_rain": keep(["is_rainy", "is_heavy_rain"]),
        "drop_weather_flags_wind": keep(["is_windy"]),
        "drop_weather_flags_heat": keep(["is_hot_day"]),
    }

    return {k: v for k, v in groups.items() if len(v) > 0}


def append_partial_csv(rows: List[Dict[str, object]]) -> None:
    """Write partial results frequently so you can see progress even if you ctrl+c."""
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"[{now()}] wrote partial results → {OUT_PATH} (rows={len(out_df)})", flush=True)


def run_one_experiment(
    train_eval_fn,
    name: str,
    drop_cols: List[str] | None,
    seeds: List[int],
    exp_idx: int,
    exp_total: int,
    max_epochs: int = 160,
    patience: int = 12,
) -> Dict[str, object]:
    maes = []
    t0 = time.time()

    print(f"\n[{now()}] EXP {exp_idx}/{exp_total} START: {name} | n_drop={0 if not drop_cols else len(drop_cols)}", flush=True)
    if drop_cols:
        print(f"  drop_cols={drop_cols}", flush=True)

    for si, s in enumerate(seeds, start=1):
        print(f"  [{now()}] seed {si}/{len(seeds)} = {s} ...", flush=True)
        seed_t0 = time.time()

        r = train_eval_fn(
            data_path=DATA_PATH,
            drop_cols=drop_cols,
            seed=s,
            max_epochs=max_epochs,
            patience=patience,
            make_plots=False,     # keep ablations fast/clean
            plot_suffix="",
            verbose=False,        # silence epochs; main.py prints progress instead
        )

        mae = float(r["test_mae"])
        maes.append(mae)
        print(f"  [{now()}] seed {s} DONE | test_mae={mae:.4f} | dt={time.time()-seed_t0:.1f}s", flush=True)

    dt = time.time() - t0
    out = {
        "run": name,
        "drop_cols": ",".join(drop_cols) if drop_cols else "",
        "n_drop": len(drop_cols) if drop_cols else 0,
        "seeds": ",".join(map(str, seeds)),
        "test_mae_mean": float(np.mean(maes)),
        "test_mae_std": float(np.std(maes)),
        "wall_seconds": float(dt),
    }
    print(f"[{now()}] EXP {exp_idx}/{exp_total} END: {name} | mean={out['test_mae_mean']:.4f} std={out['test_mae_std']:.4f} | dt={dt:.1f}s", flush=True)
    return out


def main():
    print(f"[{now()}] main() entered", flush=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run: python scripts/build_features.py")

    # Import train_eval *inside* main so we can see if import is where it hangs
    print(f"[{now()}] importing train_eval from scripts.train_nn ...", flush=True)
    try:
        from scripts.train_nn import train_eval
    except Exception:
        print("ERROR importing train_eval:", flush=True)
        traceback.print_exc()
        raise
    print(f"[{now()}] import OK", flush=True)

    # Load columns (nrows=1 is enough; header contains all cols)
    df_cols = pd.read_csv(DATA_PATH, nrows=1)
    print(f"[{now()}] features_daily.csv columns={len(df_cols.columns)}", flush=True)

    weather_cols = get_weather_cols(df_cols)
    groups = define_weather_groups(weather_cols)

    print(f"[{now()}] detected weather_cols={len(weather_cols)}", flush=True)
    print(f"[{now()}] detected weather_groups={len(groups)}", flush=True)

    # Seeds: set to [0] temporarily if you just want “does it run?”
    seeds = [0, 1, 2]

    # Build experiment plan
    plan: List[tuple[str, List[str] | None]] = []
    plan.append(("baseline_all_features", None))
    for name, drop in groups.items():
        plan.append((name, drop))
    for c in weather_cols:
        plan.append((f"drop_{c}", [c]))

    print(f"[{now()}] total experiments={len(plan)} | total trainings={len(plan) * len(seeds)}", flush=True)
    print(f"[{now()}] NOTE: if this seems slow, set seeds=[0] or reduce plan temporarily.", flush=True)

    results: List[Dict[str, object]] = []
    baseline_mae = None

    for i, (name, drop_cols) in enumerate(plan, start=1):
        try:
            row = run_one_experiment(
                train_eval_fn=train_eval,
                name=name,
                drop_cols=drop_cols,
                seeds=seeds,
                exp_idx=i,
                exp_total=len(plan),
                max_epochs=160,
                patience=12,
            )
            results.append(row)

            # Save partial results each experiment
            append_partial_csv(results)

            if name == "baseline_all_features":
                baseline_mae = row["test_mae_mean"]

        except KeyboardInterrupt:
            print("\nCTRL+C received. Writing partial results and exiting...", flush=True)
            append_partial_csv(results)
            raise
        except Exception:
            print(f"\nERROR in experiment {name}. Writing partial results and continuing...", flush=True)
            traceback.print_exc()
            append_partial_csv(results)
            # continue to next experiment
            continue

    out = pd.DataFrame(results)

    if baseline_mae is None:
        print("WARNING: baseline did not run; delta_vs_baseline will be NaN", flush=True)
        out["delta_vs_baseline"] = np.nan
    else:
        out["delta_vs_baseline"] = out["test_mae_mean"] - float(baseline_mae)

    out = out.sort_values("delta_vs_baseline", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print("\nSaved:", OUT_PATH, flush=True)
    print("\nTop 15 WORST when dropped (i.e., most helpful features/groups):", flush=True)
    print(out.head(15)[["run", "test_mae_mean", "test_mae_std", "delta_vs_baseline", "n_drop"]].to_string(index=False), flush=True)

    print("\nTop 15 BEST when dropped (i.e., likely noise / harmful):", flush=True)
    print(out.tail(15)[["run", "test_mae_mean", "test_mae_std", "delta_vs_baseline", "n_drop"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    # Force unbuffered prints even if user forgets -u
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()