# scripts/compare_models.py
"""
Is the neural network earning its complexity?

Compares the NN against naive rules and two standard learners on the same time
split and the same two feature sets (all features, no weather). Every model is
fixed before the test year is scored:

  - naive rules have no parameters
  - ridge and gradient boosting choose hyperparameters on VALIDATION only
  - the NN uses train_eval's existing setup (early stopping on validation), 5 seeds

Each model is then scored once on test. No model or setting is changed after
looking at test numbers.

Outputs:
    data/model_comparison.csv
    plots/model_comparison.png
    data/ridge_weather_ablation.csv  weather-group ablation repeated with ridge (validation only)

Run from the repo root:  .venv/bin/python -m scripts.compare_models
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from main import define_weather_groups, get_weather_cols  # noqa: E402
from scripts.train_nn import DATA_PATH, time_split, train_eval, transform_weather  # noqa: E402

OUT_CSV = BASE_DIR / "data" / "model_comparison.csv"
OUT_PLOT = BASE_DIR / "plots" / "model_comparison.png"
OUT_RIDGE_ABL = BASE_DIR / "data" / "ridge_weather_ablation.csv"
NN_SEEDS = [0, 1, 2, 3, 4]


def mae(y, p) -> float:
    return float(mean_absolute_error(y, p))


def naive_rows(val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rules = {
        "Yesterday's count": lambda d: d["lag_1"],
        "Same day last week": lambda d: d["lag_7"],
        "Mean of last 7 days": lambda d: d["roll_mean_7"],
        "Mean of same weekday, last 4 weeks": lambda d: d[["lag_7", "lag_14", "lag_28"]].mean(axis=1),
    }
    rows = []
    for name, f in rules.items():
        rows.append({
            "model": name, "family": "naive", "features": "-", "chosen_params": "",
            "val_mae": mae(val["incident_count"], f(val)),
            "test_mae": mae(test["incident_count"], f(test)), "test_mae_sd": 0.0,
        })
    return rows


def fit_select(make_model, grid: dict, Xtr, ytr, Xva, yva):
    """Try every grid point, fit on train, keep the one with the lowest validation MAE."""
    best = None
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        m = make_model(**params).fit(Xtr, ytr)
        v = mae(yva, m.predict(Xva))
        if best is None or v < best[0]:
            best = (v, params, m)
    return best


def learned_rows(train, val, test, feature_sets: dict) -> list[dict]:
    learners = {
        "Ridge regression": (
            lambda alpha: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
            {"alpha": [0.1, 1, 10, 100, 1000]},
        ),
        "Gradient boosting": (
            lambda loss, learning_rate, max_leaf_nodes: HistGradientBoostingRegressor(
                loss=loss, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                max_iter=300, random_state=0),
            {"loss": ["squared_error", "absolute_error"], "learning_rate": [0.03, 0.1],
             "max_leaf_nodes": [15, 31]},
        ),
    }
    rows = []
    for (lname, (make_model, grid)), (fname, cols) in itertools.product(learners.items(), feature_sets.items()):
        X = {k: transform_weather(d[cols]) for k, d in (("tr", train), ("va", val), ("te", test))}
        v, params, model = fit_select(make_model, grid, X["tr"], train["incident_count"], X["va"], val["incident_count"])
        rows.append({
            "model": lname, "family": "learned", "features": fname,
            "chosen_params": ",".join(f"{k}={p}" for k, p in params.items()),
            "val_mae": v, "test_mae": mae(test["incident_count"], model.predict(X["te"])), "test_mae_sd": 0.0,
        })
        print(f"  {lname:20s} {fname:11s} val {v:6.2f}  ({rows[-1]['chosen_params']})", flush=True)
    return rows


def ridge_weather_ablation(train, val, all_cols: list[str], weather: list[str], alpha: float) -> pd.DataFrame:
    """Repeat the weather-group ablation with ridge (deterministic, validation only).

    Separates what the data says about weather from how the NN handles it.
    """
    def val_mae(cols):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(transform_weather(train[cols]), train["incident_count"])
        return mae(val["incident_count"], m.predict(transform_weather(val[cols])))

    base = val_mae(all_cols)
    rows = [{"run": "baseline_all_features", "n_drop": 0, "val_mae": base, "val_delta_vs_baseline": 0.0}]
    for name, drop in define_weather_groups(weather).items():
        v = val_mae([c for c in all_cols if c not in set(drop)])
        rows.append({"run": name, "n_drop": len(drop), "val_mae": v, "val_delta_vs_baseline": v - base})
    return pd.DataFrame(rows).sort_values("val_delta_vs_baseline", ascending=False).reset_index(drop=True)


def nn_rows(feature_sets: dict, all_cols: list[str]) -> list[dict]:
    rows = []
    for fname, cols in feature_sets.items():
        drop = [c for c in all_cols if c not in cols] or None
        runs = [train_eval(drop_cols=drop, seed=s, max_epochs=160, patience=12, make_plots=False,
                           verbose=False, evaluate_test=True, save_outputs=False) for s in NN_SEEDS]
        tests = np.array([r["test_mae"] for r in runs])
        rows.append({
            "model": "Neural network", "family": "learned", "features": fname,
            "chosen_params": f"{len(NN_SEEDS)} seeds",
            "val_mae": float(np.mean([r["best_val_mae"] for r in runs])),
            "test_mae": float(tests.mean()), "test_mae_sd": float(tests.std(ddof=1)),
        })
        print(f"  {'Neural network':20s} {fname:11s} val {rows[-1]['val_mae']:6.2f}", flush=True)
    return rows


def plot(df: pd.DataFrame) -> None:
    d = df.sort_values("test_mae", ascending=False)
    labels = [m if f == "-" else f"{m} ({f})" for m, f in zip(d["model"], d["features"])]
    colors = ["#9aa5b1" if fam == "naive" else "#3b6ea5" for fam in d["family"]]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1))
    ax.barh(labels, d["test_mae"], xerr=d["test_mae_sd"], color=colors)
    for y, (v, sd) in enumerate(zip(d["test_mae"], d["test_mae_sd"])):
        ax.text(v + sd + 0.5, y, f"{v:.1f}", va="center", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#3b6ea5", label="learned model"), Patch(color="#9aa5b1", label="naive rule")],
              loc="upper right", frameon=False)
    ax.set_xlabel("Test MAE (daily incidents, lower is better)")
    ax.set_title("Daily incident forecasts, test year Feb 2025 - Feb 2026")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    all_cols = [c for c in df.columns if c not in ("date", "incident_count") and pd.api.types.is_numeric_dtype(df[c])]
    weather = set(get_weather_cols(df))
    feature_sets = {"all": all_cols, "no weather": [c for c in all_cols if c not in weather]}

    train, val, test = time_split(df)
    # Learned models can't use rows whose lag/rolling features aren't defined yet.
    train = train.dropna(subset=all_cols)
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | "
          f"features all={len(all_cols)} no-weather={len(feature_sets['no weather'])}", flush=True)

    rows = naive_rows(val, test) + learned_rows(train, val, test, feature_sets) + nn_rows(feature_sets, all_cols)
    out = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    plot(out)

    ridge_all = out[(out["model"] == "Ridge regression") & (out["features"] == "all")].iloc[0]
    alpha = float(ridge_all["chosen_params"].split("=")[1])
    abl = ridge_weather_ablation(train, val, all_cols, sorted(weather), alpha)
    abl.to_csv(OUT_RIDGE_ABL, index=False)
    print(f"\nRidge weather-group ablation (alpha={alpha:g}, validation) -> {OUT_RIDGE_ABL}")
    print(abl.to_string(index=False, float_format="%.2f"))

    print(f"\nSorted by VALIDATION MAE (the selection criterion) -> {OUT_CSV}")
    print(out[["model", "features", "val_mae", "test_mae", "test_mae_sd", "chosen_params"]]
          .to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
