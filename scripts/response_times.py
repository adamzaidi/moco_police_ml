# scripts/response_times.py
"""
Does weather, or a busy day, slow police response?

Response time = call received to first unit on scene ('Calltime Arrive'), for
calls where a unit arrived (values from 1 s to 4 h). Priority 0 is the most
urgent (in-progress assaults, domestic violence, crashes with injuries); priority
1 is the next tier (suspicious persons, welfare checks, burglar alarms).

For each priority, the daily median response time is regressed (in logs) on:
  - the weather categories used elsewhere in the project,
  - that day's total incident count (log, so the coefficient is an elasticity),
  - day of week and calendar month (year-month fixed effects, which absorb slow
    changes such as staffing and dispatch-system changes).
CIs: bootstrap over whole weeks (2000 draws), since nearby days share conditions.

Inputs:  data/incidents_slim.pkl, data/features_daily.csv
Outputs: data/response_time_effects.csv, data/response_time_monthly.csv, plots/response_times.png
Run from the repo root:  .venv/bin/python -m scripts.response_times
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.severe_weather import CATEGORIES  # noqa: E402
from scripts.train_nn import DATA_PATH  # noqa: E402
from scripts.weather_by_type import HEATMAP_WEATHER, SHORT  # noqa: E402

SLIM_PATH = BASE_DIR / "data" / "incidents_slim.pkl"
OUT_CSV = BASE_DIR / "data" / "response_time_effects.csv"
OUT_MONTHLY = BASE_DIR / "data" / "response_time_monthly.csv"
OUT_PLOT = BASE_DIR / "plots" / "response_times.png"
PRIORITIES = {0: "Priority 0 (most urgent)", 1: "Priority 1"}
MIN_CALLS = 10
N_BOOT = 2000


def daily_medians(slim: pd.DataFrame) -> pd.DataFrame:
    s = slim[slim["call_to_arrive_s"].between(1, 4 * 3600)].copy()
    s["date"] = s["time"].dt.normalize()
    out = {}
    for p, label in PRIORITIES.items():
        g = s[s["priority"] == p].groupby("date")["call_to_arrive_s"].agg(["median", "size"])
        out[label] = g.loc[g["size"] >= MIN_CALLS, "median"] / 60
    return pd.DataFrame(out)


def design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = {w: CATEGORIES[w](df).astype(float).values for w in HEATMAP_WEATHER}
    cols["log incident count"] = np.log(df["incident_count"]).values
    X = pd.DataFrame(cols, index=df.index)
    fe = pd.get_dummies(df["date"].dt.to_period("M").astype(str), prefix="m", dtype=float)
    dow = pd.get_dummies(df["date"].dt.day_of_week, prefix="dow", drop_first=True, dtype=float)
    return np.column_stack([X.values, fe.values, dow.values]), list(X.columns)


def main() -> None:
    # incidents_slim.pkl is written locally by make_incident_extract.py, never downloaded
    slim = pd.read_pickle(SLIM_PATH)
    med = daily_medians(slim)
    feats = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    monthly = med.resample("MS").median()
    monthly.to_csv(OUT_MONTHLY, float_format="%.2f")

    rng = np.random.default_rng(0)
    rows = []
    for label in PRIORITIES.values():
        df = feats.merge(med[[label]].rename(columns={label: "resp"}), left_on="date", right_index=True)
        df = df.dropna(subset=["resp"]).reset_index(drop=True)
        X, names = design(df)
        y = np.log(df["resp"].values)
        beta = np.linalg.lstsq(X, y, rcond=None)[0][:len(names)]
        weeks = df["date"].dt.to_period("W").astype(str).to_numpy()
        uw = np.unique(weeks)
        rows_by_week = {w: np.where(weeks == w)[0] for w in uw}
        draws = np.empty((N_BOOT, len(names)))
        for b in range(N_BOOT):
            idx = np.concatenate([rows_by_week[w] for w in rng.choice(uw, size=len(uw))])
            draws[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][:len(names)]
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        base = df["resp"].median()
        for i, n in enumerate(names):
            is_vol = n.startswith("log")
            f = (lambda v: 100 * (np.exp(v * np.log(1.1)) - 1)) if is_vol else (lambda v: 100 * np.expm1(v))
            rows.append({"priority": label, "factor": "10% more incidents that day" if is_vol else n,
                         "n_days": int(len(df)) if is_vol else int(X[:, i].sum()),
                         "pct_change": f(beta[i]), "ci_low": f(lo[i]), "ci_high": f(hi[i]),
                         "median_response_min": base})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(monthly.resample("YS").median().round(1).to_string())
    print(out.to_string(index=False, float_format="%.1f"))
    plot(monthly, out)


def plot(monthly: pd.DataFrame, eff: pd.DataFrame) -> None:
    colors = {"Priority 0 (most urgent)": "#104281", "Priority 1": "#86b6ef"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    for label, c in colors.items():
        a1.plot(monthly.index, monthly[label], color=c, lw=2, label=label)
    a1.set_ylabel("Median minutes, call to first unit on scene")
    a1.set_ylim(0, None)
    a1.set_title("Response time by month", fontsize=10.5, loc="left")
    a1.legend(frameon=False, fontsize=9, loc="lower left")
    a1.spines[["top", "right"]].set_visible(False)

    factors = HEATMAP_WEATHER + ["10% more incidents that day"]
    labels = [SHORT.get(f, "10% more\nincidents").replace("\n", " ") for f in factors]
    y = np.arange(len(factors))[::-1]
    for k, (label, c) in enumerate(colors.items()):
        d = eff[eff["priority"] == label].set_index("factor").loc[factors]
        yy = y + (0.17 if k == 0 else -0.17)
        a2.errorbar(d["pct_change"], yy, xerr=[d["pct_change"] - d["ci_low"], d["ci_high"] - d["pct_change"]],
                    fmt="o", color=c, ms=6, capsize=2, lw=1.5, label=label)
    a2.axvline(0, color="#333", lw=0.8)
    a2.set_yticks(y, labels, fontsize=9)
    a2.set_xlabel("% change in median response time (95% CI)")
    a2.set_title("Same month and weekday, what changes it", fontsize=10.5, loc="left")
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Police response times, Montgomery County 2017-2026", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
