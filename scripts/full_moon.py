# scripts/full_moon.py
"""
Is there a full-moon effect? (Officers widely believe it; studies keep finding none.)

Full-moon days = the local date of the full moon plus the day before and after;
new-moon days likewise. Two tests:

1. Whole days, 2020-2026: the rolling-origin ridge model's out-of-sample residual
   (severe_weather.py, all features incl. weather) on full-moon days vs all other
   days. The moon isn't in the model, so any effect would show up as a residual.
2. Nights, 2017-2026, paired by lunar cycle: calls between 20:00 and 04:00 relative
   to the same date's daytime calls (10:00-18:00), on full-moon nights vs the
   new-moon nights of the same cycle. Comparing night with day on the same date
   removes anything that moves the whole day (weekday, weather, season); pairing
   within a cycle keeps the two moons two weeks apart.
CIs: bootstrap over days (test 1) or lunar cycles (test 2). The CI half-width is
the smallest effect each test could have detected.

Inputs:  data/oos_predictions.csv, data/incidents_slim.pkl
Outputs: data/full_moon_effects.csv
Run from the repo root:  .venv/bin/python -m scripts.full_moon
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.astro import TZ, full_moon_instants, new_moon_instants  # noqa: E402

OOS_PATH = BASE_DIR / "data" / "oos_predictions.csv"
SLIM_PATH = BASE_DIR / "data" / "incidents_slim.pkl"
OUT_CSV = BASE_DIR / "data" / "full_moon_effects.csv"
N_BOOT = 2000
NIGHT_GROUPS = ["All incidents", "Violent / person", "Disorder / suspicious", "Welfare / mental health"]


def moon_days(instants: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per moon day: date, cycle id (the moon's local date), offset -1/0/+1."""
    local = instants.tz_convert(TZ).tz_localize(None).normalize()
    rows = [{"date": d + pd.Timedelta(days=k), "cycle": d} for d in local for k in (-1, 0, 1)]
    return pd.DataFrame(rows)


def whole_day_test(full: pd.DataFrame, rng) -> dict:
    o = pd.read_csv(OOS_PATH, parse_dates=["date"])
    p = o["pred_Ridge regression_all"]
    r = (100 * (o["incident_count"] - p) / p).values
    m = o["date"].isin(set(full["date"])).values
    diff = r[m].mean() - r[~m].mean()
    boot = [rng.choice(r[m], m.sum()).mean() - rng.choice(r[~m], (~m).sum()).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"test": "Whole day, vs model expectation (2020-2026)", "group": "All incidents",
            "n": f"{int(m.sum())} full-moon days", "effect_pct": diff, "ci_low": lo, "ci_high": hi}


def night_test(full: pd.DataFrame, new: pd.DataFrame, rng) -> list[dict]:
    # incidents_slim.pkl is written locally by make_incident_extract.py, never downloaded
    s = pd.read_pickle(SLIM_PATH)
    h = s["time"].dt.hour
    # a night belongs to the date it starts on: 20:00-23:59 that date, 00:00-03:59 the next morning
    s["night_of"] = np.where(h >= 20, s["time"].dt.normalize(),
                             np.where(h < 4, s["time"].dt.normalize() - pd.Timedelta(days=1), pd.NaT))
    s["day_of"] = np.where((h >= 10) & (h < 18), s["time"].dt.normalize(), pd.NaT)
    first, last = s["time"].min().normalize() + pd.Timedelta(days=1), s["time"].max().normalize() - pd.Timedelta(days=1)
    rows = []
    for g in NIGHT_GROUPS:
        sub = s if g == "All incidents" else s[s["category"] == g]
        night = sub.dropna(subset=["night_of"]).groupby("night_of").size()
        day = sub.dropna(subset=["day_of"]).groupby("day_of").size()

        def per_cycle(md: pd.DataFrame) -> pd.Series:
            md = md[(md["date"] >= first) & (md["date"] <= last)]
            n = night.reindex(md["date"]).fillna(0).values
            d = day.reindex(md["date"]).fillna(0).values
            agg = pd.DataFrame({"cycle": md["cycle"].values, "n": n, "d": d}).groupby("cycle").sum()
            return agg

        f, nw = per_cycle(full), per_cycle(new)
        # pair each full moon with the nearest new moon (about two weeks away)
        nearest = nw.index[np.abs(nw.index.values[None, :] - f.index.values[:, None]).argmin(1)]
        F, N = f.values, nw.loc[nearest].values
        est = np.log(F[:, 0].sum() / F[:, 1].sum()) - np.log(N[:, 0].sum() / N[:, 1].sum())
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(F), len(F))
            boot.append(np.log(F[i, 0].sum() / F[i, 1].sum()) - np.log(N[i, 0].sum() / N[i, 1].sum()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"test": "Night vs same-day daytime, full vs new moon (2017-2026)", "group": g,
                     "n": f"{len(F)} cycles, {int(F[:, 0].sum()):,} full-moon night calls",
                     "effect_pct": 100 * np.expm1(est), "ci_low": 100 * np.expm1(lo), "ci_high": 100 * np.expm1(hi)})
    return rows


def main() -> None:
    rng = np.random.default_rng(0)
    full = moon_days(full_moon_instants("2017-03-01", "2026-03-01"))
    new = moon_days(new_moon_instants("2017-03-01", "2026-03-01"))
    rows = [whole_day_test(full, rng)] + night_test(full, new, rng)
    out = pd.DataFrame(rows)
    out["ci_halfwidth"] = (out["ci_high"] - out["ci_low"]) / 2
    out.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
