# scripts/make_incident_extract.py
"""
A slim, incident-level extract of the 560 MB raw export for the hour-level and
response-time analyses (darkness_dst.py, response_times.py, full_moon.py).

Keeps the call's local clock time ('Start Time'; the 2-3am hour is missing on
spring-forward days, so it is local wall-clock time), its dispatch category
(same rules as make_category_series.py), a robbery flag, priority (0 = most
urgent), and two response times in seconds: call received to first unit on scene
('Calltime Arrive') and dispatch to on scene ('Dispatch Arrive').

Output: data/incidents_slim.pkl (gitignored)
Run from the repo root:  .venv/bin/python -m scripts.make_incident_extract
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.make_category_series import RAW_PATH, categorize  # noqa: E402

OUT_PATH = BASE_DIR / "data" / "incidents_slim.pkl"
ROBBERY_KEYS = ("ROBBERY", "ROBOS", "ROBP", "CAR JACKING", "CARJACKING")


def main() -> None:
    cols = ["Start Time", "Initial Type", "Priority", "Calltime Arrive", "Dispatch Arrive"]
    df = pd.read_csv(RAW_PATH, usecols=cols, dtype=str)
    out = pd.DataFrame({"time": pd.to_datetime(df["Start Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")})
    types = df["Initial Type"].fillna("")
    cat_map = {t: categorize(t) for t in types.unique()}
    out["category"] = types.map(cat_map).astype("category")
    out["robbery"] = types.str.upper().str.contains("|".join(ROBBERY_KEYS), regex=True)
    out["priority"] = pd.to_numeric(df["Priority"], errors="coerce").astype("Int8")
    out["call_to_arrive_s"] = pd.to_numeric(df["Calltime Arrive"], errors="coerce").astype("float32")
    out["dispatch_to_arrive_s"] = pd.to_numeric(df["Dispatch Arrive"], errors="coerce").astype("float32")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    out.to_pickle(OUT_PATH)
    print(f"Wrote {OUT_PATH.name}: {len(out):,} incidents, {out['time'].min()} to {out['time'].max()}; "
          f"robbery calls {int(out['robbery'].sum()):,}")


if __name__ == "__main__":
    main()
