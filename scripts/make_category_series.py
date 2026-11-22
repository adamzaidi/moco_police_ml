# scripts/make_category_series.py
"""
Daily incident counts broken down by dispatch type.

Maps each of the ~230 'Initial Type' values (what the call was dispatched as)
to one of nine categories with ordered keyword rules, then counts incidents per
day per category. Adds one objective series: incidents with a crash report
number attached ('Crash Reports' non-empty), which doesn't depend on how the
call was classified.

Also counts incidents per day by police district (1D-6D; the ~200 incidents
coded OOC/PCW/TRU or blank go to "Other" so districts sum to incident_count),
and each district's mix of categories.

Outputs:
    data/dispatch_type_map.csv          every Initial Type, its category, and its count (review this)
    data/daily_counts_by_type.csv       date x category counts (regenerated, gitignored)
    data/daily_counts_by_district.csv   date x district counts (regenerated, gitignored)
    data/district_category_mix.csv      share of each district's incidents in each category

Run from the repo root:  .venv/bin/python scripts/make_category_series.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw_incidents.csv"
MAP_PATH = BASE_DIR / "data" / "dispatch_type_map.csv"
OUT_PATH = BASE_DIR / "data" / "daily_counts_by_type.csv"
DISTRICT_PATH = BASE_DIR / "data" / "daily_counts_by_district.csv"
MIX_PATH = BASE_DIR / "data" / "district_category_mix.csv"
DISTRICTS = {"1D": "1D Rockville", "2D": "2D Bethesda", "3D": "3D Silver Spring",
             "4D": "4D Wheaton", "5D": "5D Germantown", "6D": "6D Gaithersburg"}

# First matching rule wins, so order matters (e.g. ANIMAL ABUSE must hit "Animal"
# before "Violent / person" sees ABUSE). Matching is substring, on the upper-cased type.
RULES: list[tuple[str, list[str]]] = [
    ("Animal", ["ANIMAL", "ANI -", "DEERP", "HUNT"]),
    ("Traffic crash", ["TRAFFIC/TRANSPORTATION", "PEDESTRIAN STRUCK", "COLLISION", "ACCELERATOR STUCK"]),
    ("Traffic enforcement", ["TRAFFIC VIOLATION", "PARKING", "DRIVING UNDER", "TS - TRAFFIC STOP",
                             "TD - TRAFFIC DETAIL", "TRF - TRAFFIC", "TP - TRAFFIC PURSUIT"]),
    ("Alarm", ["ALARM", "ALRM", "CRYWOLF"]),
    ("Violent / person", ["ASSAULT", "ASLT", "DOMESTIC", "DOMV", "DOMDOS", "WEAP", "SHOOT", "SHOTS", "STAB",
                          "ROBBERY", "ROBOS", "ROBP", "CAR JACKING", "CARJACKING", "SEX", "RAPE", "HARASS",
                          "HARRASS", "STALK", "THREAT", "ABDUCTION", "ASSAILANT", "INDECENCY", "ABUSE",
                          "NEGLECT", "CALLER IN DANGER"]),
    ("Property", ["THEFT", "LARCENY", "LACENY", "BURG", "VANDAL", "VANDP", "STLVEH", "STOLEN", "FRAUD"]),
    ("Disorder / suspicious", ["SUSPICIOUS", "SUSICIOUS", "DISTURBANCE", "DISPOS", "DISPJ", "DISORD", "NOISE",
                               "TRESP", "TRE - ", "FIREWORKS", "CDS", "PROSTITUTION"]),
    ("Welfare / mental health", ["WELFARE", "MENTAL", "SUICID", "MISSING", "MISSARDD", "MP - MISSING",
                                 "OVERDOSE", "DECEASED", "RESCUE WITH FRS", "MCOT", "EMERGENCY PETITION",
                                 "LOCK OUT"]),
]
OTHER = "Other / admin"
CRASH_REPORT = "Crash report filed"


def categorize(initial_type: str) -> str:
    t = str(initial_type).upper()
    for cat, keys in RULES:
        if any(k in t for k in keys):
            return cat
    return OTHER


def main() -> None:
    df = pd.read_csv(RAW_PATH, usecols=["Start Time", "Initial Type", "Crash Reports", "Police District Number"], dtype=str)
    # Same date rule as make_daily_series.py, so categories sum to incident_count.
    df["date"] = pd.to_datetime(df["Start Time"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"])

    type_counts = df["Initial Type"].value_counts()
    mapping = pd.DataFrame({"initial_type": type_counts.index, "n_incidents": type_counts.values})
    mapping["category"] = mapping["initial_type"].map(categorize)
    mapping.to_csv(MAP_PATH, index=False)

    df["category"] = df["Initial Type"].map(dict(zip(mapping["initial_type"], mapping["category"])))
    daily = df.pivot_table(index="date", columns="category", values="Start Time", aggfunc="size", fill_value=0)
    daily[CRASH_REPORT] = df[df["Crash Reports"].notna()].groupby("date").size()
    all_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(all_days, fill_value=0).fillna(0).astype(int).rename_axis("date").reset_index()
    daily.to_csv(OUT_PATH, index=False)

    df["district"] = df["Police District Number"].map(DISTRICTS).fillna("Other")
    by_d = df.pivot_table(index="date", columns="district", values="Start Time", aggfunc="size", fill_value=0)
    by_d = by_d.reindex(all_days, fill_value=0).astype(int).rename_axis("date").reset_index()
    by_d.to_csv(DISTRICT_PATH, index=False)
    mix = pd.crosstab(df["district"], df["category"], normalize="index").round(4)
    mix.to_csv(MIX_PATH)

    share = mapping.groupby("category")["n_incidents"].sum().sort_values(ascending=False)
    print((100 * share / share.sum()).round(1).to_string())
    print(f"\n{CRASH_REPORT}: {int(daily[CRASH_REPORT].sum()):,} incidents")
    print(f"Wrote {MAP_PATH.name} ({len(mapping)} types) and {OUT_PATH.name} ({len(daily)} days)")
    print(f"Wrote {DISTRICT_PATH.name} and {MIX_PATH.name}")


if __name__ == "__main__":
    main()
