# scripts/darkness_dst.py
"""
Does darkness bring crime? Daylight saving time as a natural experiment.

At each clock change the sun sets an hour earlier or later by the clock,
overnight, while everything else about the day (work and school hours, the
season, the weather) barely moves. So the clock hours around sunset switch from
light to dark (November) or dark to light (March), and nothing else does.
Doleac & Sanders (2015) used this to show robbery falls when evenings get
lighter.

Design: for each change (spring 2018-2025 except 2020, which ran into the COVID
lockdown; fall 2017-2025), take the 21 days before and after (dropping the change
Sunday and a few evening-heavy holidays). Count calls in the "switch" clock hours,
the ones whose darkness changes most, and in midday control hours (10:00-16:00,
light either way). The darkness effect is the change in switch-hour calls relative
to control-hour calls, which strips out anything that moves the whole day.
Two checks: late-evening hours (21:00-24:00, dark either way) should show no
effect, and the same comparison on fake change dates five weeks earlier should
show none either.

CIs: bootstrap over days within each change's before / after period.

Inputs:  data/incidents_slim.pkl (make_incident_extract.py)
Outputs: data/darkness_dst_effects.csv, plots/darkness_dst.png, plots/darkness_dst_effects.png
Run from the repo root:  .venv/bin/python -m scripts.darkness_dst
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.astro import TZ, sunset_local  # noqa: E402

SLIM_PATH = BASE_DIR / "data" / "incidents_slim.pkl"
OUT_CSV = BASE_DIR / "data" / "darkness_dst_effects.csv"
OUT_PLOT = BASE_DIR / "plots" / "darkness_dst.png"
OUT_EFFECTS_PLOT = BASE_DIR / "plots" / "darkness_dst_effects.png"
WINDOW = 21
N_BOOT = 2000
DUSK_MIN = 30          # dark = more than 30 min after sunset (about the end of civil twilight)
CONTROL_HOURS = list(range(10, 16))
LATE_HOURS = [21, 22, 23]
SKIP_EVENTS = {pd.Timestamp("2020-03-08")}  # post window is the COVID lockdown
PLACEBO_SHIFT = 35     # days

GROUPS = {
    "All incidents": lambda d: np.ones(len(d), bool),
    "Street crime (property, violent, disorder)": lambda d: d["category"].isin(
        ["Property", "Violent / person", "Disorder / suspicious"]).values,
    "Robbery": lambda d: d["robbery"].values,
    "Traffic crash": lambda d: (d["category"] == "Traffic crash").values,
}


def dst_changes(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    days = pd.date_range(start, end, freq="D")
    off = pd.DatetimeIndex(days + pd.Timedelta(hours=12)).tz_localize(TZ).map(lambda t: t.utcoffset())
    out = []
    for i in range(1, len(days)):
        if off[i] != off[i - 1]:
            out.append((days[i], "spring" if off[i] > off[i - 1] else "fall"))
    return out


def excluded_days(years) -> set:
    out = set()
    for y in years:
        nov1 = pd.Timestamp(f"{y}-11-01")
        thanks = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7 + 21)
        out |= {pd.Timestamp(f"{y}-10-31"), pd.Timestamp(f"{y}-11-11"), pd.Timestamp(f"{y}-03-17"),
                thanks, thanks + pd.Timedelta(days=1)}
    return out


def darkness_by_hour(days: pd.DatetimeIndex) -> np.ndarray:
    """Fraction of each clock hour (0-23) that is dark, for each day."""
    dark_from = sunset_local(days) + pd.Timedelta(minutes=DUSK_MIN)
    mins = (dark_from - dark_from.dt.normalize()).dt.total_seconds().values / 60
    starts = np.arange(24) * 60
    return np.clip((starts[None, :] + 60 - mins[:, None]) / 60, 0, 1)  # evening darkness only


def build_windows(slim: pd.DataFrame, shift: int = 0) -> pd.DataFrame:
    """One row per (day in a window): event id, season, period, hourly counts per group."""
    start, end = slim["time"].min().normalize(), slim["time"].max().normalize()
    skip = excluded_days(range(start.year, end.year + 1))
    rows = []
    for change, season in dst_changes(start, end):
        if change in SKIP_EVENTS:
            continue
        c = change - pd.Timedelta(days=shift)
        for k in range(-WINDOW, WINDOW + 1):
            d = c + pd.Timedelta(days=k)
            if k == 0 or d < start or d > end or d in skip:
                continue
            rows.append({"date": d, "event": f"{season} {change:%Y}", "season": season,
                         "period": "after" if k > 0 else "before"})
    win = pd.DataFrame(rows)
    # drop events whose window is cut off by the data's start or end
    full = win.groupby("event")["period"].nunique() == 2
    return win[win["event"].isin(full[full].index)].reset_index(drop=True)


def hourly_counts(slim: pd.DataFrame, dates: pd.Series) -> dict[str, np.ndarray]:
    s = slim[slim["time"].dt.normalize().isin(set(dates))]
    day_idx = pd.Index(dates).get_indexer(s["time"].dt.normalize())
    hour = s["time"].dt.hour.values
    out = {}
    for name, f in GROUPS.items():
        m = f(s)
        mat = np.zeros((len(dates), 24))
        np.add.at(mat, (day_idx[m], hour[m]), 1)
        out[name] = mat
    return out


def switch_hours(win: pd.DataFrame, season: str) -> list[int]:
    w = win[win["season"] == season]
    dark = darkness_by_hour(pd.DatetimeIndex(w["date"]))
    diff = dark[(w["period"] == "after").values].mean(0) - dark[(w["period"] == "before").values].mean(0)
    return [h for h in range(24) if abs(diff[h]) > 0.25]


def log_ratio_draws(win: pd.DataFrame, counts: dict, season: str, hours: list[int], control: list[int],
                    rng) -> dict[str, tuple[float, np.ndarray, int]]:
    """log darkness effect: change in (hours / control) calls when the hours turn dark, plus bootstrap draws."""
    w = win[win["season"] == season]
    events = w["event"].unique()
    cells = [(np.where((win["event"] == e) & (win["period"] == p))[0], p == "after")
             for e in events for p in ("before", "after")]
    sign = 1 if season == "fall" else -1  # spring: the hours go dark -> light, so flip
    res = {}
    for name, mat in counts.items():
        a, c = mat[:, hours].sum(1), mat[:, control].sum(1)
        tot = {k: np.zeros(N_BOOT + 1) for k in ("Ab", "Cb", "Aa", "Ca")}
        for idx, aft in cells:
            draw = idx[rng.integers(0, len(idx), size=(N_BOOT, len(idx)))]
            sa, sc = np.append(a[idx].sum(), a[draw].sum(1)), np.append(c[idx].sum(), c[draw].sum(1))
            tot["Aa" if aft else "Ab"] += sa
            tot["Ca" if aft else "Cb"] += sc
        lr = sign * (np.log(tot["Aa"] / tot["Ca"]) - np.log(tot["Ab"] / tot["Cb"]))
        n = int(a[np.concatenate([i for i, _ in cells])].sum())
        res[name] = (lr[0], lr[1:], n)
    return res


def main() -> None:
    # incidents_slim.pkl is written locally by make_incident_extract.py, never downloaded
    slim = pd.read_pickle(SLIM_PATH)
    real = build_windows(slim, 0)
    hours = {s: switch_hours(real, s) for s in ("spring", "fall")}
    tests = [("switch hours vs midday", "switch", CONTROL_HOURS),
             ("switch hours vs late evening", "switch", LATE_HOURS),
             ("check: late evening vs midday", "late", CONTROL_HOURS)]
    rows, profiles = [], {}
    for label, shift in (("DST change", 0), ("Placebo: 5 weeks earlier", PLACEBO_SHIFT)):
        rng = np.random.default_rng(0)
        win = build_windows(slim, shift)
        counts = hourly_counts(slim, win["date"])
        for test, which, control in tests:
            per = {s: log_ratio_draws(win, counts, s, hours[s] if which == "switch" else LATE_HOURS, control, rng)
                   for s in ("spring", "fall")}
            for name in GROUPS:
                combos = {s: per[s][name] for s in ("spring", "fall")}
                combos["pooled"] = ((per["spring"][name][0] + per["fall"][name][0]) / 2,
                                    (per["spring"][name][1] + per["fall"][name][1]) / 2,
                                    per["spring"][name][2] + per["fall"][name][2])
                for season, (est, draws, n) in combos.items():
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                    rows.append({"design": label, "test": test, "season": season, "group": name,
                                 "n_calls_in_hours": n, "darkness_effect_pct": 100 * np.expm1(est),
                                 "ci_low": 100 * np.expm1(lo), "ci_high": 100 * np.expm1(hi)})
        if shift == 0:
            for season in ("spring", "fall"):
                w = win["season"] == season
                prof = {}
                for name in ("Street crime (property, violent, disorder)", "Traffic crash"):
                    mat = counts[name]
                    b, a = mat[(w & (win["period"] == "before")).values], mat[(w & (win["period"] == "after")).values]
                    prof[name] = 100 * ((a.sum(0) / a[:, CONTROL_HOURS].sum()) /
                                        (b.sum(0) / b[:, CONTROL_HOURS].sum()) - 1)
                profiles[season] = (prof, hours[season])
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 220)
    print("Switch hours:", hours)
    print(out.to_string(index=False, float_format="%.1f"))
    plot(profiles)
    plot_effects(out)


def plot(profiles: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, season in zip(axes, ("spring", "fall")):
        prof, hours = profiles[season]
        x = np.arange(24)
        ax.axvspan(min(hours) - 0.5, max(hours) + 0.5, color="#f1e4c8", zorder=0)
        ax.axvspan(CONTROL_HOURS[0] - 0.5, CONTROL_HOURS[-1] + 0.5, color="#eef0f2", zorder=0)
        for (name, y), color in zip(prof.items(), ("#2a78d6", "#d9822b")):
            ax.plot(x, y, color=color, lw=2, marker="o", ms=3.5, label=name.split(" (")[0])
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)], fontsize=8.5)
        title = ("March: evenings get lighter" if season == "spring" else "November: evenings get darker")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Clock hour (shaded: hours that switch, grey: midday control)", fontsize=8.5)
    axes[0].set_ylabel("% change in calls vs midday,\n3 weeks after vs 3 weeks before")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    for ax in axes:
        ax.set_ylim(max(ax.get_ylim()[0], -40), min(ax.get_ylim()[1], 40))
    fig.suptitle("What happens to each clock hour when daylight saving time starts and ends (2017-2025)",
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)



def plot_effects(out: pd.DataFrame) -> None:
    """Pooled darkness effect per group: both control choices, plus the fake-date check."""
    groups = ["Traffic crash", "All incidents", "Street crime (property, violent, disorder)", "Robbery"]
    labels = ["Traffic crashes", "All incidents", "Street crime", "Robbery"]
    series = [("DST change", "switch hours vs midday", "vs midday", "#104281", "o"),
              ("DST change", "switch hours vs late evening", "vs late evening", "#86b6ef", "o"),
              ("Placebo: 5 weeks earlier", "switch hours vs midday", "fake dates 5 weeks earlier", "#9aa0a6", "D")]
    lim = (-40, 80)
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    y = np.arange(len(groups))[::-1]
    for k, (design, test, label, color, marker) in enumerate(series):
        d = out[(out["design"] == design) & (out["test"] == test) & (out["season"] == "pooled")].set_index("group")
        d = d.loc[groups]
        yy = y + 0.22 - 0.22 * k
        lo, hi = d["ci_low"].clip(*lim), d["ci_high"].clip(*lim)
        ax.hlines(yy, lo, hi, color=color, lw=2)
        ax.plot(d["darkness_effect_pct"], yy, marker, color=color, ms=7, label=label,
                mfc=color if k < 2 else "white", mew=1.5)
        for yi, h in zip(yy, d["ci_high"]):
            if h > lim[1]:
                ax.annotate("", (lim[1], yi), (lim[1] - 6, yi), arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlim(lim)
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlabel("% change in calls when the switch hours turn dark (95% CI; both clock changes pooled)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              title="Switch hours compared with:", title_fontsize=9, alignment="left")
    fig.suptitle("Darkness raises crash calls; the crime estimate flips with the comparison", x=0.02, ha="left")
    fig.text(0.02, 0.01, "Robbery: only 146 calls in the switch hours, so its intervals are very wide (one runs off the chart).",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_EFFECTS_PLOT, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
