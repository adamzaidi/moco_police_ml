# moco_police_ml

Predicts daily Montgomery County (MD) police incident counts from lags, calendar/holiday features, and Open-Meteo weather; asks which weather signals actually matter. **Portfolio piece**: the README is the deliverable, so keep its Results section current and honest.

## Environment

- Python 3.12 venv at `.venv/` (gitignored). `requirements.txt` is pinned and verified to reproduce the Feb 2026 baseline exactly (test MAE 41.87, seeds 0-2). Run everything with `.venv/bin/python`.
- The system anaconda Python is broken for this project (pandas needs numpy >= 1.26; anaconda has 1.24.4). Don't use it.

## Pipeline

1. `scripts/make_daily_series.py`: `data/raw_incidents.csv` (~560 MB, dataMontgomery export) to `data/daily_counts.csv`
2. `scripts/fetch_weather_openmeteo.py`: `data/weather_daily.csv`
3. `scripts/build_features.py`: `data/features_daily.csv` (lags, rolling means, DOW, holiday windows, weather tiers, streaks; past data only)
4. `scripts/train_nn.py`: `train_eval()` is the single training entry point (MLP, Huber loss, early stopping on val)
5. `main.py`: two-stage ablation (about 8 min for stage 1 on CPU)
6. `scripts/compare_models.py`: NN vs naive rules vs ridge vs gradient boosting, with/without weather; also a ridge weather-group ablation (about 1 min)
7. `scripts/severe_weather.py`: rolling-origin (6 yearly windows, Feb 2020-Feb 2026) out-of-sample predictions for ridge/GBM with and without weather; per-category weather effects with bootstrap CIs (about 1 min)
8. `scripts/make_category_series.py`: maps 233 `Initial Type` values to 9 categories (ordered substring rules; first match wins) + `Crash report filed` series; reads the 560 MB raw CSV (about 1 min)
9. `scripts/weather_by_type.py`: per-category rolling-origin ridge weather effects + heatmap (about 10 s)
10. `scripts/baseline_models.py`: original naive baselines script (superseded by compare_models)

## Evaluation rules (do not break)

- Time split, `time_split()` in `scripts/train_nn.py` (use it everywhere): test = last 365 days (2025-02-05 to 2026-02-04), val = the 180 days before that, train = everything earlier.
- **Never select on test.** Model or feature selection calls `train_eval(evaluate_test=False)`, which returns only `best_val_mae`. Only an already-chosen configuration gets `evaluate_test=True`. The pre-Nov-2026 version ranked ablations by test MAE; that was leakage and the README says so.
- Compare runs seed by seed (paired) over `SEEDS` in `main.py`; a change is "beyond noise" only past 2 SE. Single-seed differences of about 1 MAE are noise.
- `save_outputs=False` stops `train_eval` overwriting `data/test_predictions_nn*.csv` during sweeps.

## Weather data gotchas

- Open-Meteo `snowfall_sum` is already in inches when `precipitation_unit=inch` (check `daily_units` in the response). Pre-Nov-2026 code divided by 2.54, so snow was 2.54x too small and ">= 3 in" never fired.
- Open-Meteo daily sustained wind tops out around 30 mph here; wind flags/tiers use **gusts** (windy >= 40, top tier >= 46 = NWS wind advisory).
- `END_DATE` in the fetch script is pinned to the last incident date. Open-Meteo revises its archive slightly (a few days of temps move > 1 F), so refetches won't be bit-identical.

## Data / git

- `data/*` and `plots/*` are gitignored except `data/ablation_results.csv` and `data/final_test_results.csv` (the findings). The plots already tracked predate that rule.

## Status (2026-11-21)

- PR #1 `honest-eval`: env + validation-only NN ablation. PR #2 `model-comparison`: NN vs naive vs ridge vs GBM. PR #3 `severe-weather` (stacked on #2): fixes snow units + wind thresholds, reruns everything, adds rolling-origin severe-weather analysis. PR #4 (stacked on #3): dispatch-type breakdown. Merge in order.
- Findings: ridge with all features is the pick (rolling-origin mean MAE 28.3 vs 28.6 no-weather, GBM 31.0, 7-day mean 32.8). Snow has a clear dose-response effect (any snow -4%, >=1 in -7%, >=3 in -12%, day after >=1 in -8%); ice and heavy rain about -2%; gusts >= 40 mph about +3%; heat none. Severe days are ~14% of days, which is why yearly MAE barely moves with weather.
- PR #4 `dispatch-types` (stacked on #3): snow cuts crime/disorder 13-16% and raises crash calls 23% (but filed crash reports fall); gusts raise alarms 27% (explains the wind effect) and TRAFFIC VIOLATION calls 24% (unexplained).
- Next ideas: district/beat-level effects, forecast vs observed weather, holiday x weather interactions.
