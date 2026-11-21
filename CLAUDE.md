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
6. `scripts/baseline_models.py`: naive yesterday / last-week baselines

## Evaluation rules (do not break)

- Time split, in `train_eval`: test = last 365 days (2025-02-05 to 2026-02-04), val = the 180 days before that, train = everything earlier.
- **Never select on test.** Model or feature selection calls `train_eval(evaluate_test=False)`, which returns only `best_val_mae`. Only an already-chosen configuration gets `evaluate_test=True`. The pre-Nov-2026 version ranked ablations by test MAE; that was leakage and the README says so.
- Compare runs seed by seed (paired) over `SEEDS` in `main.py`; a change is "beyond noise" only past 2 SE. Single-seed differences of about 1 MAE are noise.
- `save_outputs=False` stops `train_eval` overwriting `data/test_predictions_nn*.csv` during sweeps.

## Data / git

- `data/*` and `plots/*` are gitignored except `data/ablation_results.csv` and `data/final_test_results.csv` (the findings). The plots already tracked predate that rule.

## Status (2026-11-20)

- Branch `honest-eval`: env + validation-only ablation. Result: NN with all features 41.91 test MAE vs naive last-week 41.25; NN with **no weather** 32.15. Raw temperature columns are the main source of harm; rain flags are the only weather features whose removal reliably hurts.
- Next: (3) stronger baselines (7-day mean, ridge / gradient boosting on same features); (4) error on severe-weather days vs ordinary days, which is the project's actual question; consider rolling-origin validation (current val window is Aug-Feb only).
