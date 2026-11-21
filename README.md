## Project Overview

This project models daily incident counts using time-series data enriched with calendar effects, named holidays, and weather conditions. The focus is not only on predictive performance but on interpretability: understanding which factors meaningfully influence incident volume and which ones are noise. In particular, the project is driven by the question of whether severe weather conditions such as heavy snow, storms, or ice risk have a larger and more reliable impact than mild or continuous effects like small temperature fluctuations.

The overall design emphasizes time safety, interpretability, and extensibility. All features are explicitly constructed rather than learned implicitly, no future information is used during training or feature creation, and the pipeline is structured to make systematic feature ablation straightforward. This allows model behavior to be explained in concrete, domain-relevant terms rather than treated as a black box.

## Repository Structure

The scripts directory contains the core pipeline. The build_features.py script constructs the full modeling dataset from raw daily incident counts, weather data, and calendar information. It generates lagged features, rolling averages, day-of-week encodings, named holiday windows, tiered weather severity indicators, compound danger proxies such as ice risk and blowing snow, and short-term weather streak features. All derived features are computed using past data only to avoid leakage.

The train_nn.py script trains and evaluates a neural network model on the engineered feature set. It uses a time-based train, validation, and test split, robust feature scaling, early stopping based on validation performance, and Huber loss to reduce sensitivity to extreme errors. Test performance is evaluated exactly once using the best validation model, and optional plots are generated for inspection.

The data directory is expected to contain raw input files such as daily_counts.csv and weather_daily.csv, as well as generated outputs like features_daily.csv and test_predictions_nn.csv. These files are intentionally not committed to version control and are regenerated locally. The plots directory holds generated visualizations produced during training or evaluation.

The main.py script runs the feature ablation in two stages. In the selection stage it trains the baseline model and one model per removed weather feature or feature group, each over five seeds, and ranks every run by validation MAE only; the test set is never scored, so it cannot influence which features are kept. Each removal is compared to the baseline seed by seed, and a change is flagged as real only when it exceeds two standard errors. In the evaluation stage it scores the test set exactly once, for two configurations: the all-features baseline and the run with the best validation MAE, reported next to naive baselines. Results go to data/ablation_results.csv (validation ranking) and data/final_test_results.csv (test scores).

## Typical Workflow

Set up the environment with Python 3.12: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Place the raw input files in the data directory, build the feature table with `.venv/bin/python scripts/build_features.py`, train a single baseline model with `.venv/bin/python scripts/train_nn.py`, and run the full ablation with `.venv/bin/python main.py` (about ten minutes on a laptop CPU).

## Results

Test year: 2025-02-05 to 2026-02-04 (365 days). Mean absolute error in daily incidents, averaged over five seeds; daily counts in this period average about 536.

| Model | Test MAE |
|---|---|
| Naive: yesterday's count | 43.06 |
| Neural network, all features | 41.91 (sd 1.33) |
| Naive: same day last week | 41.25 |
| **Neural network, no weather features** | **32.15** (sd 0.67) |

The model with every feature does no better than guessing last week's count. Removing all 33 weather columns was the best configuration on validation, and it held up on the untouched test year, cutting error by about a quarter. Calendar structure, lags, and holidays carry the signal; in this setup, weather mostly adds noise.

The ablation shows where that noise comes from. Raw temperature is the clearest problem: removing the three temperature columns improves validation MAE by 3.6, and each one hurts on its own. The only weather features whose removal reliably makes things worse are the threshold flags, especially rain (`is_rainy`, `is_heavy_rain`): with the rest of the weather left in, they appear to offset damage done by the raw measurements. The compound danger proxies (ice risk, blowing snow, storms) and the recent-weather streak counts show no reliable effect either way. Most single-feature effects fall within seed noise.

On the motivating question, whether severe weather matters more than mild variation: the answer so far is that neither improves the model on average. Severe days are rare, though, so an average over the year can hide them. A direct comparison of errors on severe-weather days is the next step.

Earlier versions of this project ranked features by test MAE, which let the test set leak into feature selection and overstated results. The current pipeline selects on validation only.

**Limitations.** The validation window is a single half-year (August 2024 to February 2025), so selection never sees spring or summer; rolling-origin validation would be more robust. Validation also drives early stopping, which makes validation scores slightly optimistic, which is why the chosen configuration is confirmed on the test year rather than trusted on its own. Flagging 47 comparisons at two standard errors will produce a few false positives.

## Modeling Philosophy

This project intentionally avoids end-to-end black-box approaches, opaque learned representations, and random cross-validation schemes that break the temporal structure of the data. Instead, it relies on explicit feature engineering, domain-informed thresholds for weather severity, and ablation-based evaluation to understand model behavior. Feature usefulness is assessed by observing how model performance changes when information is removed, rather than by relying on coefficients or internal network weights alone.

## Insights Enabled

The structure of the pipeline makes it possible to answer concrete questions about the data-generating process. It allows you to test whether heavy snow days matter more than temperature anomalies, whether rain-related features add value beyond snow indicators, whether compound danger conditions outperform raw weather measurements, and which weather variables actively degrade model performance when included. These insights are suitable for operational decision-making, policy discussion, or explanatory reporting to non-technical audiences.

## Reproducibility and Extensions

All experiments are reproducible due to fixed random seeds, deterministic transformations, and pinned dependencies in requirements.txt (verified to reproduce the original February 2026 numbers exactly). No derived data needs to be stored permanently, as all outputs can be regenerated from raw inputs. The project is designed to be extended easily, with natural next steps including seasonal interaction effects, weekday versus weekend differentiation, SHAP-style analysis layered on top of ablation results, comparisons to simpler statistical baselines, or higher-resolution modeling if finer-grained incident data becomes available.

