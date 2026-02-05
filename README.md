Project Overview

This project models daily incident counts using time-series features, calendar effects, holidays, and weather.
The goal is not just prediction accuracy, but interpretability: understanding which factors actually matter, especially around weather severity and holidays.

The pipeline is designed to be:
	•	Time-safe (no future data leakage)
	•	Interpretable (explicit features, not black-box embeddings)
	•	Extensible (easy ablation and feature experimentation)

The core question driving the project:
Do severe weather conditions (e.g., heavy snow, storms) meaningfully impact incident volume more than mild or continuous effects like small temperature changes?

Repository Structure

scripts/
	•	build_features.py
Builds the full feature table from raw daily counts, holidays, and weather.
Includes:
	•	Lag features
	•	Rolling averages
	•	Day-of-week encoding
	•	Named holiday windows
	•	Tiered weather severity flags
	•	Compound danger indicators (ice risk, blowing snow, storms)
	•	Short-term weather streaks
	•	train_nn.py
Trains and evaluates a neural network model on the engineered features.
Uses:
	•	Time-based train/validation/test split
	•	Robust scaling
	•	Early stopping
	•	Huber loss
	•	Proper test-only evaluation

data/
	•	daily_counts.csv (input, not committed)
	•	weather_daily.csv (input, not committed)
	•	features_daily.csv (generated, not committed)
	•	test_predictions_nn.csv (generated, not committed)

main.py
	•	Runs automated ablation experiments.
	•	Systematically removes weather features (individually and in groups).
	•	Compares test MAE to determine which features are helpful vs noise.
	•	Outputs a ranked CSV showing impact of each feature or group.

plots/
	•	Saved model prediction plots (optional, generated locally)

Typical Workflow
	1.	Place raw data files into data/:
	•	daily_counts.csv
	•	weather_daily.csv
	2.	Build features:
python scripts/build_features.py
	3.	Train a baseline model:
python scripts/train_nn.py
	4.	Run ablation studies:
python main.py
	5.	Inspect results:
	•	data/ablation_results.csv
	•	Console output ranking most important weather features

Modeling Philosophy

This project intentionally avoids:
	•	End-to-end black-box models
	•	Learned representations with unclear meaning
	•	Random cross-validation on time-series data

Instead, it emphasizes:
	•	Explicit feature construction
	•	Domain-informed thresholds (e.g., heavy snow vs light snow)
	•	Feature ablation as the primary interpretability tool
	•	Comparing model performance changes (MAE deltas) rather than raw coefficients

Key Insights Enabled

The setup allows you to answer questions like:
	•	Do heavy snow days matter more than temperature anomalies?
	•	Are rain flags useful, or is snow doing most of the work?
	•	Do compound conditions (ice risk, storms) outperform raw weather values?
	•	Which weather features actively hurt performance (noise)?

This makes the results suitable for:
	•	Policy discussions
	•	Operational planning
	•	Academic or applied writeups
	•	Explaining model behavior to non-technical stakeholders

Reproducibility Notes
	•	Random seeds are fixed.
	•	All transformations are deterministic.
	•	No test data is used during training or feature construction.
	•	Derived data can always be regenerated from raw inputs.

Future Extensions

Possible next steps include:
	•	Seasonal interaction terms
	•	Separate weekday/weekend weather effects
	•	SHAP-style analysis on top of ablation results
	•	Comparison against simpler baselines (ARIMA, linear models)
	•	Event-level modeling if finer-grained data becomes available
