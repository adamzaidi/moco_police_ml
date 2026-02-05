# scripts/train_nn.py
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "features_daily.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Helpers: reproducibility
# ============================================================
def set_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Helpers: transforms + clipping
# ============================================================
def signed_log1p(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def transform_weather(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Make weather features more model-friendly (reduce heavy tails).
    Deterministic transform (no future leakage).
    """
    df_ = df_.copy()

    # Non-negative precipitation-ish variables: log1p
    for c in ["precip_in", "rain_in", "snow_in", "precip_hours"]:
        if c in df_.columns:
            df_[c] = np.log1p(df_[c])

    # Anomaly can be negative: signed log1p
    if "precip_anom_30" in df_.columns:
        df_["precip_anom_30"] = signed_log1p(df_["precip_anom_30"])

    return df_


def clip_to_train_bounds(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    cols: List[str],
    q_low: float = 0.005,
    q_high: float = 0.995,
) -> pd.DataFrame:
    """
    Clip outliers in val/test to train-only quantile bounds.
    IMPORTANT: skips bool columns (quantile on bool can crash / behave weirdly).
    """
    cols = [c for c in cols if c in train_df.columns]

    clip_cols: List[str] = []
    for c in cols:
        if pd.api.types.is_bool_dtype(train_df[c]):
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]):
            clip_cols.append(c)

    if not clip_cols:
        return other_df

    lo = train_df[clip_cols].quantile(q_low, numeric_only=True)
    hi = train_df[clip_cols].quantile(q_high, numeric_only=True)

    clipped = other_df.copy()
    for c in clip_cols:
        clipped[c] = clipped[c].clip(lo[c], hi[c])
    return clipped


# ============================================================
# Core: one callable training run (for main.py ablations)
# ============================================================
def train_eval(
    data_path: Path = DATA_PATH,
    drop_cols: Optional[List[str]] = None,
    seed: int = 0,
    max_epochs: int = 200,
    patience: int = 12,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    huber_delta: float = 25.0,
    make_plots: bool = True,
    plot_suffix: str = "",
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Train NN with time-aware split, early stopping on VAL, evaluate once on TEST.

    Returns a dict with:
      - test_mae, best_val_mae, n_features, feature_cols, preds_df
      - saved_csv (Path), saved_plot (Path or None)
    """
    set_seed(seed)
    device = torch.device("cpu")

    # -----------------------
    # Load data
    # -----------------------
    if verbose:
        print("STEP A: loading data...")

    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # -----------------------
    # Columns
    # -----------------------
    X_cols = [c for c in df.columns if c not in ["date", "incident_count"]]
    X_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(df[c])]

    if drop_cols:
        drop_set = set(drop_cols)
        X_cols = [c for c in X_cols if c not in drop_set]

    # -----------------------
    # Time-aware split
    # -----------------------
    max_date = df["date"].max()
    test_cutoff = max_date - pd.Timedelta(days=365)
    val_cutoff = test_cutoff - pd.Timedelta(days=180)

    train = df[df["date"] <= val_cutoff].copy()
    val = df[(df["date"] > val_cutoff) & (df["date"] <= test_cutoff)].copy()
    test = df[df["date"] > test_cutoff].copy()

    if verbose:
        print(
            "STEP B: split complete | train:",
            len(train),
            "| val:",
            len(val),
            "| test:",
            len(test),
            "| features:",
            len(X_cols),
        )
        if drop_cols:
            print("Dropped:", drop_cols)
        print("Feature columns:", X_cols)

    # -----------------------
    # Build X as DataFrames (transform/clip safely)
    # -----------------------
    X_train_df = train[X_cols].copy()
    X_val_df = val[X_cols].copy()
    X_test_df = test[X_cols].copy()

    # Weather transforms (harmless for non-weather cols)
    X_train_df = transform_weather(X_train_df)
    X_val_df = transform_weather(X_val_df)
    X_test_df = transform_weather(X_test_df)

    # Clip outliers to TRAIN bounds
    X_val_df = clip_to_train_bounds(X_train_df, X_val_df, X_cols, q_low=0.005, q_high=0.995)
    X_test_df = clip_to_train_bounds(X_train_df, X_test_df, X_cols, q_low=0.005, q_high=0.995)

    # numpy arrays
    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values

    y_train = train["incident_count"].values.astype(np.float32)
    y_val = val["incident_count"].values.astype(np.float32)
    y_test = test["incident_count"].values.astype(np.float32)

    # -----------------------
    # Scale features (fit TRAIN only)
    # -----------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # -----------------------
    # Torch tensors
    # -----------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # -----------------------
    # Model
    # -----------------------
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1),
    ).to(device)

    loss_fn = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -----------------------
    # Training loop (early stop on VAL)
    # -----------------------
    if verbose:
        print("STEP C: training...")

    best_val_mae = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).numpy().flatten()
        val_mae = mean_absolute_error(y_val, val_preds)

        if verbose:
            print(f"Epoch {epoch:03d} | val_mae: {val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping.")
                break

    # -----------------------
    # Final evaluation on TEST (one-time)
    # -----------------------
    if verbose:
        print("STEP D: evaluating on TEST...")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_preds = model(X_test_t).numpy().flatten()

    test_mae = mean_absolute_error(y_test, test_preds)

    if verbose:
        print("\n=== PYTORCH RESULTS (proper test eval) ===")
        print(f"MAE (Test): {test_mae:.2f}")

    # -----------------------
    # Save predictions + plot
    # -----------------------
    pred_df = test[["date", "incident_count"]].copy()
    pred_df["pred_nn"] = test_preds

    suffix = f"_{plot_suffix}" if plot_suffix else ""
    out_csv = BASE_DIR / "data" / f"test_predictions_nn{suffix}.csv"
    pred_df.to_csv(out_csv, index=False)

    out_plot = None
    if make_plots:
        last_90 = pred_df.tail(90)
        plt.figure(figsize=(12, 4))
        plt.plot(last_90["date"], last_90["incident_count"], label="Actual")
        plt.plot(last_90["date"], last_90["pred_nn"], label="NN Pred")
        plt.title("NN vs Actual (Last 90 Days of Test)")
        plt.xlabel("Date")
        plt.ylabel("Incidents")
        plt.legend()
        plt.tight_layout()
        out_plot = PLOTS_DIR / f"nn_vs_actual_last_90_days{suffix}.png"
        plt.savefig(out_plot, dpi=150)
        plt.close()

    if verbose:
        print("Saved:", out_csv)
        if out_plot:
            print("Saved:", out_plot)

        # Worst 10 errors
        err = np.abs(pred_df["incident_count"].values - pred_df["pred_nn"].values)
        worst_idx = np.argsort(-err)[:10]
        print("\nWorst 10 absolute errors (date, actual, pred, abs_err):")
        for i in worst_idx:
            r = pred_df.iloc[i]
            print(r["date"].date(), float(r["incident_count"]), float(r["pred_nn"]), float(err[i]))

    return {
        "test_mae": float(test_mae),
        "best_val_mae": float(best_val_mae),
        "n_features": int(len(X_cols)),
        "feature_cols": X_cols,
        "preds_df": pred_df,
        "saved_csv": out_csv,
        "saved_plot": out_plot,
    }


# ============================================================
# Script entrypoint (keeps your old behavior)
# ============================================================
if __name__ == "__main__":
    # Default run = baseline training
    train_eval(
        data_path=DATA_PATH,
        drop_cols=None,
        seed=0,
        max_epochs=200,
        patience=12,
        make_plots=True,
        plot_suffix="baseline",
        verbose=True,
    )