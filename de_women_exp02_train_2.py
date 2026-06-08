import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf


# ==========================================================
# 🔧 EDIT ONLY THIS SECTION IF NEEDED
# ==========================================================

DATA_PATH = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\women_exp02_with_llm.csv"
OUT_DIR = "de_women_exp02_outputs_1999"

TARGET_COLUMN = "Rating"   # change to "rating" if your CSV uses lowercase
N_ROWS = 2000              # we attempt first 2000; actual loaded may be 1999

# Train/Test split (NEW split because row count changed)
SEED = 42
TEST_SIZE = 0.20

# Deep Ensemble settings
N_MEMBERS = 5
EPOCHS = 60
BATCH_SIZE = 64
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
DROPOUT = 0.10

# Optional: clamp predictions to rating range (Women Clothing is usually 1..5)
CLAMP_PREDICTIONS = True
Y_MIN = 1.0
Y_MAX = 5.0

# ==========================================================


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def build_model(input_dim: int, seed: int) -> tf.keras.Model:
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


def is_text_column(series: pd.Series) -> bool:
    """Detect long free-text columns to exclude from baseline features."""
    if series.dtype != object:
        return False
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return False
    return vals.str.len().mean() >= 25


def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV loader for occasional malformed lines.
    Uses python engine; skips bad lines if needed.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        pass

    try:
        return pd.read_csv(path, engine="python")
    except Exception:
        pass

    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    print("\nLoading Women Exp-02 dataset (1999-row tolerant)...")
    df = robust_read_csv(DATA_PATH)

    if N_ROWS and len(df) > N_ROWS:
        df = df.iloc[:N_ROWS].copy()

    print(f"Loaded rows: {len(df)}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found.\n"
            f"Columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    print(f"Rows after dropping missing target: {len(df)}")

    # ------------------------------------------------------
    # Feature selection:
    # - thematic cols: T1_... to T10_...
    # - baseline: numeric + categorical (excluding long text)
    # ------------------------------------------------------
    all_feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    thematic_cols = [c for c in all_feature_cols if c.startswith("T") and "_" in c]

    text_cols = []
    for c in all_feature_cols:
        if df[c].dtype == object and is_text_column(df[c]):
            text_cols.append(c)

    baseline_candidates = [c for c in all_feature_cols if c not in thematic_cols and c not in text_cols]

    baseline_numeric = [c for c in baseline_candidates if pd.api.types.is_numeric_dtype(df[c])]
    baseline_categorical = [c for c in baseline_candidates if df[c].dtype == object]

    # Ensure thematic numeric (0/1)
    for c in thematic_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    numeric_cols = baseline_numeric + thematic_cols
    categorical_cols = baseline_categorical

    print("Baseline numeric     :", baseline_numeric)
    print("Baseline categorical :", baseline_categorical)
    print("Thematic (LLM) cols  :", thematic_cols)
    print("Excluded text        :", text_cols)

    with open(os.path.join(OUT_DIR, "feature_audit.json"), "w", encoding="utf-8") as f:
        json.dump({
            "rows_used": int(len(df)),
            "target_col": TARGET_COLUMN,
            "baseline_numeric": baseline_numeric,
            "baseline_categorical": baseline_categorical,
            "thematic_cols": thematic_cols,
            "excluded_text_cols": text_cols
        }, f, indent=2)

    X = df[numeric_cols + categorical_cols].copy()
    y = df[TARGET_COLUMN].astype(float).values

    # ------------------------------------------------------
    # NEW split for 1999 rows
    # ------------------------------------------------------
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    np.savez(os.path.join(OUT_DIR, "split_indices_1999.npz"),
             idx_train=idx_train, idx_test=idx_test)

    X_train = X.iloc[idx_train].copy()
    X_test = X.iloc[idx_test].copy()
    y_train = y[idx_train]
    y_test = y[idx_test]

    # ------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    input_dim = X_train.shape[1]

    # ------------------------------------------------------
    # Train Deep Ensemble
    # ------------------------------------------------------
    print("\nTraining Deep Ensemble (Exp-02 with 1999 rows)...")
    member_preds = []

    for m in range(N_MEMBERS):
        member_seed = SEED + 2000 + m
        set_seed(member_seed)

        model = build_model(input_dim, member_seed)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        preds = model.predict(X_test, verbose=0).reshape(-1)
        if CLAMP_PREDICTIONS:
            preds = clamp(preds, Y_MIN, Y_MAX)

        member_preds.append(preds)
        model.save(os.path.join(OUT_DIR, f"member_{m+1}.keras"))

    member_preds = np.array(member_preds)
    y_pred_mean = member_preds.mean(axis=0)
    y_pred_std = member_preds.std(axis=0)

    if CLAMP_PREDICTIONS:
        y_pred_mean = clamp(y_pred_mean, Y_MIN, Y_MAX)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2 = r2_score(y_test, y_pred_mean)

    print("\n===== Experiment-02 Results (Women | Deep Ensembles | Thematic LLM | 1999 rows) =====")
    print(f"Rows used  : {len(df)}")
    print(f"Train/Test : {len(idx_train)}/{len(idx_test)}")
    print(f"Members    : {N_MEMBERS}")
    print(f"MAE        : {mae:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"R^2        : {r2:.4f}")

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------
    out_pred = pd.DataFrame({
        "idx": idx_test,
        "y_true": y_test,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std
    }).sort_values("idx").reset_index(drop=True)

    pred_path = os.path.join(OUT_DIR, "women_de_exp02_predictions_1999.csv")
    out_pred.to_csv(pred_path, index=False)

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Exp-02 (1999 rows, new split)",
            "rows_used": int(len(df)),
            "train_size": int(len(idx_train)),
            "test_size": int(len(idx_test)),
            "n_members": int(N_MEMBERS),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "predictions_csv": os.path.abspath(pred_path),
            "note": "This Exp-02 uses a NEW split because the dataset loaded as 1999 rows (bad CSV line skipped)."
        }, f, indent=2)

    print("\nSaved outputs inside:", OUT_DIR)
    print("Saved predictions   :", pred_path)
    print("Done.\n")


if __name__ == "__main__":
    main()