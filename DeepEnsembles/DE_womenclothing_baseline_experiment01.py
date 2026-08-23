import os
import random
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf


# ==========================================================
# 🔧 CONFIGURATION (EDIT ONLY THIS SECTION IF NEEDED)
# ==========================================================

DATA_PATH = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\women_clothing_reviews_normalized.csv"
TARGET_COLUMN = "Rating"      # Change to "rating" if needed
N_ROWS = 2000
TEST_SIZE = 0.20
SEED = 42

N_MEMBERS = 5
EPOCHS = 60
BATCH_SIZE = 64
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
DROPOUT = 0.10

OUT_DIR = "de_women_exp01_baseline_outputs"

# ==========================================================


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim, seed):
    init = tf.keras.initializers.GlorotUniform(seed=seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", kernel_initializer=init),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


def is_text_column(series):
    if series.dtype != object:
        return False
    avg_len = series.dropna().astype(str).str.len().mean()
    return avg_len >= 25


def main():

    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    if len(df) > N_ROWS:
        df = df.iloc[:N_ROWS].copy()

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # ------------------------------------------------------
    # Identify baseline features (exclude text-like columns)
    # ------------------------------------------------------
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    text_cols = []
    for c in feature_cols:
        if is_text_column(df[c]):
            text_cols.append(c)

    baseline_cols = [c for c in feature_cols if c not in text_cols]

    numeric_cols = [c for c in baseline_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in baseline_cols if df[c].dtype == object]

    print("Numeric cols     :", numeric_cols)
    print("Categorical cols :", categorical_cols)
    print("Excluded text    :", text_cols)

    X = df[numeric_cols + categorical_cols]
    y = df[TARGET_COLUMN].astype(float).values

    # ------------------------------------------------------
    # Split once (save for Exp-02 consistency)
    # ------------------------------------------------------
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    np.savez(os.path.join(OUT_DIR, "split_indices.npz"),
             idx_train=idx_train,
             idx_test=idx_test)

    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # ------------------------------------------------------
    # Preprocessing
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
    print("\nTraining Deep Ensemble...")

    member_predictions = []

    for m in range(N_MEMBERS):
        member_seed = SEED + 1000 + m
        set_seed(member_seed)

        model = build_model(input_dim, member_seed)

        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        preds = model.predict(X_test, verbose=0).reshape(-1)
        member_predictions.append(preds)

        model.save(os.path.join(OUT_DIR, f"member_{m+1}.keras"))

    member_predictions = np.array(member_predictions)

    y_pred_mean = member_predictions.mean(axis=0)
    y_pred_std = member_predictions.std(axis=0)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2 = r2_score(y_test, y_pred_mean)

    print("\n===== Experiment-01 Results (Women | Deep Ensembles | Baseline) =====")
    print(f"Rows used  : {len(df)}")
    print(f"Train/Test : {len(idx_train)}/{len(idx_test)}")
    print(f"Members    : {N_MEMBERS}")
    print(f"MAE        : {mae:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"R^2        : {r2:.4f}")

    # ------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------
    results_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_mean": y_pred_mean,
        "y_pred_std": y_pred_std
    })

    results_df.to_csv(
        os.path.join(OUT_DIR, "women_de_exp01_baseline_predictions.csv"),
        index=False
    )

    summary = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "rows_used": int(len(df)),
        "n_members": int(N_MEMBERS)
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs inside:", OUT_DIR)
    print("Done.\n")


if __name__ == "__main__":
    main()
