"""
Flipkart (fixed 2000 rows) — Deep Ensembles Experiment-02 (Base + LLM features)
✅ TensorFlow/Keras Deep Ensemble (regression) with BOUNDED OUTPUT for ratings
✅ Rating range fixed to [1, 5] using scaled sigmoid output (prevents crazy preds like 47)
✅ Computes MAE, RMSE, R² on TEST set
✅ Saves predictions CSV: y_true, pred_mean, pred_std (+ each ensemble member prediction)
✅ Uses a fixed train/test split saved once for fair comparison
✅ Run normally: python de_flipkart_experiment02_bounded.py

EDIT ONLY the CONFIG section (paths + columns if needed).
"""

import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# =========================================================
# CONFIG (EDIT THESE)
# =========================================================
DATA_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\flipkart_reviews_normalized_fixed2000_with_llm.csv"

OUT_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\flipkart_exp02_results_bounded"
SPLIT_JSON = os.path.join(OUT_DIR, "flipkart_fixed2000_split_seed42.json")

TARGET_COL = "rating"
TEXT_COL = "review_text"  # if not present, it will be ignored safely

# Fixed subset size (per your methodology)
N_ROWS = 2000
SEED = 42
TEST_SIZE = 0.20

# Deep Ensemble settings (laptop-friendly)
N_ENSEMBLE = 5
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10

# Model size
HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.10

# Rating bounds (IMPORTANT)
RATING_MIN = 1.0
RATING_MAX = 5.0
# =========================================================


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_fixed_2000(path: str) -> pd.DataFrame:
    df_full = pd.read_csv(path)
    df = df_full.head(N_ROWS).copy()
    return df


def make_or_load_split(n: int, split_path: str):
    """
    Create a fixed split ONCE, then reuse it for reproducibility.
    """
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        train_idx = np.array(d["train_idx"], dtype=int)
        test_idx = np.array(d["test_idx"], dtype=int)
        return train_idx, test_idx

    rng = np.random.default_rng(SEED)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    n_test = int(round(n * TEST_SIZE))
    test_idx = all_idx[:n_test]
    train_idx = all_idx[n_test:]

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "test_size": TEST_SIZE,
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            },
            f,
            indent=2,
        )

    return train_idx, test_idx


def pick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safer feature selection:
    - drop target + text
    - drop obvious ID-like numeric columns (can destabilize learning)
    - keep numeric only (assumes your normalized pipeline already encoded categoricals)
    """
    drop_cols = {TARGET_COL}

    if TEXT_COL in df.columns:
        drop_cols.add(TEXT_COL)

    # Drop common non-feature columns if present
    for c in [
        "id", "ID", "pk", "PK",
        "row", "row_id", "row_index", "index",
        "product_id", "user_id", "review_id",
        "timestamp", "date", "datetime"
    ]:
        if c in df.columns:
            drop_cols.add(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Keep numeric only
    X_num = X.select_dtypes(include=[np.number]).copy()

    if X_num.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns found after dropping target/text columns.\n"
            "Your CSV may still contain only text/categorical columns.\n"
            "Encode categoricals before training, or specify feature columns explicitly."
        )

    # Also remove any all-constant columns (rare but can happen)
    nunique = X_num.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_num = X_num.drop(columns=const_cols)

    return X_num


def build_model(input_dim: int) -> tf.keras.Model:
    """
    BOUNDED OUTPUT:
    - model outputs sigmoid in [0,1]
    - we scale predictions to [RATING_MIN, RATING_MAX] outside the model
    """
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(HIDDEN_1, activation="relu")(inp)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(HIDDEN_2, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    # bounded in [0,1]
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="mse"
    )
    return model


def scale_01_to_rating(y01: np.ndarray) -> np.ndarray:
    # [0,1] -> [RATING_MIN, RATING_MAX]
    return RATING_MIN + (RATING_MAX - RATING_MIN) * y01


def main():
    set_all_seeds(SEED)
    ensure_dir(OUT_DIR)

    df = load_fixed_2000(DATA_CSV)

    if TARGET_COL not in df.columns:
        raise KeyError(f"TARGET_COL='{TARGET_COL}' not found. Available columns: {list(df.columns)}")

    y = df[TARGET_COL].astype(float).values

    X = pick_features(df)
    feature_names = X.columns.tolist()

    # fixed split
    train_idx, test_idx = make_or_load_split(len(df), SPLIT_JSON)

    X_train = X.iloc[train_idx].values
    y_train = y[train_idx]
    X_test = X.iloc[test_idx].values
    y_test = y[test_idx]

    # IMPORTANT: map y_train into [0,1] to match sigmoid output training
    y_train_01 = (y_train - RATING_MIN) / (RATING_MAX - RATING_MIN)
    y_train_01 = np.clip(y_train_01, 0.0, 1.0)

    # Scale features using TRAIN only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    preds_members = []
    history_summaries = []

    for m in range(N_ENSEMBLE):
        member_seed = SEED + 1000 + m
        set_all_seeds(member_seed)

        # Bootstrap training set (classic deep ensemble)
        rng = np.random.default_rng(member_seed)
        boot_idx = rng.integers(low=0, high=len(X_train_s), size=len(X_train_s))
        X_boot = X_train_s[boot_idx]
        y_boot = y_train_01[boot_idx]

        model = build_model(input_dim=X_train_s.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]

        hist = model.fit(
            X_boot, y_boot,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=callbacks
        )

        # Predict in [0,1], then scale to [1,5]
        y_pred_01 = model.predict(X_test_s, verbose=0).reshape(-1)
        y_pred = scale_01_to_rating(y_pred_01)

        preds_members.append(y_pred)

        history_summaries.append({
            "member": m + 1,
            "seed": member_seed,
            "epochs_trained": int(len(hist.history.get("loss", []))),
            "best_val_loss": float(np.min(hist.history.get("val_loss", [np.nan])))
        })

        print(f"Ensemble member {m+1}/{N_ENSEMBLE} done. (epochs={history_summaries[-1]['epochs_trained']})")

    preds_members = np.vstack(preds_members)  # (M, n_test)
    pred_mean = preds_members.mean(axis=0)
    pred_std = preds_members.std(axis=0)

    # Final safety (should already be bounded, but keep it robust)
    pred_mean = np.clip(pred_mean, RATING_MIN, RATING_MAX)

    # Metrics (compatible with older sklearn)
    mae = mean_absolute_error(y_test, pred_mean)
    mse = mean_squared_error(y_test, pred_mean)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, pred_mean)

    print("\n===== Experiment-02 Results (Flipkart | Deep Ensembles | Base + LLM | BOUNDED) =====")
    print(f"Rows used : {len(df)} (fixed)")
    print(f"Train/Test: {len(train_idx)}/{len(test_idx)} (seed={SEED})")
    print(f"Features  : {len(feature_names)} numeric")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "row_index": test_idx,
        "y_true": y_test,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    })
    for m in range(N_ENSEMBLE):
        pred_df[f"pred_m{m+1}"] = preds_members[m]

    pred_path = os.path.join(OUT_DIR, "flipkart_deep_ensembles_exp02_predictions_bounded.csv")
    pred_df.to_csv(pred_path, index=False)

    # Save summary
    summary = {
        "dataset_csv": DATA_CSV,
        "target": TARGET_COL,
        "rows_fixed": int(len(df)),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "rating_range": [RATING_MIN, RATING_MAX],
        "n_ensemble": N_ENSEMBLE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "features_used_numeric": feature_names,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
        "ensemble_histories": history_summaries,
        "predictions_csv": pred_path,
    }

    summary_path = os.path.join(OUT_DIR, "flipkart_deep_ensembles_exp02_summary_bounded.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved summary    : {summary_path}")


if __name__ == "__main__":
    main()
    