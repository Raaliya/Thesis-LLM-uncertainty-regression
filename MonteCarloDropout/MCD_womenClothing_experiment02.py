import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Women Clothing | Experiment 02 | Monte Carlo Dropout
# Baseline (from normalized dataset) + LLM features (separate CSV)
# Fixed 2000 rows + seed 42 for fair comparison
# ============================================================

RANDOM_STATE = 42
N_ROWS = 2000
TEST_SIZE = 0.2

MC_SAMPLES = 20
EPOCHS = 80
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# ---- FILES ----
BASE_CSV = r"women_clothing_reviews_normalized.csv"  # <-- your normalized women clothing dataset (has Age/Rating/etc.)
LLM_CSV  = r"outputs_exp02_women_clothing_ollama_2000\women_clothing_llm_features_2000.csv"

# ---- TARGET (women dataset target is Rating) ----
TARGET_COL = "Rating"

# ---- BASE FEATURES (from Exp01 baseline) ----
BASE_FEATURES = [
    "Age",
    "Recommended IND",
    "Positive Feedback Count"
]

# ============================================================
def build_mc_model(input_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x, training=True)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x, training=True)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


def main():
    # --------------------------
    # Load baseline dataset
    # --------------------------
    base = pd.read_csv(BASE_CSV)

    # Ensure numeric target
    base[TARGET_COL] = pd.to_numeric(base[TARGET_COL], errors="coerce")
    base = base.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Fixed sampling (same as Exp01)
    if len(base) < N_ROWS:
        raise ValueError(f"Baseline dataset has only {len(base)} rows; cannot sample {N_ROWS}.")
    base = base.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    # --------------------------
    # Load LLM features
    # --------------------------
    llm = pd.read_csv(LLM_CSV)

    llm_cols = [c for c in llm.columns if c.startswith("llm_")]
    if len(llm_cols) == 0:
        raise ValueError("No LLM columns found in LLM file. Expected columns starting with 'llm_'.")

    # Align row counts safely (should both be 2000)
    min_rows = min(len(base), len(llm))
    base = base.iloc[:min_rows].reset_index(drop=True)
    llm = llm.iloc[:min_rows].reset_index(drop=True)

    # Merge horizontally by row order (this is why fixed seed sampling is important)
    df = pd.concat([base, llm[llm_cols]], axis=1)

    print("Baseline loaded:", base.shape)
    print("LLM loaded     :", llm.shape)
    print("Merged rows    :", df.shape[0])
    print("LLM features used:", len(llm_cols))

    # --------------------------
    # Prepare X, y
    # --------------------------
    features = BASE_FEATURES + llm_cols

    # Make sure all features are numeric
    for col in features + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaNs (important)
    df[features] = df[features].fillna(df[features].median())
    df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].median())

    X = df[features].values
    y = df[TARGET_COL].values

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --------------------------
    # Train MC Dropout model
    # --------------------------
    model = build_mc_model(X_train.shape[1])

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # --------------------------
    # Monte Carlo predictions
    # --------------------------
    preds = []
    for _ in range(MC_SAMPLES):
        preds.append(model(X_test, training=True).numpy().flatten())

    preds = np.array(preds)
    y_pred = preds.mean(axis=0)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print("\n===== Experiment 02 Results (Women | MC Dropout | Baseline + LLM) =====")
    print(f"Rows used: {len(df)}")
    print(f"LLM features used: {len(llm_cols)}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")


if __name__ == "__main__":
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    main()
