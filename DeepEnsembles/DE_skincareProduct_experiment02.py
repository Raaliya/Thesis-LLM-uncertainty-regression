import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# CONFIG
# =========================================================
ROOT = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\skincare"

INPUT_CSV = os.path.join(ROOT, "reviews_1000_1500_with_llm_features_2000.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs_exp02_deep_ensembles_skincare")

TARGET = "rating"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Deep Ensemble settings
N_ENSEMBLES = 5
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "de_exp02_predictions.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "de_exp02_metrics.json")
SCALER_PATH = os.path.join(OUTPUT_DIR, "de_exp02_scaler.joblib")
IMPUTER_PATH = os.path.join(OUTPUT_DIR, "de_exp02_imputer.joblib")
FEATURES_TXT = os.path.join(OUTPUT_DIR, "de_exp02_features.txt")
MODELS_DIR = os.path.join(OUTPUT_DIR, "ensemble_models")

# =========================================================
# REPRODUCIBILITY
# =========================================================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================================
# SETUP
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading Experiment 02 dataset...")
df = pd.read_csv(INPUT_CSV, low_memory=False)

print(f"Dataset shape: {df.shape}")

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

# =========================================================
# CLEAN TARGET
# =========================================================
df = df.copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

print(f"Rows after dropping missing target: {len(df)}")

# =========================================================
# KEEP ONLY NUMERIC COLUMNS
# This keeps numeric baseline features + numeric LLM features
# =========================================================
numeric_df = df.select_dtypes(include=[np.number]).copy()

if TARGET not in numeric_df.columns:
    raise ValueError(f"Target '{TARGET}' is not numeric after conversion.")

feature_cols = [c for c in numeric_df.columns if c != TARGET]

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found for Experiment 02.")

X = numeric_df[feature_cols].copy()
y = numeric_df[TARGET].astype(float).copy()

print(f"Numeric feature count used in Exp-02: {len(feature_cols)}")
print("Features used:")
for col in feature_cols:
    print(f" - {col}")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Train rows: {len(X_train)}")
print(f"Test rows : {len(X_test)}")

# =========================================================
# IMPUTE + SCALE
# =========================================================
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

input_dim = X_train_scaled.shape[1]

# =========================================================
# MODEL BUILDER
# =========================================================
def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_de_model(input_dim, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# =========================================================
# TRAIN ENSEMBLE
# =========================================================
ensemble_predictions = []
histories = []
model_paths = []

print(f"Training Deep Ensemble with {N_ENSEMBLES} members...")

for i in range(N_ENSEMBLES):
    member_seed = RANDOM_STATE + i + 1
    print("\n" + "=" * 80)
    print(f"Training ensemble member {i+1}/{N_ENSEMBLES} | seed={member_seed}")

    set_all_seeds(member_seed)

    model = build_de_model(input_dim=input_dim, learning_rate=LEARNING_RATE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_scaled,
        y_train.values,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )
    histories.append(history)

    preds = model.predict(X_test_scaled, verbose=0).reshape(-1)
    ensemble_predictions.append(preds)

    member_path = os.path.join(MODELS_DIR, f"de_member_{i+1}.keras")
    model.save(member_path)
    model_paths.append(member_path)

# shape = [n_ensembles, n_samples]
ensemble_predictions = np.array(ensemble_predictions)

y_pred_mean = ensemble_predictions.mean(axis=0)
y_pred_std = ensemble_predictions.std(axis=0)

# =========================================================
# METRICS
# =========================================================
mae = mean_absolute_error(y_test, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2 = r2_score(y_test, y_pred_mean)

print("\n===== DEEP ENSEMBLES EXPERIMENT 02 RESULTS =====")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"Mean predictive std: {float(np.mean(y_pred_std)):.6f}")

metrics = {
    "experiment": "Exp-02 Deep Ensembles with LLM Features",
    "dataset": "skincare",
    "input_file": INPUT_CSV,
    "target": TARGET,
    "rows_total": int(len(df)),
    "rows_train": int(len(X_train)),
    "rows_test": int(len(X_test)),
    "num_features": int(len(feature_cols)),
    "features_used": feature_cols,
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "mean_predictive_std": float(np.mean(y_pred_std)),
    "n_ensembles": int(N_ENSEMBLES),
    "epochs_max": int(EPOCHS),
    "batch_size": int(BATCH_SIZE),
    "learning_rate": float(LEARNING_RATE),
    "random_state": int(RANDOM_STATE),
    "test_size": float(TEST_SIZE),
    "model": "Deep Ensembles Neural Network",
    "member_model_paths": model_paths
}

# =========================================================
# SAVE PREDICTIONS
# =========================================================
pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred_mean": y_pred_mean,
    "y_pred_std": y_pred_std
})

for i in range(N_ENSEMBLES):
    pred_df[f"member_{i+1}_pred"] = ensemble_predictions[i]

pred_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

# =========================================================
# SAVE METRICS
# =========================================================
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# =========================================================
# SAVE PREPROCESSORS
# =========================================================
joblib.dump(imputer, IMPUTER_PATH)
joblib.dump(scaler, SCALER_PATH)

# =========================================================
# SAVE FEATURE LIST
# =========================================================
with open(FEATURES_TXT, "w", encoding="utf-8") as f:
    f.write("Deep Ensembles Experiment 02 Features Used\n")
    f.write("=" * 45 + "\n")
    for col in feature_cols:
        f.write(f"{col}\n")

print("\nSaved files:")
print(f"- Predictions : {PREDICTIONS_CSV}")
print(f"- Metrics     : {METRICS_JSON}")
print(f"- Imputer     : {IMPUTER_PATH}")
print(f"- Scaler      : {SCALER_PATH}")
print(f"- Features    : {FEATURES_TXT}")
print(f"- Models dir  : {MODELS_DIR}")