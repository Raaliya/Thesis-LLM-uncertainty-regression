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
OUTPUT_DIR = os.path.join(ROOT, "outputs_exp02_mcd_skincare")

TARGET = "rating"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# MCD settings
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.30
MC_PASSES = 50

MODEL_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "mcd_exp02_model.weights.h5")
SCALER_PATH = os.path.join(OUTPUT_DIR, "mcd_exp02_scaler.joblib")
IMPUTER_PATH = os.path.join(OUTPUT_DIR, "mcd_exp02_imputer.joblib")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "mcd_exp02_predictions.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "mcd_exp02_metrics.json")
FEATURES_TXT = os.path.join(OUTPUT_DIR, "mcd_exp02_features.txt")

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
# This keeps baseline numeric fields + numeric LLM features
# and ignores string/text columns automatically.
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

# =========================================================
# BUILD MCD MODEL
# =========================================================
class MCDModel(tf.keras.Model):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.dp1 = tf.keras.layers.Dropout(dropout_rate)
        self.d2 = tf.keras.layers.Dense(64, activation="relu")
        self.dp2 = tf.keras.layers.Dropout(dropout_rate)
        self.d3 = tf.keras.layers.Dense(32, activation="relu")
        self.dp3 = tf.keras.layers.Dropout(dropout_rate)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.d1(inputs)
        x = self.dp1(x, training=training)
        x = self.d2(x)
        x = self.dp2(x, training=training)
        x = self.d3(x)
        x = self.dp3(x, training=training)
        return self.out(x)

input_dim = X_train_scaled.shape[1]
model = MCDModel(input_dim=input_dim, dropout_rate=DROPOUT_RATE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

# Build once
_ = model(tf.convert_to_tensor(X_train_scaled[:1], dtype=tf.float32), training=False)

print("Training MCD Experiment 02 model...")

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

# =========================================================
# MONTE CARLO PREDICTION
# =========================================================
print(f"Running Monte Carlo Dropout with {MC_PASSES} passes...")

X_test_tensor = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)

mc_predictions = []
for _ in range(MC_PASSES):
    preds = model(X_test_tensor, training=True).numpy().reshape(-1)
    mc_predictions.append(preds)

mc_predictions = np.array(mc_predictions)  # [MC_PASSES, N]

y_pred_mean = mc_predictions.mean(axis=0)
y_pred_std = mc_predictions.std(axis=0)

# =========================================================
# METRICS
# =========================================================
mae = mean_absolute_error(y_test, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2 = r2_score(y_test, y_pred_mean)

print("\n===== MCD EXPERIMENT 02 RESULTS =====")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"Mean predictive std: {float(np.mean(y_pred_std)):.6f}")

metrics = {
    "experiment": "Exp-02 MCD with LLM Features",
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
    "mc_passes": int(MC_PASSES),
    "epochs_trained": int(len(history.history["loss"])),
    "batch_size": int(BATCH_SIZE),
    "dropout_rate": float(DROPOUT_RATE),
    "learning_rate": float(LEARNING_RATE),
    "random_state": int(RANDOM_STATE),
    "test_size": float(TEST_SIZE),
    "model": "Monte Carlo Dropout Neural Network"
}

# =========================================================
# SAVE PREDICTIONS
# =========================================================
pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred_mean": y_pred_mean,
    "y_pred_std": y_pred_std
})

pred_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

# =========================================================
# SAVE METRICS
# =========================================================
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# =========================================================
# SAVE MODEL WEIGHTS + PREPROCESSORS
# =========================================================
model.save_weights(MODEL_WEIGHTS_PATH)
joblib.dump(imputer, IMPUTER_PATH)
joblib.dump(scaler, SCALER_PATH)

# =========================================================
# SAVE FEATURE LIST
# =========================================================
with open(FEATURES_TXT, "w", encoding="utf-8") as f:
    f.write("MCD Experiment 02 Features Used\n")
    f.write("=" * 40 + "\n")
    for col in feature_cols:
        f.write(f"{col}\n")

print("\nSaved files:")
print(f"- Predictions : {PREDICTIONS_CSV}")
print(f"- Metrics     : {METRICS_JSON}")
print(f"- Weights     : {MODEL_WEIGHTS_PATH}")
print(f"- Imputer     : {IMPUTER_PATH}")
print(f"- Scaler      : {SCALER_PATH}")
print(f"- Features    : {FEATURES_TXT}")

