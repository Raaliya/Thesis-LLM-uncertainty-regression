import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "flipkart_reviews_exp02_with_llm3_ollama_2000.csv"
TARGET_COL = "rating"

# Base numeric features (same as Experiment-01)
BASE_FEATURES = [
    "upvotes",
    "downvotes",
    "date_ordinal",
    "location_encoded"
]

# LLM qualitative features (Experiment-02)
LLM_FEATURES = [
    "llm_sentiment_polarity",
    "llm_expectation_alignment",
    "llm_perceived_quality",
    "llm_llm_confidence"
]

FEATURE_COLS = BASE_FEATURES + LLM_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0   # stable choice for mixed features

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)

# Safety check
missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# PREPARE FEATURES & TARGET
# ============================================================
X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

print("\nFeatures used:")
for c in FEATURE_COLS:
    print(" -", c)

print("\nTarget:", TARGET_COL)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)

# ============================================================
# MODEL TRAINING (Experiment-02)
# ============================================================
model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# ============================================================
# PREDICTION
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# METRICS
# ============================================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== Experiment-02 Results (Flipkart: Base + LLM Features) =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

# ============================================================
# OPTIONAL: COEFFICIENT INSPECTION
# ============================================================
coef_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=np.abs, ascending=False)

print("\nModel coefficients (sorted by magnitude):")
print(coef_df.to_string(index=False))
