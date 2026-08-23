import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "flipkart_reviews_normalized.csv"
TARGET_COL = "rating"

FEATURE_COLS = [
    "upvotes",
    "downvotes",
    "date_ordinal",
    "location_encoded"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)

# Safety check: ensure columns exist
required_cols = FEATURE_COLS + [TARGET_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# ============================================================
# PREPARE FEATURES & TARGET
# ============================================================
X = df[FEATURE_COLS]
y = df[TARGET_COL]

print("\nFeatures used:", FEATURE_COLS)
print("Target:", TARGET_COL)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)

# ============================================================
# MODEL: BASELINE LINEAR REGRESSION
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================================================
# PREDICTION
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# EVALUATION METRICS
# NOTE: We compute RMSE as sqrt(MSE) for sklearn compatibility
# ============================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)      # no 'squared=' arg used
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== Baseline Regression Results (Flipkart) =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

# ============================================================
# COEFFICIENTS (INTERPRETABILITY)
# ============================================================
coef_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nModel coefficients:")
print(coef_df.to_string(index=False))

print("\nIntercept:", float(model.intercept_))
