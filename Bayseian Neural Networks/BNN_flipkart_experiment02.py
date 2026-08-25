import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# EXPERIMENT-02 TRAINING (Base + LLM features)
# Uses your already-saved file:
# D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\data_subsets\flipkart_2000_seed42_llm.csv
# ============================================================

# -------- CONFIG --------
DATA_FILE = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\data_subsets\flipkart_2000_seed42_llm.csv"

TARGET_COL = "rating"   # Flipkart target (change only if your CSV uses a different target name)

# Base numeric features (from your Experiment-01)
BASE_FEATURES = ["upvotes", "downvotes", "date_ordinal", "location_encoded"]

# LLM feature columns (adjust ONLY if your CSV has slightly different names)
LLM_FEATURES = [
    "llm_sentiment",
    "llm_product_quality",
    "llm_value_for_money",
    "llm_skin_suitability",
    "llm_ingredient_concern"
]

# Model + split
TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0

# If True: keep only rows where at least one LLM feature is non-zero
# If False: use all rows (including rows where extraction failed and features are all zeros)
USE_ONLY_SUCCESSFUL_LLM_ROWS = True

# ------------------------
def main():
    df = pd.read_csv(DATA_FILE)
    print("Loaded:", df.shape)

    # Verify required columns exist
    required_cols = BASE_FEATURES + LLM_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Optional: filter only rows where LLM extraction produced something non-zero
    if USE_ONLY_SUCCESSFUL_LLM_ROWS:
        llm_sum = df[LLM_FEATURES].sum(axis=1)
        before = len(df)
        df = df.loc[llm_sum != 0].copy()
        after = len(df)
        print(f"Filtered to successful LLM rows only: {before} -> {after}")

    # Prepare X/y
    FEATURE_COLS = BASE_FEATURES + LLM_FEATURES
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print("\nFeatures used:", FEATURE_COLS)
    print("Target:", TARGET_COL)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("\nTrain size:", X_train.shape)
    print("Test size :", X_test.shape)

    # Train (Ridge is stable when adding LLM features)
    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Predict + metrics
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== Experiment-02 Results (Flipkart: Base + LLM features) =====")
    print(f"Rows used: {len(df)}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Coefficients (optional but useful)
    coef = pd.Series(model.coef_, index=FEATURE_COLS).sort_values(key=np.abs, ascending=False)
    print("\nTop coefficients by absolute magnitude:")
    print(coef.head(12).to_string())

if __name__ == "__main__":
    main()
