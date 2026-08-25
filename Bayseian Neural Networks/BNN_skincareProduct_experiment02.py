import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# EXPERIMENT-02 TRAINING (Skincare: Base + LLM features)
# NaN-safe version
# ============================================================

DATA_FILE = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\data_subsets\skincare_2000_seed42_llm.csv"

TARGET_COL = "rating"

BASE_FEATURES = [
    "loves_count",
    "reviews",
    "price_usd",
    "value_price_usd",
    "sale_price_usd",
    "child_count",
    "child_max_price",
    "child_min_price"
]

LLM_FEATURES = [
    "llm_sentiment",
    "llm_product_quality",
    "llm_value_for_money",
    "llm_skin_suitability",
    "llm_ingredient_concern"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0

USE_ONLY_SUCCESSFUL_LLM_ROWS = True


def main():
    df = pd.read_csv(DATA_FILE)
    print("Loaded:", df.shape)

    required_cols = BASE_FEATURES + LLM_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if USE_ONLY_SUCCESSFUL_LLM_ROWS:
        llm_sum = df[LLM_FEATURES].sum(axis=1)
        before = len(df)
        df = df.loc[llm_sum != 0].copy()
        after = len(df)
        print(f"Filtered to successful LLM rows only: {before} -> {after}")

    FEATURE_COLS = BASE_FEATURES + LLM_FEATURES

    # ---- Convert to numeric safely ----
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Fill NaNs with median (important fix) ----
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].median())

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print("\nFeatures used:", FEATURE_COLS)
    print("Target:", TARGET_COL)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("\nTrain size:", X_train.shape)
    print("Test size :", X_test.shape)

    model = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== Experiment-02 Results (Skincare: Base + LLM features) =====")
    print(f"Rows used: {len(df)}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    coef = pd.Series(model.coef_, index=FEATURE_COLS)
    coef_sorted = coef.sort_values(key=np.abs, ascending=False)

    print("\nTop coefficients by absolute magnitude:")
    print(coef_sorted.head(12).to_string())


if __name__ == "__main__":
    main()
