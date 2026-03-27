import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==============================
# 1. Load dataset
# ==============================
df = pd.read_csv("amazon_reviews.csv")   # change filename if needed

# ==============================
# 2. Define target and features
# ==============================
TARGET = "Score"

X = df.drop(columns=[TARGET])
y = df[TARGET]   # regression target (DO NOT NORMALISE)

# ==============================
# 3. Columns to scale
# ==============================
NUMERIC_COLUMNS = [
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
    "Time"
]

# ==============================
# 4. Columns to exclude
# ==============================
EXCLUDE_COLUMNS = [
    "Id",
    "ProductId",
    "UserId",
    "ProfileName",
    "Summary",
    "Text"
]

# Keep only numeric columns for scaling
X_numeric = X[NUMERIC_COLUMNS]

# ==============================
# 5. Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 6. Normalisation pipeline
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 7. Convert back to DataFrame
# ==============================
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=NUMERIC_COLUMNS,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=NUMERIC_COLUMNS,
    index=X_test.index
)

# ==============================
# 8. Done
# ==============================
print("Normalisation completed successfully.")
print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)
