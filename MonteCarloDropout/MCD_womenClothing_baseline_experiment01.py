import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# Experiment 01 (Baseline) | Women Clothing Reviews
# Target: Rating
# =========================================================

RANDOM_STATE = 42
N_ROWS = 2000
TARGET = "Rating"

DATA_PATH = "women_clothing_reviews_normalized.csv"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

# ---------------------------------------------------------
# Fixed 2000-row sampling
# ---------------------------------------------------------
if len(df) < N_ROWS:
    print(f"[INFO] Dataset has only {len(df)} rows. Using all.")
    df = df.sample(n=len(df), random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df = df.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

print("Rows used:", len(df))

# ---------------------------------------------------------
# Drop non-predictive text columns
# ---------------------------------------------------------
DROP_COLUMNS = [
    "Clothing ID",
    "Title",
    "Review Text"
]

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

# ---------------------------------------------------------
# Split numeric and categorical columns
# ---------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove(TARGET)

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# ---------------------------------------------------------
# Prepare X / y
# ---------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float)

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ---------------------------------------------------------
# Model (same style as other datasets)
# ---------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== Experiment 01 Results (Baseline | Women Clothing | fixed rows) =====")
print(f"Rows used: {len(df)}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")
