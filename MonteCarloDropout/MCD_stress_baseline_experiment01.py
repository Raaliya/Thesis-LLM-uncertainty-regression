import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# Experiment 01 (Baseline) | Stress Analysis | fixed rows
# Target: confidence
# =========================================================

RANDOM_STATE = 42
N_ROWS = 2000
TARGET = "confidence"

DATA_PATH = "stress_analysis_normalized.csv"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Ensure target is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# Drop rows with missing target
df = df.dropna(subset=[TARGET]).copy()

# Fixed 2000-row sampling (same rule as other datasets)
if len(df) < N_ROWS:
    print(f"[WARN] Only {len(df)} rows available. Using all.")
    df = df.sample(n=len(df), random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df = df.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

print("Rows used:", len(df))

# ---------------------------------------------------------
# Drop non-predictive columns (IDs, text, timestamps)
# ---------------------------------------------------------
DROP_COLUMNS = [
    "id",
    "post_id",
    "sentence_range",
    "text",
    "subreddit",
    "social_timestamp",
]

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

# ---------------------------------------------------------
# Define feature columns
# ---------------------------------------------------------
NUMERIC_COLUMNS = [c for c in df.columns if c != TARGET]

# ---------------------------------------------------------
# Prepare X / y
# ---------------------------------------------------------
X = df[NUMERIC_COLUMNS]
y = df[TARGET].astype(float)

# ---------------------------------------------------------
# Preprocessing (numeric only)
# ---------------------------------------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_COLUMNS),
    ],
    remainder="drop"
)

# ---------------------------------------------------------
# Model (IDENTICAL style to previous experiments)
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
# Train / Test split (fixed seed)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------
# Train & Evaluate
# ---------------------------------------------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== Experiment 01 Results (Baseline | Stress Analysis | fixed rows) =====")
print(f"Rows used: {len(df)}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")
