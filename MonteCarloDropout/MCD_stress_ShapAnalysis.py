# mcd_stress_shap.py
# Monte Carlo Dropout + SHAP for Stress Analysis dataset
# Target: confidence
# Notes:
# - confidence is kept as the target and removed from predictors automatically
# - requested feature columns are dropped
# - remaining feature names are renamed with llm_ prefix for consistency

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# USER SETTINGS
# =========================================================
FILE_PATH = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\3 MCD_SHAP LIME"
FILE_NAME = "stress_llm_features_2000.csv"   # change if needed
TARGET_COL = "confidence"

# Columns to drop completely before training and SHAP
# confidence is NOT included here because it is the target
IGNORE_COLS = [
    "source_index",
    "text",
    "overall_sentiment",
    "parse_error",
    "runtime_error"
]

# Train/test settings
TEST_SIZE = 0.20
RANDOM_STATE = 42

# MCD settings
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.20
MC_SAMPLES = 50
GRAD_CLIP = 1.0

# SHAP settings
BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 200
WATERFALL_INDEX = 0
MAX_DISPLAY = 20

# Output files
MODEL_FILE = os.path.join(FILE_PATH, "mcd_stress_model.pt")
SCALER_FILE = os.path.join(FILE_PATH, "mcd_stress_scaler.joblib")
PREDICTIONS_FILE = os.path.join(FILE_PATH, "mcd_stress_predictions.csv")

BAR_PLOT_FILE = os.path.join(FILE_PATH, "mcd_stress_shap_bar.png")
BEESWARM_PLOT_FILE = os.path.join(FILE_PATH, "mcd_stress_shap_beeswarm.png")
WATERFALL_PLOT_FILE = os.path.join(FILE_PATH, "mcd_stress_shap_waterfall.png")


# =========================================================
# REPRODUCIBILITY
# =========================================================
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_actual_target_column(df, requested_target):
    for col in df.columns:
        if col.strip().lower() == requested_target.strip().lower():
            return col
    raise ValueError(f"Target column '{requested_target}' not found in dataset.")


def add_llm_prefix_if_needed(col_name, target_col):
    if col_name == target_col:
        return col_name
    if col_name.startswith("llm_"):
        return col_name
    return f"llm_{col_name}"


def sanitize_dataframe_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    df_num = df_num.copy()
    df_num = df_num.replace([np.inf, -np.inf], np.nan)

    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
        median_val = df_num[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_num[col] = df_num[col].fillna(median_val)

        lower = df_num[col].quantile(0.01)
        upper = df_num[col].quantile(0.99)

        if pd.isna(lower):
            lower = df_num[col].min()
        if pd.isna(upper):
            upper = df_num[col].max()

        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            df_num[col] = df_num[col].clip(lower=lower, upper=upper)

    df_num = df_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_num


class MCDRegressor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.out(x)
        return x


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def train_mcd_model(model, train_loader, criterion, optimizer, epochs=300, device="cpu"):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                continue

            optimizer.zero_grad()
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            valid_batches += 1

        if valid_batches == 0:
            raise ValueError("Training failed: all batches produced non-finite values.")

        epoch_loss = running_loss / len(train_loader.dataset)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")


def mc_predict(model, X_tensor, mc_samples=50, device="cpu"):
    model.to(device)
    model.eval()
    enable_dropout(model)

    preds = []
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        for _ in range(mc_samples):
            pred = model(X_tensor).squeeze(1)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            preds.append(pred.cpu().numpy())

    preds = np.array(preds)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    pred_mean = np.nan_to_num(pred_mean, nan=0.0, posinf=0.0, neginf=0.0)
    pred_std = np.nan_to_num(pred_std, nan=0.0, posinf=0.0, neginf=0.0)

    return pred_mean, pred_std


def predict_from_raw_for_shap(X_raw, model, scaler, feature_names, mc_samples=50, device="cpu"):
    X_df = pd.DataFrame(X_raw, columns=feature_names)
    X_scaled = scaler.transform(X_df)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = np.clip(X_scaled, -10, 10)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    pred_mean, _ = mc_predict(model, X_tensor, mc_samples=mc_samples, device=device)
    return pred_mean


def style_axes(ax):
    ax.set_facecolor("#e8e8e8")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)


def save_bar_plot(shap_values, out_file):
    plt.figure(figsize=(8, 8), facecolor="#e8e8e8")
    shap.plots.bar(shap_values, max_display=MAX_DISPLAY, show=False)
    fig = plt.gcf()
    fig.set_facecolor("#e8e8e8")
    ax = plt.gca()
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_beeswarm_plot(shap_values, out_file):
    plt.figure(figsize=(8, 8), facecolor="#e8e8e8")
    shap.plots.beeswarm(shap_values, max_display=MAX_DISPLAY, show=False)
    fig = plt.gcf()
    fig.set_facecolor("#e8e8e8")
    ax = plt.gca()
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def save_waterfall_plot(single_shap_value, out_file):
    plt.figure(figsize=(9, 8), facecolor="#e8e8e8")
    shap.plots.waterfall(single_shap_value, max_display=MAX_DISPLAY, show=False)
    fig = plt.gcf()
    fig.set_facecolor("#e8e8e8")
    ax = plt.gca()
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# =========================================================
# LOAD DATA
# =========================================================
full_file = os.path.join(FILE_PATH, FILE_NAME)
if not os.path.exists(full_file):
    raise FileNotFoundError(f"File not found: {full_file}")

df = pd.read_csv(full_file)
print(f"Loaded data shape: {df.shape}")

actual_target_col = find_actual_target_column(df, TARGET_COL)
print(f"Detected target column: {actual_target_col}")

existing_ignore = [c for c in IGNORE_COLS if c in df.columns]
df = df.drop(columns=existing_ignore, errors="ignore")

print("\nDropped columns:")
for col in existing_ignore:
    print(f" - {col}")

# rename remaining columns with llm_ prefix for consistency
rename_map = {}
for col in df.columns:
    rename_map[col] = add_llm_prefix_if_needed(col, actual_target_col)

df = df.rename(columns=rename_map)
actual_target_col = rename_map.get(actual_target_col, actual_target_col)

print("\nRenamed feature columns for report consistency:")
for old_name, new_name in rename_map.items():
    if old_name != new_name:
        print(f" - {old_name} --> {new_name}")

# separate target and predictors
y = pd.to_numeric(df[actual_target_col], errors="coerce")
y = y.replace([np.inf, -np.inf], np.nan)

X = df.drop(columns=[actual_target_col], errors="ignore").copy()

# keep only numeric features
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

if len(non_numeric_cols) > 0:
    print("\nDropped non-numeric columns:")
    for col in non_numeric_cols:
        print(f" - {col}")

X = X[numeric_cols].copy()
X = sanitize_dataframe_numeric(X)

valid_mask = y.notna()
X = X.loc[valid_mask].reset_index(drop=True)
y = y.loc[valid_mask].reset_index(drop=True)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Final target shape: {y.shape}")

if X.shape[1] == 0:
    raise ValueError("No usable numeric predictor columns remained after preprocessing.")

feature_names = X.columns.tolist()

print("\nFinal features used for training and SHAP:")
for col in feature_names:
    print(f" - {col}")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Feature matrix still contains non-finite values after cleaning.")

if not np.isfinite(y.to_numpy()).all():
    raise ValueError("Target still contains non-finite values after cleaning.")


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")


# =========================================================
# SCALE FEATURES
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_test_scaled = np.clip(X_test_scaled, -10, 10)

joblib.dump(scaler, SCALER_FILE)
print(f"\nScaler saved to: {SCALER_FILE}")

if not np.isfinite(X_train_scaled).all():
    raise ValueError("X_train_scaled contains non-finite values.")
if not np.isfinite(X_test_scaled).all():
    raise ValueError("X_test_scaled contains non-finite values.")


# =========================================================
# PREPARE PYTORCH DATA
# =========================================================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# =========================================================
# TRAIN MCD MODEL
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

model = MCDRegressor(input_dim=X_train.shape[1], dropout_rate=DROPOUT_RATE)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

print("\nTraining MCD model...")
train_mcd_model(model, train_loader, criterion, optimizer, epochs=EPOCHS, device=device)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": X_train.shape[1],
    "feature_names": feature_names,
    "target_col": actual_target_col,
    "ignore_cols": IGNORE_COLS,
    "dropout_rate": DROPOUT_RATE
}, MODEL_FILE)
print(f"Model saved to: {MODEL_FILE}")


# =========================================================
# EVALUATION
# =========================================================
y_pred_mean, y_pred_std = mc_predict(model, X_test_tensor, mc_samples=MC_SAMPLES, device=device)

y_pred_mean = np.nan_to_num(y_pred_mean, nan=0.0, posinf=0.0, neginf=0.0)
y_pred_std = np.nan_to_num(y_pred_std, nan=0.0, posinf=0.0, neginf=0.0)

mae = mean_absolute_error(y_test, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2 = r2_score(y_test, y_pred_mean)

print("\n===== MCD STRESS ANALYSIS RESULTS =====")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"Mean predictive std: {y_pred_std.mean():.6f}")

pred_df = pd.DataFrame({
    "actual": y_test.values,
    "pred_mean": y_pred_mean,
    "pred_std": y_pred_std
})
pred_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"Predictions saved to: {PREDICTIONS_FILE}")


# =========================================================
# SHAP EXPLAINER
# =========================================================
background_size = min(BACKGROUND_SIZE, len(X_train))
explain_size = min(EXPLAIN_SIZE, len(X_test))
waterfall_index = min(WATERFALL_INDEX, explain_size - 1)

X_background = X_train.sample(n=background_size, random_state=RANDOM_STATE).copy()
X_explain = X_test.iloc[:explain_size].copy()

print(f"\nBackground samples for SHAP: {len(X_background)}")
print(f"Samples explained by SHAP  : {len(X_explain)}")

print("\nCreating SHAP explainer...")
explainer = shap.Explainer(
    lambda X_raw: predict_from_raw_for_shap(
        X_raw=X_raw,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        mc_samples=MC_SAMPLES,
        device=device
    ),
    X_background,
    feature_names=feature_names
)

print("Computing SHAP values...")
shap_values = explainer(X_explain)


# =========================================================
# SAVE INDIVIDUAL SHAP PLOTS
# =========================================================
print("\nSaving SHAP bar plot...")
save_bar_plot(shap_values, BAR_PLOT_FILE)

print("Saving SHAP beeswarm plot...")
save_beeswarm_plot(shap_values, BEESWARM_PLOT_FILE)

print("Saving SHAP waterfall plot...")
save_waterfall_plot(shap_values[waterfall_index], WATERFALL_PLOT_FILE)


# =========================================================
# DONE
# =========================================================
print("\nAll files created successfully.")
print(f"Model file       : {MODEL_FILE}")
print(f"Scaler file      : {SCALER_FILE}")
print(f"Predictions file : {PREDICTIONS_FILE}")
print(f"Bar plot         : {BAR_PLOT_FILE}")
print(f"Beeswarm plot    : {BEESWARM_PLOT_FILE}")
print(f"Waterfall plot   : {WATERFALL_PLOT_FILE}")

