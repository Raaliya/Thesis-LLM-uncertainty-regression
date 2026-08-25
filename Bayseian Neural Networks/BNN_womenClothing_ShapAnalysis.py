# ============================================================
# SHAP Explanation for Women Clothing Dataset - BNN Model
# Generates: Bar Plot, Beeswarm Plot, Waterfall Plot, Combined Figure
# Baseline features + best 15 LLM-derived features
# ============================================================

import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import shap

warnings.filterwarnings("ignore")

# ============================================================
# 1. USER SETTINGS
# ============================================================

DATASET_PATH = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\1 BNN_ShapLime\women_clothing_reviews_llm_2000_seed42.csv"

OUTPUT_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\1 BNN_ShapLime\SHAP_BNN_Women"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "Rating"

DROP_COLUMNS = [
    "Unnamed: 0",
    "Clothing ID",
    "Title",
    "Review Text",
    "Rating",
    "Division Name",
    "Department Name",
    "Class Name",
    "row_index",
    "llm_overall_sentiment",
    "llm_confidence",
    "Recommended IND"
]

TOP_LLM_FEATURES = 15

TEST_SIZE = 0.20
RANDOM_STATE = 42

EPOCHS = 600
BATCH_SIZE = 64
LEARNING_RATE = 0.001
KL_WEIGHT = 1e-4

SHAP_BACKGROUND_SIZE = 80
SHAP_EXPLAIN_SIZE = 200
WATERFALL_INSTANCE_INDEX = 0

# This will be updated automatically after selecting features
MAX_DISPLAY = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def make_numeric(series):
    """
    Converts common Boolean/string formats to numeric.
    Keeps numeric columns as numeric.
    """
    if series.dtype == bool:
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "recommended": 1,
        "not recommended": 0,
        "none": np.nan,
        "nan": np.nan,
        "": np.nan,
    }

    mapped = s.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")

    if mapped.notna().sum() > numeric.notna().sum():
        return mapped

    return numeric


def is_llm_feature(col):
    """
    Identifies LLM-derived features.
    """
    col = str(col)

    if col.startswith("llm_"):
        return True

    if (
        col.endswith("_present")
        or col.endswith("_polarity")
        or col.endswith("_intensity")
    ):
        return True

    return False


def clean_display_name(col, is_llm=False):
    """
    Creates readable SHAP display names.
    LLM-derived features are given the prefix 'llm'.
    """
    name = str(col)

    name = name.replace("llm_", "")
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    if is_llm:
        return f"llm {name}"

    return name


def ensure_unique_names(names):
    """
    Avoids duplicate display names after cleaning.
    """
    seen = {}
    final_names = []

    for name in names:
        if name not in seen:
            seen[name] = 0
            final_names.append(name)
        else:
            seen[name] += 1
            final_names.append(f"{name} ({seen[name]})")

    return final_names


# ============================================================
# 4. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    alt_path = DATASET_PATH.replace(".csv", "")
    if os.path.exists(alt_path):
        DATASET_PATH = alt_path
    else:
        raise FileNotFoundError(f"Dataset not found:\n{DATASET_PATH}")

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("\nDataset loaded successfully")
print("Path :", DATASET_PATH)
print("Shape:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' was not found.\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )

# Convert target to numeric before removing display/drop columns
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

# Drop unwanted columns from the feature-selection dataframe,
# but keep the target column for modelling.
feature_drop_columns = [
    c for c in DROP_COLUMNS
    if c in df.columns and c != TARGET_COL
]

candidate_df = df.drop(columns=feature_drop_columns, errors="ignore").copy()

# Remove rows with missing target
candidate_df = candidate_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# ============================================================
# 5. IDENTIFY BASELINE AND LLM FEATURES
# ============================================================

for col in candidate_df.columns:
    if col != TARGET_COL:
        candidate_df[col] = make_numeric(candidate_df[col])

all_numeric_features = [
    c for c in candidate_df.columns
    if c != TARGET_COL
    and pd.api.types.is_numeric_dtype(candidate_df[c])
    and candidate_df[c].notna().sum() > 0
]

llm_candidate_features = [
    c for c in all_numeric_features
    if is_llm_feature(c)
]

baseline_features = [
    c for c in all_numeric_features
    if not is_llm_feature(c)
]

if len(baseline_features) == 0:
    print("\nWarning: No baseline numeric features were found after dropping metadata columns.")

if len(llm_candidate_features) == 0:
    raise ValueError("No numeric LLM-derived candidate features were found.")

print("\nBaseline/original dataset features used:")
for col in baseline_features:
    print(" -", col)

print("\nTotal LLM candidate features found:", len(llm_candidate_features))

# ============================================================
# 6. SELECT BEST 15 LLM FEATURES USING MUTUAL INFORMATION
# ============================================================

mi_df = candidate_df[llm_candidate_features + [TARGET_COL]].copy()

for col in llm_candidate_features:
    median_value = mi_df[col].median()

    if pd.isna(median_value):
        mi_df[col] = mi_df[col].fillna(0)
    else:
        mi_df[col] = mi_df[col].fillna(median_value)

X_mi = mi_df[llm_candidate_features]
y_mi = mi_df[TARGET_COL]

mi_scores = mutual_info_regression(
    X_mi,
    y_mi,
    random_state=RANDOM_STATE
)

mi_table = pd.DataFrame({
    "feature": llm_candidate_features,
    "mutual_information": mi_scores
}).sort_values("mutual_information", ascending=False)

selected_llm_features = mi_table.head(TOP_LLM_FEATURES)["feature"].tolist()

print(f"\nSelected top {TOP_LLM_FEATURES} LLM-derived features:")
for i, col in enumerate(selected_llm_features, 1):
    print(f"{i}. {col}")

selected_features = baseline_features + selected_llm_features

if len(selected_features) == 0:
    raise ValueError("No features selected for SHAP analysis.")

MAX_DISPLAY = len(selected_features)

print("\nTotal selected features for SHAP display:", MAX_DISPLAY)

# ============================================================
# 7. FINAL DATASET PREPARATION
# ============================================================

model_df = candidate_df[selected_features + [TARGET_COL]].copy()

for col in selected_features:
    model_df[col] = make_numeric(model_df[col])

    median_value = model_df[col].median()

    if pd.isna(median_value):
        model_df[col] = model_df[col].fillna(0)
    else:
        model_df[col] = model_df[col].fillna(median_value)

model_df = model_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

X_raw = model_df[selected_features].copy()
y_raw = model_df[TARGET_COL].values.reshape(-1, 1)

# Create readable display names
display_names = []

for col in selected_features:
    if col in baseline_features:
        display_names.append(clean_display_name(col, is_llm=False))
    else:
        display_names.append(clean_display_name(col, is_llm=True))

display_names = ensure_unique_names(display_names)

raw_to_display = dict(zip(selected_features, display_names))
display_to_raw = {v: k for k, v in raw_to_display.items()}

X_display = X_raw.rename(columns=raw_to_display)

print("\nFinal features used for Women BNN + SHAP:")
for col in X_display.columns:
    print(" -", col)

# ============================================================
# 8. TRAIN-TEST SPLIT AND SCALING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_raw,
    y_raw,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# 9. BAYESIAN NEURAL NETWORK MODEL
# ============================================================

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(0, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(-3, 0.1)
        )

        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).normal_(0, 0.1)
        )
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).normal_(-3, 0.1)
        )

    def forward(self, x, sample=True):
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))

            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        kl_weight = 0.5 * torch.sum(
            self.weight_mu ** 2
            + weight_sigma ** 2
            - torch.log(weight_sigma ** 2 + 1e-8)
            - 1
        )

        kl_bias = 0.5 * torch.sum(
            self.bias_mu ** 2
            + bias_sigma ** 2
            - torch.log(bias_sigma ** 2 + 1e-8)
            - 1
        )

        return kl_weight + kl_bias


class BNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.b1 = BayesianLinear(input_dim, 64)
        self.b2 = BayesianLinear(64, 32)
        self.b3 = BayesianLinear(32, 1)

    def forward(self, x, sample=True):
        x = F.relu(self.b1(x, sample=sample))
        x = F.relu(self.b2(x, sample=sample))
        x = self.b3(x, sample=sample)
        return x

    def kl_loss(self):
        return self.b1.kl_loss() + self.b2.kl_loss() + self.b3.kl_loss()


model = BNNRegressor(input_dim=X_train_scaled.shape[1]).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# ============================================================
# 10. TRAIN BNN
# ============================================================

print("\nTraining BNN model...")

model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        preds = model(xb, sample=True)
        mse_loss = criterion(preds, yb)
        kl_loss = model.kl_loss()

        loss = mse_loss + KL_WEIGHT * kl_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"Loss: {epoch_loss / len(train_loader):.6f}"
        )

print("BNN training completed.")

# ============================================================
# 11. MODEL EVALUATION
# ============================================================

@torch.no_grad()
def predict_mean_scaled(X_scaled, sample=False):
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    preds = model(X_tensor, sample=sample)
    return preds.cpu().numpy()


@torch.no_grad()
def predict_original_scale_from_raw(X_input_raw):
    X_scaled = x_scaler.transform(X_input_raw)
    y_pred_scaled = predict_mean_scaled(X_scaled, sample=False)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred.ravel()


y_pred_test = predict_original_scale_from_raw(X_test)

rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred_test))
mae = mean_absolute_error(y_test.ravel(), y_pred_test)
r2 = r2_score(y_test.ravel(), y_pred_test)

print("\nBNN Test Performance")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ============================================================
# 12. SHAP PREDICTION FUNCTION
# ============================================================

X_train_display = X_train.rename(columns=raw_to_display)
X_test_display = X_test.rename(columns=raw_to_display)

def shap_predict(display_data):
    """
    Prediction function for SHAP.
    Accepts SHAP input using display feature names.
    Returns prediction in the original target scale.
    """
    if isinstance(display_data, pd.DataFrame):
        df_input = display_data.copy()
    else:
        df_input = pd.DataFrame(display_data, columns=X_train_display.columns)

    raw_input = df_input.rename(columns=display_to_raw)
    raw_input = raw_input[selected_features]

    return predict_original_scale_from_raw(raw_input)

# ============================================================
# 13. COMPUTE SHAP VALUES
# ============================================================

background_size = min(SHAP_BACKGROUND_SIZE, len(X_train_display))
explain_size = min(SHAP_EXPLAIN_SIZE, len(X_test_display))

background_data = shap.sample(
    X_train_display,
    background_size,
    random_state=RANDOM_STATE
)

explain_data = X_test_display.iloc[:explain_size].copy()

print("\nComputing SHAP values...")
print(f"Background samples: {background_size}")
print(f"Explanation samples: {explain_size}")

masker = shap.maskers.Independent(background_data)

explainer = shap.Explainer(
    shap_predict,
    masker,
    algorithm="permutation"
)

shap_values = explainer(
    explain_data,
    max_evals=2 * explain_data.shape[1] + 1
)

print("SHAP computation completed.")

# ============================================================
# 14. SAVE INDIVIDUAL SHAP PLOTS
# ============================================================

bar_path = os.path.join(OUTPUT_DIR, "women_bnn_shap_bar.png")
beeswarm_path = os.path.join(OUTPUT_DIR, "women_bnn_shap_beeswarm.png")
waterfall_path = os.path.join(OUTPUT_DIR, "women_bnn_shap_waterfall.png")
combined_path = os.path.join(OUTPUT_DIR, "women_bnn_shap_combined.png")

# -------------------------
# Bar Plot
# -------------------------
plt.figure(figsize=(8, 8))
shap.plots.bar(shap_values, max_display=MAX_DISPLAY, show=False)
plt.title("")
plt.tight_layout()
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# Beeswarm Plot
# -------------------------
plt.figure(figsize=(8, 8))
shap.plots.beeswarm(shap_values, max_display=MAX_DISPLAY, show=False)
plt.title("")
plt.tight_layout()
plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# Waterfall Plot
# -------------------------
waterfall_index = min(WATERFALL_INSTANCE_INDEX, explain_size - 1)

plt.figure(figsize=(8, 8))
shap.plots.waterfall(
    shap_values[waterfall_index],
    max_display=MAX_DISPLAY,
    show=False
)
plt.title("")
plt.tight_layout()
plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nIndividual SHAP plots saved:")
print(bar_path)
print(beeswarm_path)
print(waterfall_path)

# ============================================================
# 15. CREATE COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

plot_paths = [bar_path, beeswarm_path, waterfall_path]
subtitles = ["(a) Bar Plot", "(b) Beeswarm Plot", "(c) Waterfall Plot"]

for ax, path, subtitle in zip(axes, plot_paths, subtitles):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    ax.set_title(subtitle, fontsize=14, y=-0.10)

fig.suptitle(
    "Figure: SHAP-based explanation for Women Clothing dataset",
    fontsize=18,
    y=0.02
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nCombined SHAP figure saved:")
print(combined_path)

# ============================================================
# 16. SAVE SELECTED FEATURE LIST
# ============================================================

selected_feature_table = pd.DataFrame({
    "raw_feature_name": selected_features,
    "display_feature_name": display_names,
    "feature_type": [
        "Original dataset feature" if c in baseline_features else "LLM-derived feature"
        for c in selected_features
    ]
})

feature_list_path = os.path.join(OUTPUT_DIR, "selected_features_for_women_bnn_shap.csv")
selected_feature_table.to_csv(feature_list_path, index=False)

print("\nSelected feature list saved:")
print(feature_list_path)

print("\nDone. SHAP bar, beeswarm, waterfall, and combined figure generated successfully.")