# ============================================================
# SHAP Explanation for Stress Dataset - Deep Ensembles Model
# Generates: Bar Plot, Beeswarm Plot, Waterfall Plot, Combined Figure
# Original numeric features + best 15 LLM-derived features
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
from torch.utils.data import TensorDataset, DataLoader

import shap

warnings.filterwarnings("ignore")

# ============================================================
# 1. USER SETTINGS
# ============================================================

DATASET_PATH = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\1 BNN_ShapLime\stress_llm_features_2000.csv"

OUTPUT_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\2 DE_ShapLime\NEW BNN\SHAP_DE_Stress"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "confidence"

DROP_COLUMNS = [
    "Unnamed: 0",
    "source_index",
    "row_index",
    "post_id",
    "sentence_range",
    "text",
    "subreddit",
    "social_timestamp",
    "overall_sentiment",
    "llm_overall_sentiment",
    "llm_confidence"
]

TOP_LLM_FEATURES = 15

TEST_SIZE = 0.20
RANDOM_STATE = 42

NUM_ENSEMBLES = 5
EPOCHS = 600
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

SHAP_BACKGROUND_SIZE = 80
SHAP_EXPLAIN_SIZE = 200
WATERFALL_INSTANCE_INDEX = 0

MAX_DISPLAY = 15

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
    Converts Boolean/string formats to numeric where possible.
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
        "positive": 1,
        "negative": -1,
        "neutral": 0,
        "mixed": 0,
        "stress": 1,
        "no stress": 0,
        "not stress": 0,
        "none": np.nan,
        "nan": np.nan,
        "": np.nan
    }

    mapped = s.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")

    if mapped.notna().sum() > numeric.notna().sum():
        return mapped

    return numeric


def is_llm_feature(col):
    """
    Detects LLM-derived thematic features.
    Works whether columns start with llm_ or use theme_present,
    theme_polarity, and theme_intensity naming.
    """
    col_lower = str(col).lower()

    if col_lower.startswith("llm_"):
        return True

    llm_keywords = [
        "present",
        "polarity",
        "intensity",
        "stress_trigger",
        "emotional",
        "distress",
        "coping",
        "support",
        "anxiety",
        "depression",
        "pressure",
        "workload",
        "academic",
        "family",
        "financial",
        "health",
        "relationship",
        "mental",
        "negative_emotion",
        "urgency",
        "severity"
    ]

    return any(key in col_lower for key in llm_keywords)


def clean_display_name(col, is_llm=False):
    """
    Cleans feature names for SHAP display.
    LLM-derived features are displayed with the prefix 'llm'.
    """
    name = str(col)

    name = name.replace("llm_", "")
    name = name.replace("llm", "")
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    if is_llm:
        return f"llm {name}"

    return name


def ensure_unique_names(names):
    """
    Ensures display feature names remain unique after cleaning.
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

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully")
print("Shape:", df.shape)

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' was not found. "
        f"Available columns are:\n{df.columns.tolist()}"
    )

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# ============================================================
# 5. PREPARE NUMERIC FEATURES
# ============================================================

candidate_df = df.copy()

for col in candidate_df.columns:
    if col != TARGET_COL:
        candidate_df[col] = make_numeric(candidate_df[col])

numeric_feature_cols = [
    c for c in candidate_df.columns
    if c != TARGET_COL
    and pd.api.types.is_numeric_dtype(candidate_df[c])
    and candidate_df[c].notna().sum() > 0
]

if len(numeric_feature_cols) == 0:
    raise ValueError("No numeric features were found after preprocessing.")

# ============================================================
# 6. SEPARATE ORIGINAL FEATURES AND LLM-DERIVED FEATURES
# ============================================================

llm_candidate_features = [
    c for c in numeric_feature_cols
    if is_llm_feature(c)
]

original_features = [
    c for c in numeric_feature_cols
    if c not in llm_candidate_features
]

if len(llm_candidate_features) == 0:
    raise ValueError(
        "No LLM-derived features were detected. "
        "Check whether your LLM columns contain names such as 'llm_', 'present', 'polarity', or 'intensity'."
    )

print("\nOriginal numeric features detected:")
for col in original_features:
    print("-", col)

print("\nTotal LLM candidate features found:", len(llm_candidate_features))

# ============================================================
# 7. SELECT BEST 15 LLM FEATURES USING MUTUAL INFORMATION
# ============================================================

mi_df = candidate_df[llm_candidate_features + [TARGET_COL]].copy()

for col in llm_candidate_features:
    mi_df[col] = mi_df[col].fillna(mi_df[col].median())

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

print("\nSelected best 15 LLM-derived features:")
for i, col in enumerate(selected_llm_features, 1):
    print(f"{i}. {col}")

selected_features = original_features + selected_llm_features

# ============================================================
# 8. FINAL DATASET PREPARATION
# ============================================================

model_df = candidate_df[selected_features + [TARGET_COL]].copy()

for col in selected_features:
    model_df[col] = make_numeric(model_df[col])
    model_df[col] = model_df[col].fillna(model_df[col].median())

model_df = model_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

X_raw = model_df[selected_features].copy()
y_raw = model_df[TARGET_COL].values.reshape(-1, 1)

display_names = []

for col in selected_features:
    if col in selected_llm_features:
        display_names.append(clean_display_name(col, is_llm=True))
    else:
        display_names.append(clean_display_name(col, is_llm=False))

display_names = ensure_unique_names(display_names)

raw_to_display = dict(zip(selected_features, display_names))
display_to_raw = {v: k for k, v in raw_to_display.items()}

X_display = X_raw.rename(columns=raw_to_display)

print("\nFinal features used for Deep Ensembles + SHAP:")
for col in X_display.columns:
    print("-", col)

# ============================================================
# 9. TRAIN-TEST SPLIT AND SCALING
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

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# ============================================================
# 10. DEEP ENSEMBLE MODEL
# ============================================================

class EnsembleRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_single_model(model_index, seed):
    """
    Trains one model in the deep ensemble.
    """
    set_seed(seed)

    model = EnsembleRegressor(
        input_dim=X_train_scaled.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    print(f"\nTraining Deep Ensemble model {model_index + 1}/{NUM_ENSEMBLES}...")

    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            preds = model(xb)
            loss = criterion(preds, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(
                f"Model {model_index + 1} | "
                f"Epoch [{epoch + 1}/{EPOCHS}] | "
                f"Loss: {epoch_loss / len(train_loader):.6f}"
            )

    return model

# ============================================================
# 11. TRAIN DEEP ENSEMBLE MODELS
# ============================================================

ensemble_models = []

for i in range(NUM_ENSEMBLES):
    model_seed = RANDOM_STATE + i
    trained_model = train_single_model(i, model_seed)
    ensemble_models.append(trained_model)

print("\nDeep Ensemble training completed.")

# ============================================================
# 12. DEEP ENSEMBLE PREDICTION FUNCTIONS
# ============================================================

@torch.no_grad()
def ensemble_predict_scaled(X_scaled):
    """
    Predicts using all ensemble models.
    Returns mean and standard deviation in scaled target space.
    """
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    preds = []

    for model in ensemble_models:
        model.eval()
        pred = model(X_tensor)
        preds.append(pred.cpu().numpy())

    preds = np.stack(preds, axis=0)

    mean_scaled = preds.mean(axis=0)
    std_scaled = preds.std(axis=0)

    return mean_scaled, std_scaled


@torch.no_grad()
def predict_original_scale_from_raw(X_input_raw):
    """
    Returns ensemble mean prediction and ensemble predictive standard deviation
    in the original target scale.
    """
    X_scaled = x_scaler.transform(X_input_raw)

    mean_scaled, std_scaled = ensemble_predict_scaled(X_scaled)

    mean_original = y_scaler.inverse_transform(mean_scaled)
    std_original = std_scaled * y_scaler.scale_[0]

    return mean_original.ravel(), std_original.ravel()

# ============================================================
# 13. MODEL EVALUATION
# ============================================================

y_pred_test, y_std_test = predict_original_scale_from_raw(X_test)

rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred_test))
mae = mean_absolute_error(y_test.ravel(), y_pred_test)
r2 = r2_score(y_test.ravel(), y_pred_test)

print("\nDeep Ensembles Test Performance")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")
print(f"Mean ensemble predictive std: {np.mean(y_std_test):.4f}")

# ============================================================
# 14. SHAP PREDICTION FUNCTION
# ============================================================

X_train_display = X_train.rename(columns=raw_to_display)
X_test_display = X_test.rename(columns=raw_to_display)

def shap_predict(display_data):
    """
    SHAP prediction function.
    Uses the Deep Ensemble mean prediction as the model output.
    """
    if isinstance(display_data, pd.DataFrame):
        df_input = display_data.copy()
    else:
        df_input = pd.DataFrame(display_data, columns=X_train_display.columns)

    raw_input = df_input.rename(columns=display_to_raw)
    raw_input = raw_input[selected_features]

    mean_pred, _ = predict_original_scale_from_raw(raw_input)

    return mean_pred

# ============================================================
# 15. COMPUTE SHAP VALUES
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
# 16. SAVE INDIVIDUAL SHAP PLOTS
# ============================================================

bar_path = os.path.join(OUTPUT_DIR, "stress_de_shap_bar.png")
beeswarm_path = os.path.join(OUTPUT_DIR, "stress_de_shap_beeswarm.png")
waterfall_path = os.path.join(OUTPUT_DIR, "stress_de_shap_waterfall.png")
combined_path = os.path.join(OUTPUT_DIR, "stress_de_shap_combined.png")

# Bar Plot
plt.figure(figsize=(8, 8))
shap.plots.bar(
    shap_values,
    max_display=MAX_DISPLAY,
    show=False
)
plt.title("")
plt.tight_layout()
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()

# Beeswarm Plot
plt.figure(figsize=(8, 8))
shap.plots.beeswarm(
    shap_values,
    max_display=MAX_DISPLAY,
    show=False
)
plt.title("")
plt.tight_layout()
plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
plt.close()

# Waterfall Plot
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
# 17. CREATE COMBINED FIGURE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(22, 8))

plot_paths = [bar_path, beeswarm_path, waterfall_path]
subtitles = ["(a) Bar Plot", "(b) Beeswarm Plot", "(c) Waterfall Plot"]

for ax, path, subtitle in zip(axes, plot_paths, subtitles):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    ax.set_title(subtitle, fontsize=14, y=-0.10)

fig.suptitle(
    "Figure 8.3: SHAP-based explanation for Stress dataset using Deep Ensembles",
    fontsize=18,
    y=0.02
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nCombined SHAP figure saved:")
print(combined_path)

# ============================================================
# 18. SAVE SELECTED FEATURE LIST
# ============================================================

selected_feature_table = pd.DataFrame({
    "raw_feature_name": selected_features,
    "display_feature_name": display_names,
    "feature_type": [
        "LLM-derived feature" if c in selected_llm_features else "Original dataset feature"
        for c in selected_features
    ]
})

feature_list_path = os.path.join(
    OUTPUT_DIR,
    "selected_features_for_stress_de_shap.csv"
)

selected_feature_table.to_csv(feature_list_path, index=False)

# ============================================================
# 19. SAVE TEST PREDICTION SUMMARY
# ============================================================

prediction_summary = pd.DataFrame({
    "actual": y_test.ravel(),
    "predicted_mean": y_pred_test,
    "predictive_std": y_std_test,
    "lower_95_pi": y_pred_test - 1.96 * y_std_test,
    "upper_95_pi": y_pred_test + 1.96 * y_std_test
})

prediction_summary_path = os.path.join(
    OUTPUT_DIR,
    "stress_de_test_prediction_summary.csv"
)

prediction_summary.to_csv(prediction_summary_path, index=False)

print("\nSelected feature list saved:")
print(feature_list_path)

print("\nPrediction summary saved:")
print(prediction_summary_path)

print("\nDone. SHAP bar, beeswarm, waterfall, and combined figure generated successfully.")

