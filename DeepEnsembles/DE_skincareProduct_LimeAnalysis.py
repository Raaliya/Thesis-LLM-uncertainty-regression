import os
import re
import html
import webbrowser
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

# =========================================================
# PATHS
# =========================================================
WORK_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\1 BNN_ShapLime"

DATA_PATH = os.path.join(WORK_DIR, "reviews_1000_1500_with_llm_features_2000.csv")

if not os.path.exists(DATA_PATH):
    alt_path = os.path.join(WORK_DIR, "reviews_1000_1500_with_llm_features_2000")
    if os.path.exists(alt_path):
        DATA_PATH = alt_path
    else:
        raise FileNotFoundError(f"Skincare dataset not found:\n{DATA_PATH}")

OUTPUT_HTML = os.path.join(
    WORK_DIR,
    "de_skincare_lime_dashboard_top15_llm_original_features_with_prefix.html"
)

MODEL_SCALER_SEARCH_DIRS = [
    WORK_DIR,
    r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles",
    r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\DE_ShapLime",
    r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2",
]

# =========================================================
# SETTINGS
# =========================================================
TARGET_COL = "rating"
INSTANCE_INDEX = 10
MAX_ROWS_FOR_DASHBOARD = 500

TOP_N_LLM_FEATURES = 15
LIME_NUM_SAMPLES = 2500

# =========================================================
# ORIGINAL SKINCARE FEATURES TO DISPLAY WITHOUT llm_ PREFIX
# =========================================================
ORIGINAL_FEATURES_REQUIRED = [
    "is_recommended",
    "helpfulness",
    "total_feedback_count",
    "total_neg_feedback_count",
    "total_pos_feedback_count",
]

# =========================================================
# COLUMNS TO DROP FROM DISPLAY
# =========================================================
DROP_FROM_DISPLAY = [
    "source_index",
    "Unnamed: 0",
    "author_id",
    "submission_time",
    "review_title",
    "review_text",
    "skin_tone",
    "eye_color",
    "skin_type",
    "hair_color",
    "product_id",
    "product_name",
    "product_title",
    "brand_name",
    "price_usd",
    "overall_sentiment",
    "confidence",
    "parse_error",
    "runtime_error",
]

ALWAYS_EXCLUDE = DROP_FROM_DISPLAY + [TARGET_COL]

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset:", DATA_PATH)
df_original_full = pd.read_csv(DATA_PATH, low_memory=False)
print("Original dataset shape:", df_original_full.shape)

if TARGET_COL not in df_original_full.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def skip_directory(dirname):
    skip_names = {
        "venv",
        "venv311",
        ".venv",
        "__pycache__",
        ".git",
        "Lib",
        "Scripts",
        "site-packages",
        "node_modules",
    }
    return dirname in skip_names


def safe_torch_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def clean_state_dict(state):
    cleaned = {}

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue

        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value

    return cleaned


def is_valid_state_dict(state):
    if not isinstance(state, dict):
        return False

    state = clean_state_dict(state)

    return (
        "net.0.weight" in state
        and any(k.startswith("net.") and k.endswith(".weight") for k in state.keys())
    )


def extract_state_dicts_from_object(obj):
    state_dicts = []

    if isinstance(obj, dict):
        possible_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "models",
            "state_dicts",
            "model_state_dicts",
            "ensemble",
            "ensemble_state_dicts",
            "model_states",
        ]

        if is_valid_state_dict(obj):
            state_dicts.append(clean_state_dict(obj))

        for key in possible_keys:
            if key in obj:
                value = obj[key]

                if isinstance(value, dict) and is_valid_state_dict(value):
                    state_dicts.append(clean_state_dict(value))

                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, dict) and is_valid_state_dict(item):
                            state_dicts.append(clean_state_dict(item))

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict) and is_valid_state_dict(item):
                state_dicts.append(clean_state_dict(item))

    return state_dicts


def get_input_dim_from_state(state):
    if "net.0.weight" not in state:
        return None
    return int(state["net.0.weight"].shape[1])


def get_linear_indices_from_state(state):
    linear_indices = []

    for key, value in state.items():
        match = re.match(r"net\.(\d+)\.weight$", key)

        if match and torch.is_tensor(value) and len(value.shape) == 2:
            linear_indices.append(int(match.group(1)))

    return sorted(linear_indices)


def build_model_from_state(state):
    linear_indices = get_linear_indices_from_state(state)

    if len(linear_indices) < 2:
        raise ValueError("Could not detect enough Linear layers in model state_dict.")

    layers = []
    current_index = 0

    for i, layer_index in enumerate(linear_indices):
        weight_key = f"net.{layer_index}.weight"
        out_features = state[weight_key].shape[0]
        in_features = state[weight_key].shape[1]

        while current_index < layer_index:
            if current_index in [1, 4]:
                layers.append(nn.ReLU())
            elif current_index in [2, 5]:
                layers.append(nn.Dropout(0.10))
            else:
                layers.append(nn.ReLU())

            current_index += 1

        layers.append(nn.Linear(in_features, out_features))
        current_index += 1

        if i < len(linear_indices) - 1:
            next_index = linear_indices[i + 1]

            while current_index < next_index:
                if current_index in [1, 4]:
                    layers.append(nn.ReLU())
                elif current_index in [2, 5]:
                    layers.append(nn.Dropout(0.10))
                else:
                    layers.append(nn.ReLU())

                current_index += 1

    class DERegressor(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = DERegressor(layers).to(device)
    model.load_state_dict(state)
    model.eval()

    return model


def is_llm_feature(feature_name: str) -> bool:
    """
    Original Skincare features are not LLM features.
    All remaining thematic present/polarity/intensity features are treated as LLM-derived.
    """
    if feature_name in ORIGINAL_FEATURES_REQUIRED:
        return False

    if feature_name in DROP_FROM_DISPLAY:
        return False

    if feature_name == TARGET_COL:
        return False

    if feature_name.startswith("llm_"):
        return True

    if (
        feature_name.endswith("_present")
        or feature_name.endswith("_polarity")
        or feature_name.endswith("_intensity")
    ):
        return True

    return False


def force_llm_prefix_for_display(feature_name: str) -> str:
    """
    Adds llm_ prefix only for dashboard display.
    The actual dataset/model/scaler column names are not changed.
    """
    if feature_name in ORIGINAL_FEATURES_REQUIRED:
        return feature_name

    if feature_name.startswith("llm_"):
        return feature_name

    if is_llm_feature(feature_name):
        return "llm_" + feature_name

    return feature_name


def dataset_has_feature(df_columns, feature_name: str) -> bool:
    """
    Allows the script to work even if the CSV was manually edited and
    llm_ was added to some LLM-derived columns.
    """
    if feature_name in df_columns:
        return True

    prefixed = "llm_" + feature_name
    if feature_name not in ORIGINAL_FEATURES_REQUIRED and prefixed in df_columns:
        return True

    if feature_name.startswith("llm_") and feature_name[4:] in df_columns:
        return True

    return False


def get_dataset_series(df, feature_name: str):
    """
    Returns the correct dataset column while preserving the model/scaler
    expected feature name internally.
    """
    if feature_name in df.columns:
        return df[feature_name]

    prefixed = "llm_" + feature_name
    if feature_name not in ORIGINAL_FEATURES_REQUIRED and prefixed in df.columns:
        return df[prefixed]

    if feature_name.startswith("llm_") and feature_name[4:] in df.columns:
        return df[feature_name[4:]]

    raise KeyError(f"Feature not found in dataset: {feature_name}")


def extract_base_feature_from_rule(rule: str, actual_features: list[str], display_labels: list[str]):
    """
    Maps a LIME rule string back to the actual model feature name.
    """
    pairs = list(zip(actual_features, display_labels))
    pairs = sorted(pairs, key=lambda x: len(x[1]), reverse=True)

    rule_lower = rule.lower()

    for actual, label in pairs:
        if label.lower() in rule_lower:
            return actual, label

    return rule, rule


def format_value(value):
    if isinstance(value, (np.floating, float)):
        return f"{value:.4f}"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    return str(value)


def build_probability_bar(label: str, prob: float, color: str) -> str:
    return f"""
    <div class="prob-row">
        <div class="prob-label">{html.escape(label)}</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{prob * 100:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{prob:.2f}</div>
    </div>
    """


def apply_llm_prefix_to_lime_html(lime_html: str, actual_features: list[str]) -> str:
    """
    Forces llm_ prefix inside the LIME iframe HTML if LIME shortens or displays
    the original feature names without the prefix.
    """
    # Longest names first to avoid partial replacement conflicts
    for feature in sorted(actual_features, key=len, reverse=True):
        display_feature = force_llm_prefix_for_display(feature)

        if display_feature != feature:
            # Replace only unprefixed feature text. Avoid double-prefixing already prefixed names.
            lime_html = lime_html.replace(feature, display_feature)

    # Safety cleanup in case any double prefixing occurs
    lime_html = lime_html.replace("llm_llm_", "llm_")

    return lime_html

# =========================================================
# AUTO-FIND COMPATIBLE DE SKINCARE MODEL + SCALER
# =========================================================
available_columns = set(df_original_full.columns)

scaler_candidates = []
model_candidates = []

for search_dir in MODEL_SCALER_SEARCH_DIRS:
    if not os.path.exists(search_dir):
        continue

    for root, dirs, files in os.walk(search_dir):
        dirs[:] = [d for d in dirs if not skip_directory(d)]

        for file in files:
            file_path = os.path.join(root, file)

            if file.lower().endswith(".joblib"):
                try:
                    scaler_obj = joblib.load(file_path)

                    if not hasattr(scaler_obj, "feature_names_in_"):
                        continue

                    scaler_features = list(scaler_obj.feature_names_in_)

                    missing_from_dataset = [
                        c for c in scaler_features
                        if not dataset_has_feature(available_columns, c)
                    ]

                    missing_original_features = [
                        c for c in ORIGINAL_FEATURES_REQUIRED
                        if c not in scaler_features
                    ]

                    llm_features_available = [
                        c for c in scaler_features
                        if is_llm_feature(c)
                    ]

                    if missing_from_dataset:
                        continue

                    if missing_original_features:
                        continue

                    if len(llm_features_available) == 0:
                        continue

                    scaler_candidates.append({
                        "path": file_path,
                        "features": scaler_features,
                        "feature_count": len(scaler_features),
                    })

                except Exception:
                    continue

            elif file.lower().endswith((".pt", ".pth")):
                try:
                    obj = safe_torch_load(file_path)
                    states = extract_state_dicts_from_object(obj)

                    if not states:
                        continue

                    input_dim = get_input_dim_from_state(states[0])

                    if input_dim is None:
                        continue

                    model_candidates.append({
                        "path": file_path,
                        "input_dim": input_dim,
                        "states": states,
                        "n_models": len(states),
                    })

                except Exception:
                    continue

if not scaler_candidates:
    raise FileNotFoundError(
        "No compatible Skincare scaler was found. Restore the original CSV column names "
        "or ensure the scaler includes the original Skincare features and Skincare LLM features."
    )

if not model_candidates:
    raise FileNotFoundError(
        "No compatible Deep Ensembles model file was found. The model file must "
        "contain one or more state_dicts with 'net.0.weight'."
    )

possible_pairs = []

for scaler_info in scaler_candidates:
    for model_info in model_candidates:
        if scaler_info["feature_count"] == model_info["input_dim"]:
            combined_path = (
                scaler_info["path"].lower() + " " + model_info["path"].lower()
            )

            same_folder = (
                os.path.dirname(scaler_info["path"]).lower()
                == os.path.dirname(model_info["path"]).lower()
            )

            score = scaler_info["feature_count"]

            if "skincare" in combined_path:
                score += 160

            if "deep" in combined_path:
                score += 150

            if "ensemble" in combined_path:
                score += 150

            if "de_" in combined_path or "\\de" in combined_path:
                score += 80

            if "original" in combined_path:
                score += 60

            if "llm" in combined_path:
                score += 60

            if "lime" in combined_path:
                score += 30

            if same_folder:
                score += 80

            if model_info["n_models"] > 1:
                score += 100

            if "bnn" in combined_path:
                score -= 80

            if "mcd" in combined_path:
                score -= 100

            possible_pairs.append({
                "score": score,
                "scaler": scaler_info,
                "model": model_info,
            })

if not possible_pairs:
    raise ValueError(
        "Model and scaler files were found, but no pair has matching input dimensions."
    )

best_pair = sorted(possible_pairs, key=lambda x: x["score"], reverse=True)[0]

SCALER_PATH = best_pair["scaler"]["path"]
MODEL_PATH = best_pair["model"]["path"]
state_dicts = best_pair["model"]["states"]
model_feature_names = best_pair["scaler"]["features"]

print("\nSelected DE Skincare scaler:")
print(SCALER_PATH)

print("\nSelected DE Skincare model:")
print(MODEL_PATH)

print("\nNumber of ensemble members detected:", len(state_dicts))
print("Model/scaler feature count:", len(model_feature_names))

# =========================================================
# LIMIT DATA FOR DASHBOARD BACKGROUND
# =========================================================
df_original = df_original_full.copy()

if MAX_ROWS_FOR_DASHBOARD is not None:
    df_original = df_original.head(MAX_ROWS_FOR_DASHBOARD).copy()
    print(f"\nUsing first {MAX_ROWS_FOR_DASHBOARD} rows for dashboard background.")

# =========================================================
# PREPARE TARGET
# =========================================================
df = df_original.copy()

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()

y = df[TARGET_COL].copy()

if INSTANCE_INDEX >= len(df):
    raise ValueError(
        f"INSTANCE_INDEX={INSTANCE_INDEX} is out of range. "
        f"Dataset contains only {len(df)} rows after cleaning."
    )

# =========================================================
# CHECK REQUIRED FEATURES
# =========================================================
missing_original_from_dataset = [
    c for c in ORIGINAL_FEATURES_REQUIRED
    if c not in df.columns
]

if missing_original_from_dataset:
    raise ValueError(
        "The following original Skincare features are missing from the dataset:\n"
        f"{missing_original_from_dataset}"
    )

missing_original_from_model = [
    c for c in ORIGINAL_FEATURES_REQUIRED
    if c not in model_feature_names
]

if missing_original_from_model:
    raise ValueError(
        "The following original Skincare features exist in the dataset but are NOT "
        "included in the selected DE model/scaler:\n"
        f"{missing_original_from_model}"
    )

missing_model_features = [
    c for c in model_feature_names
    if not dataset_has_feature(set(df.columns), c)
]

if missing_model_features:
    raise ValueError(
        "The following features required by the DE scaler/model are missing "
        f"from the dataset:\n{missing_model_features}"
    )

# =========================================================
# PREPARE MODEL INPUT FEATURES
# =========================================================
X_model_raw = pd.DataFrame(index=df.index)

for col in model_feature_names:
    X_model_raw[col] = get_dataset_series(df, col)

for col in X_model_raw.columns:
    X_model_raw[col] = pd.to_numeric(X_model_raw[col], errors="coerce")

for col in X_model_raw.columns:
    if X_model_raw[col].isna().sum() > 0:
        median_value = X_model_raw[col].median()

        if pd.isna(median_value):
            X_model_raw[col] = X_model_raw[col].fillna(0)
        else:
            X_model_raw[col] = X_model_raw[col].fillna(median_value)

scaler = joblib.load(SCALER_PATH)
X_model_scaled = scaler.transform(X_model_raw)

# =========================================================
# LOAD DEEP ENSEMBLE MEMBERS
# =========================================================
ensemble_models = []

for i, state in enumerate(state_dicts):
    input_dim = get_input_dim_from_state(state)

    if input_dim != len(model_feature_names):
        raise ValueError(
            f"Ensemble member {i} input dimension is {input_dim}, but scaler "
            f"expects {len(model_feature_names)} features."
        )

    ensemble_models.append(build_model_from_state(state))

if len(ensemble_models) == 0:
    raise ValueError("No valid ensemble models were loaded.")

print("\nDeep Ensemble Skincare model loaded successfully.")
print("Loaded ensemble members:", len(ensemble_models))

# =========================================================
# DE PREDICTION FUNCTIONS
# =========================================================
def ensemble_predict_matrix(x_scaled: np.ndarray) -> np.ndarray:
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    preds = []

    for member in ensemble_models:
        member.eval()
        with torch.no_grad():
            pred = member(x_tensor).cpu().numpy().flatten()
            preds.append(pred)

    return np.array(preds)


def ensemble_predict_mean(x_scaled: np.ndarray) -> np.ndarray:
    preds = ensemble_predict_matrix(x_scaled)
    return preds.mean(axis=0)


def ensemble_predict_single_distribution(full_scaled_row: np.ndarray) -> np.ndarray:
    x_scaled = full_scaled_row.reshape(1, -1)
    preds = ensemble_predict_matrix(x_scaled).flatten()
    return np.array(preds, dtype=float)

# =========================================================
# STEP 1: SELECT BEST 15 LLM FEATURES USING FULL LOCAL LIME
# =========================================================
all_display_features = [
    c for c in model_feature_names
    if c not in ALWAYS_EXCLUDE
]

llm_candidate_features = [
    c for c in all_display_features
    if is_llm_feature(c)
]

if len(llm_candidate_features) == 0:
    raise ValueError(
        "No LLM-derived Skincare features were detected in the selected DE model/scaler."
    )

full_selection_features = ORIGINAL_FEATURES_REQUIRED + llm_candidate_features

full_selection_features = [
    c for c in full_selection_features
    if c in model_feature_names
]

full_selection_labels = [
    force_llm_prefix_for_display(c) for c in full_selection_features
]

full_selection_positions = [
    model_feature_names.index(c)
    for c in full_selection_features
]

base_instance_full_raw = X_model_raw.iloc[INSTANCE_INDEX].values.astype(float)


def predict_fn_full_selection(display_raw_array: np.ndarray) -> np.ndarray:
    display_raw_array = np.asarray(display_raw_array, dtype=float)

    if display_raw_array.ndim == 1:
        display_raw_array = display_raw_array.reshape(1, -1)

    full_raw = np.tile(base_instance_full_raw, (display_raw_array.shape[0], 1))

    for i, pos in enumerate(full_selection_positions):
        full_raw[:, pos] = display_raw_array[:, i]

    full_raw_df = pd.DataFrame(full_raw, columns=model_feature_names)
    full_scaled = scaler.transform(full_raw_df)

    return ensemble_predict_mean(full_scaled)


X_lime_full_raw = X_model_raw[full_selection_features].copy()

full_explainer = LimeTabularExplainer(
    training_data=X_lime_full_raw.values,
    feature_names=full_selection_labels,
    mode="regression",
    discretize_continuous=True,
    random_state=42
)

print("\nGenerating full LIME explanation to select top 15 LLM features...")

full_exp = full_explainer.explain_instance(
    data_row=X_lime_full_raw.iloc[INSTANCE_INDEX].values,
    predict_fn=predict_fn_full_selection,
    num_features=len(full_selection_features),
    num_samples=LIME_NUM_SAMPLES
)

full_lime_rules = full_exp.as_list()

llm_contributions = {}

for rule, contribution in full_lime_rules:
    base_feature, shown_label = extract_base_feature_from_rule(
        rule,
        full_selection_features,
        full_selection_labels
    )

    if base_feature in llm_candidate_features:
        llm_contributions[base_feature] = abs(float(contribution))

top_llm_features = sorted(
    llm_contributions,
    key=llm_contributions.get,
    reverse=True
)[:TOP_N_LLM_FEATURES]

if len(top_llm_features) < TOP_N_LLM_FEATURES:
    remaining = [
        c for c in llm_candidate_features
        if c not in top_llm_features
    ]

    top_llm_features.extend(
        remaining[:TOP_N_LLM_FEATURES - len(top_llm_features)]
    )

compact_display_features = []

for c in ORIGINAL_FEATURES_REQUIRED + top_llm_features:
    if c in model_feature_names and c not in compact_display_features:
        compact_display_features.append(c)

compact_display_labels = [
    force_llm_prefix_for_display(c) for c in compact_display_features
]

NUM_FEATURES_FINAL = len(compact_display_features)

print("\nOriginal Skincare features included:")
for c in ORIGINAL_FEATURES_REQUIRED:
    print(" -", c)

print(f"\nTop {TOP_N_LLM_FEATURES} LLM features selected:")
for c in top_llm_features:
    print(" -", force_llm_prefix_for_display(c))

print("\nFinal compact display feature count:", NUM_FEATURES_FINAL)

# =========================================================
# STEP 2: FINAL COMPACT LIME EXPLANATION
# =========================================================
compact_display_positions = [
    model_feature_names.index(c)
    for c in compact_display_features
]


def predict_fn_compact(display_raw_array: np.ndarray) -> np.ndarray:
    display_raw_array = np.asarray(display_raw_array, dtype=float)

    if display_raw_array.ndim == 1:
        display_raw_array = display_raw_array.reshape(1, -1)

    full_raw = np.tile(base_instance_full_raw, (display_raw_array.shape[0], 1))

    for i, pos in enumerate(compact_display_positions):
        full_raw[:, pos] = display_raw_array[:, i]

    full_raw_df = pd.DataFrame(full_raw, columns=model_feature_names)
    full_scaled = scaler.transform(full_raw_df)

    return ensemble_predict_mean(full_scaled)


X_lime_compact_raw = X_model_raw[compact_display_features].copy()
instance_raw_display = X_lime_compact_raw.iloc[INSTANCE_INDEX].copy()

compact_explainer = LimeTabularExplainer(
    training_data=X_lime_compact_raw.values,
    feature_names=compact_display_labels,
    mode="regression",
    discretize_continuous=True,
    random_state=42
)

print("\nGenerating compact DE Skincare LIME explanation...")

exp = compact_explainer.explain_instance(
    data_row=X_lime_compact_raw.iloc[INSTANCE_INDEX].values,
    predict_fn=predict_fn_compact,
    num_features=NUM_FEATURES_FINAL,
    num_samples=LIME_NUM_SAMPLES
)

# =========================================================
# PREDICTION SUMMARY
# =========================================================
ensemble_preds = ensemble_predict_single_distribution(
    X_model_scaled[INSTANCE_INDEX]
)

pred_mean = float(np.mean(ensemble_preds))
pred_std = float(np.std(ensemble_preds, ddof=1)) if len(ensemble_preds) > 1 else 0.0
pred_low = float(np.percentile(ensemble_preds, 2.5))
pred_high = float(np.percentile(ensemble_preds, 97.5))
actual_value = float(y.iloc[INSTANCE_INDEX])

p_lt_3 = float(np.mean(ensemble_preds < 3.0))
p_3_to_4 = float(np.mean((ensemble_preds >= 3.0) & (ensemble_preds < 4.0)))
p_ge_4 = float(np.mean(ensemble_preds >= 4.0))

# =========================================================
# FEATURE-VALUE TABLE
# =========================================================
lime_rules = exp.as_list()

table_rows = []
used_features = set()

for rule, contribution in lime_rules:
    base_feature, shown_label = extract_base_feature_from_rule(
        rule,
        compact_display_features,
        compact_display_labels
    )

    if base_feature in instance_raw_display.index and base_feature not in used_features:
        raw_value = instance_raw_display[base_feature]

        table_rows.append({
            "feature": force_llm_prefix_for_display(base_feature),
            "value": format_value(raw_value),
            "contribution": float(contribution)
        })

        used_features.add(base_feature)

if not table_rows:
    for feat in compact_display_features:
        table_rows.append({
            "feature": force_llm_prefix_for_display(feat),
            "value": format_value(instance_raw_display[feat]),
            "contribution": 0.0
        })

# =========================================================
# HTML PANELS
# =========================================================
summary_html = f"""
<div class="card">
    <h2>Prediction Summary</h2>

    <div class="summary-grid">
        <div><span class="k">Predicted value</span><span class="v">{pred_mean:.3f}</span></div>
        <div><span class="k">Actual value</span><span class="v">{actual_value:.3f}</span></div>
        <div><span class="k">Predictive std</span><span class="v">{pred_std:.3f}</span></div>
        <div><span class="k">95% interval</span><span class="v">[{pred_low:.3f}, {pred_high:.3f}]</span></div>
    </div>

    <h3>Prediction distribution by rating band</h3>

    {build_probability_bar("Rating < 3", p_lt_3, "#1f77b4")}
    {build_probability_bar("3 ≤ Rating < 4", p_3_to_4, "#ffbf00")}
    {build_probability_bar("Rating ≥ 4", p_ge_4, "#ff7f0e")}
</div>
"""

lime_html_raw = exp.as_html()

# Force llm_ prefix inside the LIME iframe display only
lime_html_raw = apply_llm_prefix_to_lime_html(
    lime_html_raw,
    compact_display_features
)

lime_panel = f"""
<div class="card">
    <h2>LIME Explanation</h2>
    <iframe class="lime-frame" srcdoc="{html.escape(lime_html_raw)}"></iframe>
</div>
"""

table_html_rows = ""

for row in table_rows:
    sign_class = "pos" if row["contribution"] >= 0 else "neg"

    table_html_rows += f"""
    <tr>
        <td>{html.escape(row["feature"])}</td>
        <td>{html.escape(row["value"])}</td>
        <td class="{sign_class}">{row["contribution"]:+.3f}</td>
    </tr>
    """

feature_table_html = f"""
<div class="card">
    <h2>Feature-Value Table</h2>

    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>LIME<br>Contribution</th>
            </tr>
        </thead>
        <tbody>
            {table_html_rows}
        </tbody>
    </table>

    <p class="note">
        Values shown are raw feature values from the selected test instance.
    </p>
</div>
"""

# =========================================================
# FULL HTML DASHBOARD
# =========================================================
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DE Skincare LIME Dashboard</title>

<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 0;
        padding: 8px;
        background: #f7f7f7;
        color: #222;
    }}

    h1 {{
        margin: 0 0 6px 0;
        font-size: 28px;
        font-weight: 800;
    }}

    .subtitle {{
        margin-bottom: 18px;
        color: #555;
        font-size: 14px;
    }}

    .layout {{
        display: grid;
        grid-template-columns: 1.05fr 1.45fr 1.05fr;
        gap: 14px;
        align-items: start;
    }}

    .card {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        padding: 18px;
    }}

    .card h2 {{
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 20px;
        font-weight: 800;
    }}

    .card h3 {{
        margin-top: 18px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: 800;
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
    }}

    .summary-grid div {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        padding: 10px 12px;
        background: #fafafa;
        border-radius: 8px;
    }}

    .k {{
        color: #555;
        font-weight: 600;
    }}

    .v {{
        color: #111;
        font-weight: 700;
    }}

    .prob-row {{
        display: grid;
        grid-template-columns: 115px 1fr 38px;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
    }}

    .prob-label {{
        font-size: 14px;
    }}

    .prob-bar-wrap {{
        height: 18px;
        background: #e9e9e9;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #ddd;
    }}

    .prob-bar {{
        height: 100%;
        border-radius: 10px;
    }}

    .prob-value {{
        text-align: right;
        font-weight: 700;
        font-size: 14px;
    }}

    .lime-frame {{
        width: 100%;
        height: 720px;
        border: none;
        border-radius: 8px;
        background: white;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}

    th, td {{
        border-bottom: 1px solid #e6e6e6;
        padding: 8px 7px;
        text-align: left;
        vertical-align: top;
    }}

    th {{
        background: #fafafa;
        font-weight: 800;
    }}

    td:nth-child(1) {{
        word-break: break-word;
    }}

    td:nth-child(2) {{
        text-align: right;
        white-space: nowrap;
    }}

    td:nth-child(3) {{
        text-align: left;
        white-space: nowrap;
    }}

    .pos {{
        color: #ff7f0e;
        font-weight: 800;
    }}

    .neg {{
        color: #1f77b4;
        font-weight: 800;
    }}

    .note {{
        margin-top: 14px;
        color: #666;
        font-size: 12px;
    }}

    @media screen and (max-width: 1200px) {{
        .layout {{
            grid-template-columns: 1fr;
        }}

        .lime-frame {{
            height: 720px;
        }}
    }}
</style>
</head>

<body>
    <h1>DE Skincare – Single-Instance Explanation Dashboard</h1>

    <div class="subtitle">
        Instance index: {INSTANCE_INDEX} &nbsp;|&nbsp;
        Predicted mean: {pred_mean:.3f} &nbsp;|&nbsp;
        Actual: {actual_value:.3f} &nbsp;|&nbsp;
        Displayed features: {NUM_FEATURES_FINAL}
    </div>

    <div class="layout">
        {summary_html}
        {lime_panel}
        {feature_table_html}
    </div>
</body>
</html>
"""

# =========================================================
# SAVE HTML
# =========================================================
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(full_html)

print("\nDE Skincare compact LIME dashboard with llm_ display prefix saved to:")
print(OUTPUT_HTML)

webbrowser.open("file:///" + OUTPUT_HTML.replace("\\", "/"))
