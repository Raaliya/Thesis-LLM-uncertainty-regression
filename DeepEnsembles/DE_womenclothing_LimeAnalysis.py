# ============================================================
# LIME Explanation Dashboard for Women Clothing Dataset - Deep Ensembles
# Baseline features + best 15 LLM-derived features
# ============================================================

import os
import html
import random
import warnings
import webbrowser

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

WORK_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\1 BNN_ShapLime"

DATA_PATH = os.path.join(
    WORK_DIR,
    "women_clothing_reviews_llm_2000_seed42.csv"
)

OUTPUT_HTML = os.path.join(
    WORK_DIR,
    "de_women_lime_dashboard_top15_llm_baseline_features.html"
)

# ============================================================
# 2. SETTINGS
# ============================================================

TARGET_COL = "Rating"
INSTANCE_INDEX = 10

TOP_N_LLM_FEATURES = 15

TEST_SIZE = 0.20
RANDOM_STATE = 42

NUM_ENSEMBLES = 5
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

LIME_NUM_SAMPLES = 2500

# ============================================================
# 3. COLUMNS TO DROP
# ============================================================

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
    "Recommended IND",
]

# ============================================================
# 4. DEVICE AND REPRODUCIBILITY
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def make_numeric(series):
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


def display_name(col):
    """
    Adds llm_ prefix only for LLM-derived features.
    Baseline features remain unchanged.
    """
    col = str(col)

    if is_llm_feature(col):
        if col.startswith("llm_"):
            return col
        return "llm_" + col

    return col


def ensure_unique_names(names):
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


def extract_actual_feature_from_rule(rule, actual_features, display_features):
    pairs = list(zip(actual_features, display_features))
    pairs = sorted(pairs, key=lambda x: len(x[1]), reverse=True)

    rule_lower = rule.lower()

    for actual, shown in pairs:
        if shown.lower() in rule_lower:
            return actual

    actual_sorted = sorted(actual_features, key=len, reverse=True)

    for actual in actual_sorted:
        if actual.lower() in rule_lower:
            return actual

    return rule


def format_value(value):
    if isinstance(value, (np.floating, float)):
        return f"{value:.4f}"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    return str(value)


def build_probability_bar(label, prob, color):
    return f"""
    <div class="prob-row">
        <div class="prob-label">{html.escape(label)}</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{prob * 100:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{prob:.2f}</div>
    </div>
    """


# ============================================================
# 6. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    alt_path = DATA_PATH.replace(".csv", "")
    if os.path.exists(alt_path):
        DATA_PATH = alt_path
    else:
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

print("Loading dataset:", DATA_PATH)
df_original = pd.read_csv(DATA_PATH, low_memory=False)
df_original.columns = df_original.columns.astype(str).str.strip()

print("Original dataset shape:", df_original.shape)

if TARGET_COL not in df_original.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found. Available columns:\n"
        f"{df_original.columns.tolist()}"
    )

df_original[TARGET_COL] = pd.to_numeric(df_original[TARGET_COL], errors="coerce")

feature_drop_columns = [
    c for c in DROP_COLUMNS
    if c in df_original.columns and c != TARGET_COL
]

df = df_original.drop(columns=feature_drop_columns, errors="ignore").copy()
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# ============================================================
# 7. IDENTIFY BASELINE FEATURES AND LLM FEATURES
# ============================================================

for col in df.columns:
    if col != TARGET_COL:
        df[col] = make_numeric(df[col])

numeric_features = [
    c for c in df.columns
    if c != TARGET_COL
    and pd.api.types.is_numeric_dtype(df[c])
    and df[c].notna().sum() > 0
]

baseline_features = [
    c for c in numeric_features
    if not is_llm_feature(c)
]

llm_candidate_features = [
    c for c in numeric_features
    if is_llm_feature(c)
]

if len(baseline_features) == 0:
    print("\nWarning: No baseline numeric features were found.")

if len(llm_candidate_features) == 0:
    raise ValueError("No LLM-derived numeric features were found.")

print("\nBaseline/original features included:")
for c in baseline_features:
    print(" -", c)

print("\nTotal LLM candidate features:", len(llm_candidate_features))

# ============================================================
# 8. SELECT BEST 15 LLM FEATURES
# ============================================================

mi_df = df[llm_candidate_features + [TARGET_COL]].copy()

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

selected_llm_features = mi_table.head(TOP_N_LLM_FEATURES)["feature"].tolist()

print(f"\nSelected top {TOP_N_LLM_FEATURES} LLM-derived features:")
for i, c in enumerate(selected_llm_features, 1):
    print(f"{i}. {c}")

selected_features = baseline_features + selected_llm_features

if len(selected_features) == 0:
    raise ValueError("No features selected for modelling.")

display_features = ensure_unique_names([
    display_name(c) for c in selected_features
])

actual_to_display = dict(zip(selected_features, display_features))
display_to_actual = dict(zip(display_features, selected_features))

print("\nFinal display features:")
for c in display_features:
    print(" -", c)

# ============================================================
# 9. FINAL MODEL DATASET
# ============================================================

model_df = df[selected_features + [TARGET_COL]].copy()

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

if INSTANCE_INDEX >= len(X_raw):
    raise ValueError(
        f"INSTANCE_INDEX={INSTANCE_INDEX} is out of range. "
        f"Dataset has only {len(X_raw)} rows after cleaning."
    )

# ============================================================
# 10. TRAIN-TEST SPLIT AND SCALING
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
# 11. DEEP ENSEMBLE MODEL
# ============================================================

class EnsembleRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_single_model(model_index, seed):
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

    model.eval()
    return model


# ============================================================
# 12. TRAIN DEEP ENSEMBLES
# ============================================================

ensemble_models = []

for i in range(NUM_ENSEMBLES):
    model_seed = RANDOM_STATE + i
    trained_model = train_single_model(i, model_seed)
    ensemble_models.append(trained_model)

print("\nDeep Ensemble training completed.")

# ============================================================
# 13. PREDICTION FUNCTIONS
# ============================================================

@torch.no_grad()
def ensemble_predict_scaled(X_scaled):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    preds = []

    for model in ensemble_models:
        model.eval()
        pred = model(X_tensor)
        preds.append(pred.cpu().numpy())

    preds = np.stack(preds, axis=0)

    mean_scaled = preds.mean(axis=0)
    std_scaled = preds.std(axis=0)

    return mean_scaled, std_scaled, preds


@torch.no_grad()
def predict_original_scale_from_raw(X_input_raw):
    X_scaled = x_scaler.transform(X_input_raw)

    mean_scaled, std_scaled, _ = ensemble_predict_scaled(X_scaled)

    mean_original = y_scaler.inverse_transform(mean_scaled)
    std_original = std_scaled * y_scaler.scale_[0]

    return mean_original.ravel(), std_original.ravel()


@torch.no_grad()
def ensemble_predict_single_distribution(raw_row):
    raw_df = pd.DataFrame(
        raw_row.reshape(1, -1),
        columns=selected_features
    )

    X_scaled = x_scaler.transform(raw_df)
    _, _, preds_scaled = ensemble_predict_scaled(X_scaled)

    preds_original = []

    for i in range(preds_scaled.shape[0]):
        pred_original = y_scaler.inverse_transform(preds_scaled[i])
        preds_original.append(pred_original.flatten()[0])

    return np.array(preds_original, dtype=float)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

y_pred_test, y_std_test = predict_original_scale_from_raw(X_test)

rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred_test))
mae = mean_absolute_error(y_test.ravel(), y_pred_test)
r2 = r2_score(y_test.ravel(), y_pred_test)

print("\nDeep Ensembles Test Performance")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")
print(f"Mean predictive std: {np.mean(y_std_test):.4f}")

# ============================================================
# 15. LIME EXPLAINER
# ============================================================

X_lime_train = X_train.copy()
X_lime_train_display = X_lime_train.rename(columns=actual_to_display)

instance_raw = X_raw.iloc[INSTANCE_INDEX].copy()
instance_display = instance_raw.rename(index=actual_to_display)

actual_features = selected_features
shown_features = display_features


def lime_predict(display_array):
    display_array = np.asarray(display_array, dtype=float)

    if display_array.ndim == 1:
        display_array = display_array.reshape(1, -1)

    display_df = pd.DataFrame(display_array, columns=shown_features)

    actual_df = display_df.rename(columns=display_to_actual)
    actual_df = actual_df[actual_features]

    pred_mean, _ = predict_original_scale_from_raw(actual_df)

    return pred_mean


lime_explainer = LimeTabularExplainer(
    training_data=X_lime_train_display.values,
    feature_names=shown_features,
    mode="regression",
    discretize_continuous=True,
    random_state=RANDOM_STATE
)

print("\nGenerating LIME explanation...")

exp = lime_explainer.explain_instance(
    data_row=instance_display.values,
    predict_fn=lime_predict,
    num_features=len(shown_features),
    num_samples=LIME_NUM_SAMPLES
)

# ============================================================
# 16. PREDICTION SUMMARY
# ============================================================

ensemble_preds = ensemble_predict_single_distribution(
    X_raw.iloc[INSTANCE_INDEX].values.astype(float)
)

pred_mean = float(np.mean(ensemble_preds))
pred_std = float(np.std(ensemble_preds, ddof=1)) if len(ensemble_preds) > 1 else 0.0
pred_low = float(np.percentile(ensemble_preds, 2.5))
pred_high = float(np.percentile(ensemble_preds, 97.5))
actual_value = float(y_raw[INSTANCE_INDEX][0])

p_lt_3 = float(np.mean(ensemble_preds < 3.0))
p_3_to_4 = float(np.mean((ensemble_preds >= 3.0) & (ensemble_preds < 4.0)))
p_ge_4 = float(np.mean(ensemble_preds >= 4.0))

# ============================================================
# 17. FEATURE-VALUE TABLE
# ============================================================

table_rows = []
used_features = set()

for rule, contribution in exp.as_list():
    actual_feature = extract_actual_feature_from_rule(
        rule,
        actual_features,
        shown_features
    )

    if actual_feature in instance_raw.index and actual_feature not in used_features:
        table_rows.append({
            "feature": actual_to_display[actual_feature],
            "value": format_value(instance_raw[actual_feature]),
            "contribution": float(contribution)
        })

        used_features.add(actual_feature)

if not table_rows:
    for actual_feature in actual_features:
        table_rows.append({
            "feature": actual_to_display[actual_feature],
            "value": format_value(instance_raw[actual_feature]),
            "contribution": 0.0
        })

# ============================================================
# 18. HTML DASHBOARD
# ============================================================

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

full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DE Women Clothing LIME Dashboard</title>

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
    <h1>DE Women Clothing – Single-Instance Explanation Dashboard</h1>

    <div class="subtitle">
        Instance index: {INSTANCE_INDEX} &nbsp;|&nbsp;
        Predicted mean: {pred_mean:.3f} &nbsp;|&nbsp;
        Actual: {actual_value:.3f} &nbsp;|&nbsp;
        Displayed features: {len(shown_features)}
    </div>

    <div class="layout">
        {summary_html}
        {lime_panel}
        {feature_table_html}
    </div>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(full_html)

print("\nDE Women Clothing LIME dashboard saved to:")
print(OUTPUT_HTML)

webbrowser.open("file:///" + OUTPUT_HTML.replace("\\", "/"))

