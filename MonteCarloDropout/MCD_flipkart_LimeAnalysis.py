# mcd_flipkart_lime_dashboard.py

import os
import html
import random
import warnings
import webbrowser

import joblib
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# =========================================================
# SETTINGS
# =========================================================
SEED = 42
TEST_SIZE = 0.20

INSTANCE_INDEX = 10
NUM_FEATURES = 12

WORK_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\3 MCD_SHAP LIME"
DATA_PATH = os.path.join(WORK_DIR, "flipkart_merged_original_withllm_features.csv")

MODEL_PATH = os.path.join(WORK_DIR, "mcd_flipkart_model.pt")
SCALER_PATH = os.path.join(WORK_DIR, "mcd_flipkart_scaler.joblib")
OUTPUT_HTML = os.path.join(WORK_DIR, "mcd_flipkart_lime_dashboard.html")
PRED_PATH = os.path.join(WORK_DIR, "mcd_flipkart_predictions.csv")

TARGET_COL = "rating"

# Requested dropped columns + extra columns that caused mismatch
DROP_COLS = [
    "product_id",
    "product_title",
    "upvotes",
    "downvotes",
    "date_ordinal",
    "location_encoded",
    "source_index",
    "overall_sentiment",
    "confidence",
    "summary",
    "review",
    "parse_error",
    "runtime_error"
]

# MCD SETTINGS
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.20
MC_SAMPLES = 50


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
# MODEL
# =========================================================
class MCDRegressor(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

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

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

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
            pred = model(X_tensor).squeeze(1).cpu().numpy()
            preds.append(pred)

    preds = np.array(preds)  # [mc_samples, n_samples]
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std, preds


# =========================================================
# HELPERS
# =========================================================
def find_actual_target_column(df, requested_target):
    for col in df.columns:
        if col.strip().lower() == requested_target.strip().lower():
            return col
    raise ValueError(f"Target column '{requested_target}' not found in dataset.")


def add_llm_prefix_if_needed(col_name, target_col):
    """
    Add llm_ prefix to displayed feature names for consistency.
    Do not rename the target column.
    """
    if col_name == target_col:
        return col_name
    if col_name.startswith("llm_"):
        return col_name
    return f"llm_{col_name}"


def extract_base_feature(rule, columns):
    sorted_cols = sorted(columns, key=len, reverse=True)
    rule_lower = rule.lower()
    for col in sorted_cols:
        if col.lower() in rule_lower:
            return col
    return rule


# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Original dataset shape:", df.shape)

actual_target_col = find_actual_target_column(df, TARGET_COL)
print("Detected target column:", actual_target_col)

# =========================================================
# DROP REQUESTED COLUMNS + EXTRA NON-MODEL COLUMNS
# =========================================================
existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
df = df.drop(columns=existing_drop_cols, errors="ignore")
print("Dropped columns:", existing_drop_cols)

# =========================================================
# RENAME COLUMNS FOR DISPLAY CONSISTENCY
# =========================================================
rename_map = {}
for col in df.columns:
    rename_map[col] = add_llm_prefix_if_needed(col, actual_target_col)

df = df.rename(columns=rename_map)
actual_target_col = rename_map.get(actual_target_col, actual_target_col)

print("\nRenamed feature columns for display consistency:")
for old_name, new_name in rename_map.items():
    if old_name != new_name:
        print(f" - {old_name} --> {new_name}")

# =========================================================
# CLEAN DATA
# =========================================================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.select_dtypes(include=[np.number]).copy()

df = df.dropna(subset=[actual_target_col])

for col in df.columns:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

print("After cleaning:", df.shape)

# =========================================================
# FEATURES / TARGET
# =========================================================
X = df.drop(columns=[actual_target_col]).copy()
y = df[actual_target_col].copy()
feature_names = X.columns.tolist()

print("Number of input features:", X.shape[1])

if INSTANCE_INDEX >= len(X):
    raise ValueError(f"INSTANCE_INDEX {INSTANCE_INDEX} is out of range for dataset size {len(X)}")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

X_test_raw = X_test.copy()

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# =========================================================
# SCALE FEATURES
# =========================================================
def fit_and_save_scaler():
    scaler = StandardScaler()
    X_train_scaled_local = scaler.fit_transform(X_train)
    X_test_scaled_local = scaler.transform(X_test)

    if not np.isfinite(X_train_scaled_local).all():
        raise ValueError("Non-finite values found in X_train_scaled after scaling.")
    if not np.isfinite(X_test_scaled_local).all():
        raise ValueError("Non-finite values found in X_test_scaled after scaling.")

    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to:", SCALER_PATH)
    return scaler, X_train_scaled_local, X_test_scaled_local


# =========================================================
# TRAIN MODEL IF NEEDED
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

if (not os.path.exists(MODEL_PATH)) or (not os.path.exists(SCALER_PATH)):
    print("\nSaved model/scaler not found. Training MCD model first...\n")

    scaler, X_train_scaled, X_test_scaled = fit_and_save_scaler()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MCDRegressor(input_dim=X_train.shape[1], dropout_rate=DROPOUT_RATE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training MCD model...")
    train_mcd_model(model, train_loader, criterion, optimizer, epochs=EPOCHS, device=device)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "feature_names": feature_names,
        "target_col": actual_target_col,
        "drop_cols": DROP_COLS,
        "dropout_rate": DROPOUT_RATE,
        "seed": SEED
    }, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

    y_pred_mean, y_pred_std, _ = mc_predict(model, X_test_tensor, mc_samples=MC_SAMPLES, device=device)

    y_true = y_test.values
    mae = mean_absolute_error(y_true, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2 = r2_score(y_true, y_pred_mean)

    print("\n===== MCD FLIPKART RESULTS =====")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"Mean predictive std: {y_pred_std.mean():.6f}")

    pred_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred_mean,
        "PredictiveStd": y_pred_std
    })
    pred_df.to_csv(PRED_PATH, index=False)
    print("Predictions saved to:", PRED_PATH)

else:
    print("\nFound existing model and scaler. Skipping training.\n")

# =========================================================
# LOAD MODEL + SCALER FOR DASHBOARD
# =========================================================
scaler = joblib.load(SCALER_PATH)

checkpoint = torch.load(MODEL_PATH, map_location=device)
input_dim = checkpoint["input_dim"]
saved_dropout_rate = checkpoint.get("dropout_rate", DROPOUT_RATE)

model = MCDRegressor(input_dim=input_dim, dropout_rate=saved_dropout_rate)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# Extra protection: align features with saved model feature names
saved_feature_names = checkpoint.get("feature_names", feature_names)

missing_in_current = [c for c in saved_feature_names if c not in X_train.columns]
extra_in_current = [c for c in X_train.columns if c not in saved_feature_names]

if missing_in_current:
    raise ValueError(
        "Current dataset is missing feature(s) expected by the saved model: "
        + ", ".join(missing_in_current)
    )

if extra_in_current:
    print("\nDropping extra feature(s) not used by saved model:")
    for c in extra_in_current:
        print(f" - {c}")
    X_train = X_train[saved_feature_names].copy()
    X_test = X_test[saved_feature_names].copy()
    X_test_raw = X_test_raw[saved_feature_names].copy()
    feature_names = saved_feature_names

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed successfully.")
print("MCD model loaded successfully.")

instance_raw = X_test.iloc[INSTANCE_INDEX].copy()

# =========================================================
# PREDICTION FUNCTIONS FOR LIME
# =========================================================
def predict_fn(x):
    """
    LIME expects raw arrays. We scale them here and return MCD mean predictions.
    """
    x_df = pd.DataFrame(x, columns=feature_names)
    x_scaled = scaler.transform(x_df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    pred_mean, _, _ = mc_predict(model, x_tensor, mc_samples=MC_SAMPLES, device=device)
    return pred_mean


def mcd_predict_single_distribution(x_row):
    """
    Returns one prediction from each Monte Carlo forward pass for a single instance.
    """
    x_row = x_row.reshape(1, -1)
    x_df = pd.DataFrame(x_row, columns=feature_names)
    x_scaled = scaler.transform(x_df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    model.eval()
    enable_dropout(model)

    preds = []
    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        for _ in range(MC_SAMPLES):
            pred = model(x_tensor).squeeze(1).cpu().numpy()[0]
            preds.append(pred)

    return np.array(preds, dtype=float)

# =========================================================
# LIME EXPLAINER
# =========================================================
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=True,
    random_state=SEED
)

print("Generating LIME explanation for test instance:", INSTANCE_INDEX)

exp = explainer.explain_instance(
    data_row=X_test.iloc[INSTANCE_INDEX].values,
    predict_fn=predict_fn,
    num_features=NUM_FEATURES
)

# =========================================================
# PREDICTION SUMMARY
# =========================================================
mc_preds = mcd_predict_single_distribution(X_test.iloc[INSTANCE_INDEX].values)

pred_mean = float(np.mean(mc_preds))
pred_std = float(np.std(mc_preds, ddof=1)) if len(mc_preds) > 1 else 0.0
pred_low = float(np.percentile(mc_preds, 2.5))
pred_high = float(np.percentile(mc_preds, 97.5))
actual_value = float(y_test.iloc[INSTANCE_INDEX])

p_lt_3 = float(np.mean(mc_preds < 3.0))
p_3_to_4 = float(np.mean((mc_preds >= 3.0) & (mc_preds < 4.0)))
p_ge_4 = float(np.mean(mc_preds >= 4.0))

# =========================================================
# FEATURE TABLE
# =========================================================
lime_rules_all = exp.as_list()

table_rows = []
used_features = set()

for rule, contribution in lime_rules_all:
    base_feature = extract_base_feature(rule, feature_names)

    if base_feature in instance_raw.index and base_feature not in used_features:
        raw_value = instance_raw[base_feature]

        if isinstance(raw_value, (np.floating, float)):
            raw_value_fmt = f"{raw_value:.4f}"
        elif isinstance(raw_value, (np.integer, int)):
            raw_value_fmt = str(int(raw_value))
        else:
            raw_value_fmt = str(raw_value)

        table_rows.append({
            "feature": base_feature,
            "value": raw_value_fmt,
            "contribution": contribution
        })
        used_features.add(base_feature)

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

if table_html_rows.strip() == "":
    table_html_rows = """
    <tr>
        <td colspan="3">No features available for display in this local explanation.</td>
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
                <th>LIME Contribution</th>
            </tr>
        </thead>
        <tbody>
            {table_html_rows}
        </tbody>
    </table>
    <div class="note">Values shown are raw feature values from the selected test instance.</div>
</div>
"""

# =========================================================
# LIME PANEL
# =========================================================
lime_panel = f"""
<div class="card">
    <h2>LIME Explanation</h2>
    <iframe class="lime-frame" srcdoc="{html.escape(exp.as_html())}"></iframe>
</div>
"""

# =========================================================
# PREDICTION SUMMARY HTML
# =========================================================
def build_probability_bar(label, prob, color):
    return f"""
    <div class="prob-row">
        <div class="prob-label">{html.escape(label)}</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{prob*100:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{prob:.2f}</div>
    </div>
    """

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
    {build_probability_bar("Rating < 3", p_lt_3, "#d9d9d9")}
    {build_probability_bar("3 ≤ Rating < 4", p_3_to_4, "#ffbf00")}
    {build_probability_bar("Rating ≥ 4", p_ge_4, "#ff7f0e")}
</div>
"""

# =========================================================
# FINAL HTML
# =========================================================
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MCD Flipkart – Single-Instance Explanation Dashboard</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 0;
        padding: 0;
        background: #f5f5f5;
        color: #222;
    }}
    .page {{
        padding: 12px;
    }}
    h1 {{
        margin: 0 0 8px 0;
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
        grid-template-columns: 1fr 1.42fr 1fr;
        gap: 18px;
        align-items: start;
    }}
    .card {{
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        padding: 18px;
    }}
    .card h2 {{
        margin-top: 0;
        margin-bottom: 14px;
        font-size: 20px;
        font-weight: 800;
    }}
    .card h3 {{
        margin-top: 18px;
        margin-bottom: 10px;
        font-size: 15px;
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
        padding: 12px 12px;
        background: #fafafa;
        border-radius: 8px;
    }}
    .k {{
        color: #555;
        font-weight: 700;
    }}
    .v {{
        color: #111;
        font-weight: 800;
    }}
    .prob-row {{
        display: grid;
        grid-template-columns: 135px 1fr 45px;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
    }}
    .prob-label {{
        font-size: 14px;
    }}
    .prob-bar-wrap {{
        height: 18px;
        background: #e6e6e6;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #d9d9d9;
    }}
    .prob-bar {{
        height: 100%;
        border-radius: 10px;
    }}
    .prob-value {{
        text-align: right;
        font-weight: 800;
        font-size: 14px;
    }}
    .lime-frame {{
        width: 100%;
        height: 760px;
        border: none;
        border-radius: 8px;
        background: white;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    th, td {{
        border-bottom: 1px solid #e6e6e6;
        padding: 10px 8px;
        text-align: left;
        vertical-align: top;
    }}
    th {{
        background: #fafafa;
        font-weight: 800;
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
        margin-top: 12px;
        color: #666;
        font-size: 12px;
    }}
</style>
</head>
<body>
    <div class="page">
        <h1>MCD Flipkart – Single-Instance Explanation Dashboard</h1>
        <div class="subtitle">
            Instance index: {INSTANCE_INDEX} &nbsp;|&nbsp;
            Predicted mean: {pred_mean:.3f} &nbsp;|&nbsp;
            Actual: {actual_value:.3f}
        </div>

        <div class="layout">
            {summary_html}
            {lime_panel}
            {feature_table_html}
        </div>
    </div>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(full_html)

print("HTML dashboard saved to:")
print(OUTPUT_HTML)

webbrowser.open("file:///" + OUTPUT_HTML.replace("\\", "/"))

