import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================
RANDOM_STATE = 42
N_ROWS = 2000
TEST_SIZE = 0.2

EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3

# BNN settings
PRIOR_SIGMA = 1.0
KL_WEIGHT = 1e-3          # try 1e-3 or 1e-4 if KL is too strong
N_PRED_SAMPLES = 50       # predictive samples for uncertainty

# Files (edit if needed)
BASE_CSV = r"women_clothing_reviews_normalized.csv"
LLM_CSV  = r"outputs_exp02_women_clothing_ollama_2000\women_clothing_llm_features_2000.csv"

# Output
OUT_DIR = "outputs_exp02_women_bnn"
os.makedirs(OUT_DIR, exist_ok=True)
PRED_OUT = os.path.join(OUT_DIR, "women_exp02_bnn_predictions.csv")
METRICS_OUT = os.path.join(OUT_DIR, "women_exp02_bnn_metrics.txt")

TARGET_COL = "Rating"

BASE_FEATURES = [
    "Age",
    "Recommended IND",
    "Positive Feedback Count"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# BAYESIAN LINEAR LAYER (Variational)
# ============================================================
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational posterior parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features).normal_(-3, 0.1))

        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(out_features).normal_(-3, 0.1))

    def _softplus(self, rho):
        return torch.log1p(torch.exp(rho))

    def forward(self, x):
        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)

        # Reparameterization trick
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        # KL(q||p) where q is Normal(mu, sigma), p is Normal(0, prior_sigma)
        weight_sigma = self._softplus(self.weight_rho)
        bias_sigma = self._softplus(self.bias_rho)

        prior_var = self.prior_sigma ** 2

        # KL for weights
        kl_w = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * prior_var)
            - 0.5
        ).sum()

        # KL for bias
        kl_b = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * prior_var)
            - 0.5
        ).sum()

        return kl_w + kl_b


# ============================================================
# BNN MODEL
# ============================================================
class BNNRegressor(nn.Module):
    def __init__(self, input_dim, prior_sigma=1.0):
        super().__init__()
        self.b1 = BayesianLinear(input_dim, 64, prior_sigma)
        self.b2 = BayesianLinear(64, 32, prior_sigma)
        self.b3 = BayesianLinear(32, 1, prior_sigma)

    def forward(self, x):
        x = torch.relu(self.b1(x))
        x = torch.relu(self.b2(x))
        x = self.b3(x)
        return x

    def kl(self):
        return self.b1.kl_divergence() + self.b2.kl_divergence() + self.b3.kl_divergence()


# ============================================================
# DATA LOADING + MERGE
# ============================================================
def load_and_merge():
    base = pd.read_csv(BASE_CSV)
    llm = pd.read_csv(LLM_CSV)

    # LLM columns
    llm_cols = [c for c in llm.columns if c.startswith("llm_")]
    if len(llm_cols) == 0:
        raise ValueError("No llm_* columns found in LLM CSV.")

    # Ensure target exists
    if TARGET_COL not in base.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in baseline CSV.")

    # Coerce numeric base cols + target
    base[TARGET_COL] = pd.to_numeric(base[TARGET_COL], errors="coerce")
    base = base.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Sample fixed 2000 first for consistency
    if len(base) < N_ROWS:
        raise ValueError(f"Baseline dataset has only {len(base)} rows; cannot sample {N_ROWS}.")
    base = base.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    # Align rows (should be 2000)
    min_rows = min(len(base), len(llm))
    base = base.iloc[:min_rows].reset_index(drop=True)
    llm = llm.iloc[:min_rows].reset_index(drop=True)

    df = pd.concat([base, llm[llm_cols]], axis=1)

    # Build features
    missing_base = [c for c in BASE_FEATURES if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing baseline features in BASE_CSV: {missing_base}")

    feature_cols = BASE_FEATURES + llm_cols

    # Numeric coercion + fill NaNs
    for c in feature_cols + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].median())

    return df, feature_cols, llm_cols


# ============================================================
# TRAIN + EVAL
# ============================================================
def main():
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df, feature_cols, llm_cols = load_and_merge()
    print("Merged DF shape:", df.shape)
    print("LLM features:", len(llm_cols))

    X = df[feature_cols].values
    y = df[TARGET_COL].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Split
    idx = np.arange(len(df))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Model
    model = BNNRegressor(input_dim=X_train.shape[1], prior_sigma=PRIOR_SIGMA).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    n_train = X_train.shape[0]

    # Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for i in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            opt.zero_grad()

            pred = model(xb)
            mse = F.mse_loss(pred, yb)

            kl = model.kl() / n_train  # normalize KL by number of samples
            loss = mse + KL_WEIGHT * kl

            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={epoch_loss:.4f} (mse+kl)")

    # Predictive sampling
    model.eval()
    with torch.no_grad():
        samples = []
        for _ in range(N_PRED_SAMPLES):
            y_hat = model(X_test_t).cpu().numpy().reshape(-1)
            samples.append(y_hat)
        samples = np.array(samples)

    y_pred_mean = samples.mean(axis=0)
    y_pred_std  = samples.std(axis=0)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred_mean)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_mean)))
    r2 = r2_score(y_test, y_pred_mean)

    print("\n===== Experiment 02 Results (Women | BNN | Baseline + LLM) =====")
    print(f"Rows used: {len(df)}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "row_index": idx_test,
        "true_rating": y_test,
        "predicted_rating_mean": y_pred_mean,
        "predicted_rating_std": y_pred_std
    }).sort_values("row_index")

    pred_df.to_csv(PRED_OUT, index=False)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        f.write("Experiment 02 (Women Clothing | BNN | Baseline + LLM)\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"LLM features used: {len(llm_cols)}\n")
        f.write(f"EPOCHS: {EPOCHS}, KL_WEIGHT: {KL_WEIGHT}, PRIOR_SIGMA: {PRIOR_SIGMA}, N_PRED_SAMPLES: {N_PRED_SAMPLES}\n")

    print("\nSaved predictions to:", PRED_OUT)
    print("Saved metrics to     :", METRICS_OUT)


if __name__ == "__main__":
    main()
