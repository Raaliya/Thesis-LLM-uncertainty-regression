import os
import math
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================
# Reproducibility
# ==========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================
# Dataset
# ==========================
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ==========================
# MC Dropout Model (Regression)
# ==========================
class MCDropoutRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128, 64), dropout_p=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))  # dropout stays active during MC inference
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================
# Train / Eval Helpers
# ==========================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def mc_dropout_predict(model, X: np.ndarray, device, T: int = 50, batch_size: int = 4096):
    """
    MC Dropout prediction:
    - keep dropout ACTIVE by calling model.train()
    - run T stochastic forward passes
    Returns:
        mean_pred: (N,)
        std_pred:  (N,)
    """
    model.train()  # IMPORTANT: dropout ON
    X_tensor = torch.tensor(X, dtype=torch.float32)

    preds_T = []
    for t in range(T):
        preds = []
        for i in range(0, X_tensor.size(0), batch_size):
            xb = X_tensor[i:i + batch_size].to(device)
            yhat = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(yhat)
        preds_T.append(np.concatenate(preds, axis=0))

    preds_T = np.stack(preds_T, axis=0)  # (T, N)
    mean_pred = preds_T.mean(axis=0)
    std_pred = preds_T.std(axis=0)
    return mean_pred, std_pred


def main():
    set_seed(42)

    # ==========================
    # Paths / Config
    # ==========================
    CSV_PATH = "amazon_reviews.csv"  # <-- change if needed
    OUT_DIR = "outputs_exp01_amazon"
    os.makedirs(OUT_DIR, exist_ok=True)

    TARGET = "Score"
    NUMERIC_COLUMNS = ["HelpfulnessNumerator", "HelpfulnessDenominator", "Time"]

    # Training hyperparams
    BATCH_SIZE = 4096
    EPOCHS = 12
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT_P = 0.2
    MC_SAMPLES = 50  # T forward passes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ==========================
    # Load data
    # ==========================
    df = pd.read_csv(CSV_PATH)

    # Basic cleaning (optional but recommended)
    # Ensure numeric and handle missing safely
    for c in NUMERIC_COLUMNS + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUMERIC_COLUMNS + [TARGET]).copy()

    X = df[NUMERIC_COLUMNS].values
    y = df[TARGET].values.astype(np.float32)

    # ==========================
    # Split: Train / Val / Test
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # ==========================
    # Scale features (fit on TRAIN only)
    # ==========================
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    # ==========================
    # DataLoaders
    # ==========================
    train_loader = DataLoader(
        NumpyDataset(X_train_s, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        NumpyDataset(X_val_s, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ==========================
    # Model
    # ==========================
    model = MCDropoutRegressor(
        in_dim=X_train_s.shape[1],
        hidden_dims=(128, 64),
        dropout_p=DROPOUT_P
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # ==========================
    # Train with simple early stopping
    # ==========================
    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "best_model.pt")
    patience = 3
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_loss(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:02d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device))

    # ==========================
    # MC Dropout Inference on TEST
    # ==========================
    mean_pred, std_pred = mc_dropout_predict(
        model, X_test_s, device=device, T=MC_SAMPLES, batch_size=BATCH_SIZE
    )

    # Metrics (in original target scale; Score not scaled)
    mae = mean_absolute_error(y_test, mean_pred)
    rmse = math.sqrt(mean_squared_error(y_test, mean_pred))
    r2 = r2_score(y_test, mean_pred)

    print("\n===== Experiment 01 Results (Score regression) =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")
    print(f"Avg predictive std (uncertainty): {std_pred.mean():.4f}")

    # Save predictions
    out_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_mean": mean_pred,
        "y_pred_std": std_pred
    })
    out_csv = os.path.join(OUT_DIR, "test_predictions_mc_dropout.csv")
    out_df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
