# de_stress_experiment02_train.py
# Deep Ensembles — Experiment 02 (Stress): Baseline numeric + LLM thematic features
# Run normally:  python de_stress_experiment02_train.py
#
# ✅ Uses same split style as Exp-01 (seeded train/test, val from train)
# ✅ Standardizes X using train-only stats
# ✅ Trains an ensemble of MLP regressors
# ✅ Reports MAE, RMSE, R^2
# ✅ Saves predictions (mean + std) to CSV
#
# IMPORTANT:
# - Set DATA_PATH to your Exp-02 CSV:
#   stress_analysis_normalized_fixed2000_with_llm.csv
# - Ensure TARGET_COL = "confidence"
# - LLM columns must match exactly the 8 T1..T8 names

import os
import json
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================
# SETTINGS (EDIT ONLY THIS)
# =========================
DATA_PATH = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\stress_analysis_normalized_fixed2000_with_llm.csv"
TARGET_COL = "confidence"

OUT_DIR = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\outputs_stress_de_exp02"

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.20

ENSEMBLE_SIZE = 5

EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 128

HIDDEN_DIM = 128
DROPOUT = 0.05
LR = 1e-3
WEIGHT_DECAY = 1e-4

FORCE_CPU = False  # set True if your uni GPU gives "operation not supported"

# LLM theme columns (must exist in Exp-02 CSV)
LLM_COLS = [
    "T1_STRESS_INTENSITY",
    "T2_ANXIETY_WORRY",
    "T3_BURNOUT_EXHAUSTION",
    "T4_SLEEP_PROBLEMS",
    "T5_WORK_STUDY_PRESSURE",
    "T6_RELATIONSHIP_SOCIAL_STRESS",
    "T7_PHYSICAL_SYMPTOMS",
    "T8_COPING_STRATEGY",
]

# Columns to exclude from baseline numeric selection (common non-features)
# (We will keep numeric columns except these)
EXCLUDE_COLS = [
    TARGET_COL,
    "text",        # your text column
    "row_index",   # if present from FAST script
]

# If you have an ID column you want excluded, add it here, e.g. "id"
# EXCLUDE_COLS.append("id")


# =========================
# IMPLEMENTATION
# =========================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    patience: int


def load_and_prepare(data_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found.\nAvailable columns: {list(df.columns)}"
        )

    missing_llm = [c for c in LLM_COLS if c not in df.columns]
    if missing_llm:
        raise ValueError(f"Missing LLM columns in CSV: {missing_llm}")

    # Keep all numeric columns except excluded ones
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_numeric = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    # Ensure we include LLM columns even if they were read as non-numeric somehow
    # (but in your case they should be numeric)
    for c in LLM_COLS:
        if c not in keep_numeric and c != target:
            keep_numeric.append(c)

    if len(keep_numeric) == 0:
        raise ValueError("No numeric feature columns found after exclusions.")

    X = df[keep_numeric].copy()
    y = df[target].copy()

    # Clean + missing handling
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.fillna(y.median())

    return X, y


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="mean")
    total = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = mse(pred, yb)
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


def train_one_member(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    member_seed: int,
    device: torch.device,
    cfg: TrainConfig,
) -> nn.Module:
    set_global_seed(member_seed)

    model = MLPRegressor(in_dim=X_train.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss(reduction="mean")

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, cfg.batch_size)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            optimizer.step()

        val = eval_loss(model, val_loader, device)
        if val + 1e-12 < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def ensemble_predict(
    models: List[nn.Module],
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    member_preds = []
    for m in models:
        m.eval()
        preds = []
        for (xb,) in loader:
            xb = xb.to(device)
            p = m(xb).detach().cpu().numpy()
            preds.append(p)
        member_preds.append(np.concatenate(preds, axis=0))

    P = np.stack(member_preds, axis=0)  # (M, N)
    mean = P.mean(axis=0)
    std = P.std(axis=0, ddof=0)
    return mean, std


def main():
    safe_makedirs(OUT_DIR)
    set_global_seed(SEED)
    device = pick_device(force_cpu=FORCE_CPU)
    print(f"Using device: {device}")

    X_df, y_ser = load_and_prepare(DATA_PATH, TARGET_COL)

    # Split test
    idx = np.arange(len(X_df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )

    X_train_df = X_df.iloc[train_idx].reset_index(drop=True)
    y_train = y_ser.iloc[train_idx].to_numpy(dtype=np.float32)

    X_test_df = X_df.iloc[test_idx].reset_index(drop=True)
    y_test = y_ser.iloc[test_idx].to_numpy(dtype=np.float32)

    # Validation from train
    tr_idx2, val_idx2 = train_test_split(
        np.arange(len(X_train_df)),
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=SEED,
        shuffle=True,
    )

    X_tr = X_train_df.iloc[tr_idx2].to_numpy(dtype=np.float32)
    y_tr = y_train[tr_idx2]
    X_val = X_train_df.iloc[val_idx2].to_numpy(dtype=np.float32)
    y_val = y_train[val_idx2]

    # Standardize (train-only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test_df.to_numpy(dtype=np.float32))

    # Save split + feature list
    split_info = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "val_size_from_train": VAL_SIZE_FROM_TRAIN,
        "n_rows": int(len(X_df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "target": TARGET_COL,
        "feature_cols": list(X_df.columns),
        "llm_cols": LLM_COLS,
        "device": str(device),
        "data_path": DATA_PATH,
    }
    with open(os.path.join(OUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    np.save(os.path.join(OUT_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(OUT_DIR, "test_idx.npy"), test_idx)

    scaler_dump = {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist(), "var_": scaler.var_.tolist()}
    with open(os.path.join(OUT_DIR, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump(scaler_dump, f, indent=2)

    cfg = TrainConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        patience=PATIENCE,
    )

    # Train ensemble
    models: List[nn.Module] = []
    for m in range(ENSEMBLE_SIZE):
        member_seed = SEED + 1000 * (m + 1)
        print(f"Training member {m+1}/{ENSEMBLE_SIZE} (seed={member_seed}) ...")
        model = train_one_member(X_tr_s, y_tr, X_val_s, y_val, member_seed, device, cfg)
        models.append(model)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"member_{m+1:02d}_state.pt"))

    # Predict on test
    y_mean, y_std = ensemble_predict(models, X_test_s, device=device, batch_size=2048)

    # Metrics
    mae = mean_absolute_error(y_test, y_mean)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_mean)))  # sklearn-safe
    r2 = r2_score(y_test, y_mean)

    print("\n===== Experiment-02 Results (Stress | Deep Ensemble | Baseline + LLM) =====")
    print(f"Data         : {DATA_PATH}")
    print(f"Rows used    : {len(X_df)}")
    print(f"Features used: {X_df.shape[1]} (baseline numeric + LLM)")
    print("------------------------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    # Save metrics
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "ensemble_size": int(ENSEMBLE_SIZE),
        "epochs": int(EPOCHS),
        "patience": int(PATIENCE),
        "batch_size": int(BATCH_SIZE),
        "hidden_dim": int(HIDDEN_DIM),
        "dropout": float(DROPOUT),
        "lr": float(LR),
        "weight_decay": float(WEIGHT_DECAY),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions + uncertainty
    pred_df = pd.DataFrame(
        {
            "y_true": y_test.astype(np.float32),
            "y_pred_mean": y_mean.astype(np.float32),
            "y_pred_std": y_std.astype(np.float32),
        }
    )
    pred_path = os.path.join(OUT_DIR, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved metrics     : {os.path.join(OUT_DIR, 'metrics.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
    