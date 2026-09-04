import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# EXPERIMENT 01 - BASELINE CONFIGURATION
# Bayesian Neural Network (BNN)
# Dataset: Skincare Reviews
#
# Baseline:
#   Original structured numerical features only
#
# Target:
#   rating
# ============================================================


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BNN BASELINE EXPERIMENT - SKINCARE DATASET")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# 2. DATASET PATH
# ============================================================

DATA_FILE = (
    r"D:\MAT-UNI\SEMESTER 02\THESIS"
    r"\VS Code Implementation 1.2"
    r"\reviews_1000_1500.csv"
)

TARGET_COL = "rating"


# ============================================================
# 3. COLUMNS TO DROP
# ============================================================

DROP_COLUMNS = [
    "skin_tone",
    "eye_color",
    "skin_type",
    "hair_color",
    "product_id",
    "product_name",
    "review_title",
    "review_text",
    "submission_time",
    "author_id"
]


# ============================================================
# 4. BASELINE FEATURES
# ============================================================

BASE_FEATURES = [
    "is_recommended",
    "helpfulness",
    "total_feedback_count",
    "total_neg_feedback_count",
    "total_pos_feedback_count",
    "price_usd"
]


# ============================================================
# 5. CHECK DATASET EXISTS
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n"
        "Please check the file name and extension."
    )


# ============================================================
# 6. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nOriginal dataset shape:", df.shape)

print("\nColumns in dataset:")
for column in df.columns:
    print(" -", column)


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = BASE_FEATURES + [TARGET_COL]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nThe following required columns are missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 8. DROP UNUSED COLUMNS
# ============================================================

# errors="ignore" prevents failure if one of the specified
# columns is already absent from the dataset.

df = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

print("\nDataset shape after dropping unused columns:", df.shape)


# ============================================================
# 9. RETAIN ONLY BASELINE FEATURES + TARGET
# ============================================================

df_model = df[
    BASE_FEATURES + [TARGET_COL]
].copy()

print("\nColumns used in baseline experiment:")

for column in BASE_FEATURES:
    print(" -", column)

print("\nTarget variable:")
print(" -", TARGET_COL)


# ============================================================
# 10. CONVERT VALUES TO NUMERIC
# ============================================================

for column in BASE_FEATURES + [TARGET_COL]:

    df_model[column] = pd.to_numeric(
        df_model[column],
        errors="coerce"
    )


# ============================================================
# 11. REPLACE INFINITE VALUES WITH NaN
# ============================================================

df_model.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 12. DISPLAY MISSING VALUES BEFORE CLEANING
# ============================================================

print("\nMissing values before cleaning:")

print(
    df_model.isnull().sum()
)


# ============================================================
# 13. REMOVE ROWS WITH MISSING TARGET
# ============================================================

df_model = df_model.dropna(
    subset=[TARGET_COL]
).copy()


# ============================================================
# 14. HANDLE MISSING FEATURE VALUES
# ============================================================

# Median imputation is applied only to the predictor variables.
# The target rating is never imputed.

for column in BASE_FEATURES:

    median_value = df_model[column].median()

    df_model[column] = df_model[column].fillna(
        median_value
    )


# ============================================================
# 15. FINAL DATA CHECK
# ============================================================

print("\nMissing values after cleaning:")

print(
    df_model.isnull().sum()
)

print(
    "\nFinal number of samples:",
    len(df_model)
)


# ============================================================
# 16. CREATE X AND y
# ============================================================

X = df_model[
    BASE_FEATURES
].values.astype(np.float32)

y = df_model[
    TARGET_COL
].values.astype(np.float32)


print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 17. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 18. STANDARDISE INPUT FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 19. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
).to(device)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
).to(device)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1).to(device)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1).to(device)


# ============================================================
# 20. BAYESIAN LINEAR LAYER
# ============================================================

class BayesianLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        prior_sigma=1.0
    ):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma


        # ----------------------------------------------------
        # Weight posterior parameters
        # ----------------------------------------------------

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features
            ).normal_(
                mean=0.0,
                std=0.1
            )
        )

        self.weight_rho = nn.Parameter(
            torch.empty(
                out_features,
                in_features
            ).fill_(-3.0)
        )


        # ----------------------------------------------------
        # Bias posterior parameters
        # ----------------------------------------------------

        self.bias_mu = nn.Parameter(
            torch.empty(
                out_features
            ).normal_(
                mean=0.0,
                std=0.1
            )
        )

        self.bias_rho = nn.Parameter(
            torch.empty(
                out_features
            ).fill_(-3.0)
        )


    # ========================================================
    # Convert rho to positive standard deviation
    # ========================================================

    def sigma(self, rho):

        return F.softplus(rho)


    # ========================================================
    # Sample weights using reparameterisation trick
    # ========================================================

    def forward(self, x):

        weight_sigma = self.sigma(
            self.weight_rho
        )

        bias_sigma = self.sigma(
            self.bias_rho
        )


        weight_epsilon = torch.randn_like(
            self.weight_mu
        )

        bias_epsilon = torch.randn_like(
            self.bias_mu
        )


        sampled_weight = (
            self.weight_mu
            +
            weight_sigma
            *
            weight_epsilon
        )

        sampled_bias = (
            self.bias_mu
            +
            bias_sigma
            *
            bias_epsilon
        )


        output = F.linear(
            x,
            sampled_weight,
            sampled_bias
        )


        # ----------------------------------------------------
        # KL divergence
        # Approximate posterior q(w|theta)
        # versus standard Gaussian prior
        # ----------------------------------------------------

        weight_kl = self.kl_divergence(
            self.weight_mu,
            weight_sigma
        )

        bias_kl = self.kl_divergence(
            self.bias_mu,
            bias_sigma
        )


        total_kl = (
            weight_kl
            +
            bias_kl
        )

        return output, total_kl


    # ========================================================
    # KL divergence between Gaussian posterior and prior
    # ========================================================

    def kl_divergence(
        self,
        mu,
        sigma
    ):

        prior_sigma = self.prior_sigma

        kl = (
            torch.log(
                prior_sigma / sigma
            )
            +
            (
                sigma.pow(2)
                +
                mu.pow(2)
            )
            /
            (
                2
                *
                prior_sigma ** 2
            )
            -
            0.5
        )

        return kl.sum()


# ============================================================
# 21. BAYESIAN NEURAL NETWORK
# ============================================================

class BayesianNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()


        self.layer1 = BayesianLinear(
            input_dim,
            64
        )

        self.layer2 = BayesianLinear(
            64,
            32
        )

        self.output_layer = BayesianLinear(
            32,
            1
        )


    def forward(self, x):

        x, kl1 = self.layer1(x)

        x = F.relu(x)


        x, kl2 = self.layer2(x)

        x = F.relu(x)


        x, kl3 = self.output_layer(x)


        total_kl = (
            kl1
            +
            kl2
            +
            kl3
        )

        return x, total_kl


# ============================================================
# 22. INITIALISE BNN
# ============================================================

model = BayesianNeuralNetwork(
    input_dim=len(BASE_FEATURES)
).to(device)


print("\nBNN Architecture:")
print(model)


# ============================================================
# 23. OPTIMISER
# ============================================================

LEARNING_RATE = 0.001

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 24. TRAINING SETTINGS
# ============================================================

EPOCHS = 500

# Scale KL term relative to training set size
KL_WEIGHT = 1.0 / len(X_train_tensor)


print("\nTraining configuration:")
print("Epochs       :", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("KL weight    :", KL_WEIGHT)


# ============================================================
# 25. TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BNN")
print("=" * 70)


for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    predictions, kl_loss = model(
        X_train_tensor
    )


    # --------------------------------------------------------
    # Data fitting component
    # --------------------------------------------------------

    mse_loss = F.mse_loss(
        predictions,
        y_train_tensor
    )


    # --------------------------------------------------------
    # Variational loss
    #
    # Loss = likelihood/data-fitting component
    #        + KL complexity component
    # --------------------------------------------------------

    total_loss = (
        mse_loss
        +
        KL_WEIGHT
        *
        kl_loss
    )


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    total_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if (
        epoch == 0
        or
        (epoch + 1) % 50 == 0
    ):

        print(
            f"Epoch {epoch + 1:4d}/{EPOCHS} | "
            f"Total Loss: {total_loss.item():.6f} | "
            f"MSE: {mse_loss.item():.6f} | "
            f"KL: {kl_loss.item():.6f}"
        )


# ============================================================
# 26. MONTE CARLO POSTERIOR PREDICTION
# ============================================================

# Each forward pass samples a different set of Bayesian weights.

MC_SAMPLES = 100

model.eval()

all_predictions = []


print("\nGenerating Bayesian posterior predictions...")


with torch.no_grad():

    for _ in range(MC_SAMPLES):

        prediction, _ = model(
            X_test_tensor
        )

        all_predictions.append(
            prediction
            .cpu()
            .numpy()
            .flatten()
        )


all_predictions = np.array(
    all_predictions
)


print(
    "Prediction matrix shape:",
    all_predictions.shape
)


# ============================================================
# 27. PREDICTIVE MEAN
# ============================================================

prediction_mean = np.mean(
    all_predictions,
    axis=0
)


# ============================================================
# 28. PREDICTIVE STANDARD DEVIATION
# ============================================================

prediction_std = np.std(
    all_predictions,
    axis=0
)


# ============================================================
# 29. 95% PREDICTIVE INTERVAL
# ============================================================

lower_95 = prediction_mean - (
    1.96 * prediction_std
)

upper_95 = prediction_mean + (
    1.96 * prediction_std
)


# ============================================================
# 30. EVALUATION METRICS
# ============================================================

MAE = mean_absolute_error(
    y_test,
    prediction_mean
)


RMSE = np.sqrt(
    mean_squared_error(
        y_test,
        prediction_mean
    )
)


R2 = r2_score(
    y_test,
    prediction_mean
)


MEAN_PREDICTIVE_STD = np.mean(
    prediction_std
)


MEAN_INTERVAL_WIDTH = np.mean(
    upper_95 - lower_95
)


# ============================================================
# 31. 95% INTERVAL COVERAGE
# ============================================================

coverage = np.mean(
    (
        y_test >= lower_95
    )
    &
    (
        y_test <= upper_95
    )
) * 100


# ============================================================
# 32. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 01 RESULTS")
print("BNN BASELINE - SKINCARE")
print("=" * 70)

print(
    f"MAE                         : {MAE:.6f}"
)

print(
    f"RMSE                        : {RMSE:.6f}"
)

print(
    f"R²                          : {R2:.6f}"
)

print(
    f"Mean Predictive Std         : {MEAN_PREDICTIVE_STD:.6f}"
)

print(
    f"Mean 95% Interval Width     : {MEAN_INTERVAL_WIDTH:.6f}"
)

print(
    f"95% Interval Coverage       : {coverage:.2f}%"
)

print("=" * 70)


# ============================================================
# 33. CREATE PREDICTION RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame({

    "Actual_Rating":
        y_test,

    "Predicted_Rating":
        prediction_mean,

    "Predictive_Std":
        prediction_std,

    "Lower_95":
        lower_95,

    "Upper_95":
        upper_95
})


# ============================================================
# 34. DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")

print(
    results_df.head(10)
)


# ============================================================
# 35. SAVE RESULTS
# ============================================================

OUTPUT_FILE = os.path.join(
    os.path.dirname(DATA_FILE),
    "skincare_BNN_baseline_results.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nPrediction results saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 36. EXPERIMENT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE EXPERIMENT SUMMARY")
print("=" * 70)

print(
    "Dataset:",
    os.path.basename(DATA_FILE)
)

print(
    "Target:",
    TARGET_COL
)

print(
    "Number of baseline features:",
    len(BASE_FEATURES)
)

print("\nBaseline features:")

for feature in BASE_FEATURES:
    print(" -", feature)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "MC posterior samples:",
    MC_SAMPLES
)

print("=" * 70)