import os
import json
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PATHS ------------------
DATA_PATH = "wine_quality.csv"
OUTPUT_DIR = "outputs"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# ------------------ STRATIFICATION ------------------
y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y_bins,
    random_state=42
)

results = []

# ------------------ EXPERIMENT CONFIGS ------------------
experiments = [
    {
        "id": "EXP-01",
        "model": LinearRegression(),
        "name": "Linear Regression",
        "preprocessing": None,
        "feature_selection": "All features",
        "hyperparameters": "Default"
    },
    {
        "id": "EXP-02",
        "model": Ridge(alpha=0.75),
        "name": "Ridge Regression",
        "preprocessing": "Standardization",
        "feature_selection": "Correlation-based",
        "hyperparameters": "alpha=0.75"
    },
    {
        "id": "EXP-03",
        "model": RandomForestRegressor(
            n_estimators=25, max_depth=13, random_state=42
        ),
        "name": "Random Forest",
        "preprocessing": None,
        "feature_selection": "All features",
        "hyperparameters": "25 trees, depth=13"
    },
    {
        "id": "EXP-04",
        "model": RandomForestRegressor(
            n_estimators=105, max_depth=15, random_state=42
        ),
        "name": "Random Forest",
        "preprocessing": None,
        "feature_selection": "Selected features",
        "hyperparameters": "105 trees, depth=15"
    }
]

# ------------------ RUN EXPERIMENTS ------------------
for exp in experiments:
    Xtr, Xte = X_train.copy(), X_test.copy()

    # Preprocessing
    if exp["preprocessing"] == "Standardization":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    # Train
    model = exp["model"]
    model.fit(Xtr, y_train)

    # Predict
    y_pred = model.predict(Xte)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model
    model_path = os.path.join(
        OUTPUT_DIR, f"{exp['id']}_model.joblib"
    )
    joblib.dump(model, model_path)

    # Store results
    results.append({
        "experiment_id": exp["id"],
        "model": exp["name"],
        "hyperparameters": exp["hyperparameters"],
        "preprocessing": exp["preprocessing"] or "None",
        "feature_selection": exp["feature_selection"],
        "train_test_split": "80/20 (Stratified)",
        "mse": round(mse, 6),
        "r2_score": round(r2, 6)
    })

    print(f"{exp['id']} | MSE: {mse:.4f} | R2: {r2:.4f}")

# ------------------ SAVE RESULTS ------------------
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nAll experiments completed. Results saved.")
