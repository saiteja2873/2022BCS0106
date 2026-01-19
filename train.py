import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PATHS ------------------
DATA_PATH = "wine_quality.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
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

# ------------------ TRAIN BEST MODEL (EXP-03) ------------------
model = RandomForestRegressor(n_estimators=25, max_depth=13, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ SAVE ARTIFACTS ------------------
joblib.dump(model, MODEL_PATH)

results = [
    {
        "experiment_id": "EXP-03",
        "model": "Random Forest",
        "hyperparameters": "25 trees, depth=13",
        "preprocessing": "None",
        "feature_selection": "All features",
        "train_test_split": "80/20 (Stratified)",
        "mse": round(mse, 6),
        "r2_score": round(r2, 6)
    }
]

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Model saved to outputs/model.joblib")
print("Results saved to outputs/results.json")
