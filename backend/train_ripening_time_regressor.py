"""
Ripening-Time Regression Training & Evaluation Module for FruitTwin

Dataset:
- Real Kasetsart University Nam Dok Mai Si Tong Mango Dataset
- 100 train samples (Train_Data.csv), 100 validation samples (Validate_Data.csv)
- Predicts remaining days to peak ripeness (Storage Day 7) from physical, optical, and environmental features.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data.parse_real_dataset import load_real_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "mango_ripening_regressor.pkl")

def train_and_evaluate_ripening_regressor():
    try:
        datasets = load_real_dataset()
    except Exception as e:
        print(f"[Ripening Regressor] Dataset load error: {e}")
        return None

    train_df = datasets["train"]
    val_df = datasets["validate"]

    # Features: W (Weight), C (Capacitance), R, G, B, plus temperature proxy
    feature_cols = ["W", "C", "R", "G", "B"]

    X_train = train_df[feature_cols].values
    y_train = train_df["days_to_ripe"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["days_to_ripe"].values

    reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\n================== RIPENING-TIME REGRESSION REPORT ==================")
    print("Dataset: Kasetsart University Nam Dok Mai Si Tong Mango (Real Storage Telemetry)")
    print(f"Validation Samples: {len(y_val)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"R² Score:                 {r2:.3f}")
    print("Conditions: Ambient 25-28°C storage, monitored on Days 1, 3, 5, 7, 9")
    print("=====================================================================\n")

    eval_info = {
        "supported_fruit": "mango",
        "dataset": "Kasetsart University Nam Dok Mai Si Tong Mango",
        "validation_samples": len(y_val),
        "mae_days": round(float(mae), 2),
        "r2_score": round(float(r2), 3),
        "evaluation_conditions": "Monitored at 25-28°C ambient temperature over 9-day storage cycle",
        "feature_names": feature_cols
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump({"model": reg, "evaluation": eval_info}, f)

    print(f"[Ripening Regressor] Saved regressor model to: {MODEL_OUTPUT_PATH}")
    return eval_info

if __name__ == "__main__":
    train_and_evaluate_ripening_regressor()
