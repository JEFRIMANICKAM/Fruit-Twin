"""
Train Machine Learning Models on the Real Nam Dok Mai Si Tong Mango Dataset
Location: D:\\IOT LAB\\GAN FRUIT RIPENESS\\MANGO

Models Created:
  1. models/mango/mango_ripeness_model.pkl (Scikit-Learn Random Forest Classifier for Ripeness Stage)
  2. models/mango/mango_ripening_model.pkl (Scikit-Learn Random Forest Regressor for Days to Ripe & Shelf Life)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from models.base import MANGO_MODELS_DIR
from data.parse_real_dataset import load_real_dataset

os.makedirs(MANGO_MODELS_DIR, exist_ok=True)

def train_ripeness_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("\n[1/2] Training Scikit-Learn Ripeness Stage Classifier on Real Dataset...")
    
    # Feature columns: R, G, B, W, C
    X_train = train_df[["R", "G", "B", "W", "C"]].values
    y_train = train_df["ripeness_stage"].values
    
    X_val = val_df[["R", "G", "B", "W", "C"]].values
    y_val = val_df["ripeness_stage"].values
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    
    print(f"  [OK] Train Accuracy: {train_acc * 100:.1f}%")
    print(f"  [OK] Validation Accuracy: {val_acc * 100:.1f}%")
    
    model_path = os.path.join(MANGO_MODELS_DIR, "mango_ripeness_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_names": ["R", "G", "B", "W", "C"],
            "classes": list(clf.classes_)
        }, f)
    print(f"  [OK] Saved classifier to {model_path}")

def train_ripening_regressor(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("\n[2/2] Training Scikit-Learn Ripening Timeline Regressor on Real Dataset...")
    
    # Map stage to number: Unripe: 1, Semi-Ripe: 2, Ripe: 3, Overripe: 4
    stage_map = {"Unripe": 1, "Semi-Ripe": 2, "Ripe": 3, "Overripe": 4}
    
    X_train = []
    y_train = []
    
    for _, row in train_df.iterrows():
        s_num = stage_map.get(row["ripeness_stage"], 2)
        r, g, b = row["R"], row["G"], row["B"]
        w, c = row["W"], row["C"]
        # Ambient temp from dataset observations (around 28°C)
        temp = 28.0
        X_train.append([s_num, temp, r, g, b, w, c])
        days_ripe = row["days_to_ripe"]
        shelf_life = max(0.5, days_ripe + 4.0)
        y_train.append([days_ripe, shelf_life])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    reg = RandomForestRegressor(n_estimators=50, random_state=42)
    reg.fit(X_train, y_train)
    
    model_path = os.path.join(MANGO_MODELS_DIR, "mango_ripening_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(reg, f)
    print(f"  [OK] Saved regressor to {model_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("FruitTwin Model Training on Real Dataset (Nam Dok Mai Si Tong)")
    print("Dataset Path: D:\\IOT LAB\\GAN FRUIT RIPENESS\\MANGO")
    print("=" * 70)
    
    datasets = load_real_dataset()
    train_ripeness_classifier(datasets["train"], datasets["validate"])
    train_ripening_regressor(datasets["train"], datasets["validate"])
    print("\nTraining Complete! Real models ready in backend/models/mango/")
