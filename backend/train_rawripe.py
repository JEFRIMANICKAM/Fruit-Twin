"""
Training & Evaluation Pipeline for RawRipe Fruit Maturity Classification Model

Dataset:
- RawRipe (1,630 images: 1,216 train, 414 test across 10 fruits)
- Official train/test splits from sets/train.txt and sets/test.txt

Features:
- Extracts 48-dimensional morphological, color, and texture features
- Trains both a Global Maturity Model and fruit-specific models
- Evaluates on the held-out 414 test images
- Computes Accuracy, Precision, Recall, F1-Score
- Saves trained model to models/rawripe_model.pkl
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from data.rawripe_loader import (
    load_rawripe_records,
    get_rawripe_root,
    SUPPORTED_RAWRIPE_FRUITS
)
from models.fruits360_classifier import extract_image_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "rawripe_model.pkl")

def train_and_evaluate_rawripe():
    root = get_rawripe_root()
    if not root:
        print("[Train RawRipe] Error: RawRipe dataset not found.")
        return None

    print(f"[Train RawRipe] Loading dataset from: {root}")
    splits = load_rawripe_records(root)
    train_records = splits["train"]
    test_records = splits["test"]

    print(f"[Train RawRipe] Train records: {len(train_records)}, Test records: {len(test_records)}")

    # Extract features for training set
    print("[Train RawRipe] Extracting training features...")
    X_train = []
    y_train = []
    fruit_train = []

    for idx, rec in enumerate(train_records):
        try:
            im = Image.open(rec["image_path"])
            feat = extract_image_features(im)
            X_train.append(feat)
            y_train.append(rec["label"])
            fruit_train.append(rec["fruit"])
            if (idx + 1) % 300 == 0:
                print(f"  Processed {idx + 1}/{len(train_records)} train images...")
        except Exception as e:
            pass

    # Extract features for testing set
    print("[Train RawRipe] Extracting test features...")
    X_test = []
    y_test = []
    fruit_test = []

    for idx, rec in enumerate(test_records):
        try:
            im = Image.open(rec["image_path"])
            feat = extract_image_features(im)
            X_test.append(feat)
            y_test.append(rec["label"])
            fruit_test.append(rec["fruit"])
            if (idx + 1) % 150 == 0:
                print(f"  Processed {idx + 1}/{len(test_records)} test images...")
        except Exception as e:
            pass

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"[Train RawRipe] Training matrix: {X_train.shape}, Test matrix: {X_test.shape}")

    # Train Global Maturity Model
    print("[Train RawRipe] Fitting Global Maturity Classifier...")
    clf = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate Global Model on Held-Out Test Set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print("\n================== RAWRIPE GLOBAL EVALUATION REPORT ==================")
    print(f"Test Samples: {len(X_test)}")
    print(f"Overall Accuracy:  {acc * 100.0:.2f}%")
    print(f"Overall Precision: {prec * 100.0:.2f}%")
    print(f"Overall Recall:    {rec * 100.0:.2f}%")
    print(f"Overall F1-Score:  {f1 * 100.0:.2f}%")
    print("======================================================================\n")

    # Evaluate per-fruit performance
    per_fruit_eval = {}
    fruit_models = {}
    
    for fruit in SUPPORTED_RAWRIPE_FRUITS:
        test_mask = [f == fruit for f in fruit_test]
        train_mask = [f == fruit for f in fruit_train]
        
        if sum(train_mask) > 10:
            f_clf = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42)
            f_clf.fit(X_train[train_mask], y_train[train_mask])
            fruit_models[fruit] = f_clf

        if sum(test_mask) > 0:
            f_y_test = y_test[test_mask]
            f_y_pred = y_pred[test_mask]
            f_acc = accuracy_score(f_y_test, f_y_pred)
            f_prec, f_rec, f_f1, _ = precision_recall_fscore_support(f_y_test, f_y_pred, average="weighted", zero_division=0)
            per_fruit_eval[fruit] = {
                "test_count": int(sum(test_mask)),
                "accuracy": round(float(f_acc), 4),
                "precision": round(float(f_prec), 4),
                "recall": round(float(f_rec), 4),
                "f1_score": round(float(f_f1), 4)
            }
            print(f" - {fruit:14s}: Acc = {f_acc*100:.1f}%, F1 = {f_f1*100:.1f}% ({sum(test_mask)} test samples)")

    # Save model checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    payload = {
        "global_model": clf,
        "fruit_models": fruit_models,
        "supported_fruits": SUPPORTED_RAWRIPE_FRUITS,
        "evaluation": {
            "overall_accuracy": float(acc),
            "overall_precision": float(prec),
            "overall_recall": float(rec),
            "overall_f1": float(f1),
            "test_samples": len(X_test),
            "per_fruit": per_fruit_eval
        }
    }

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[Train RawRipe] Successfully saved trained RawRipe model to: {MODEL_OUTPUT_PATH}")
    return payload["evaluation"]

if __name__ == "__main__":
    train_and_evaluate_rawripe()
