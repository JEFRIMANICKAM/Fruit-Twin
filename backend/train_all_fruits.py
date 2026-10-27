"""
Comprehensive Multi-Fruit & Variety AI Model Trainer for FruitTwin
Supports: Mango, Banana, Papaya, and Apple.
Specifically calibrates:
  1. Red Banana (Sevazhai / செவ்வாழை) vs Poovan vs Nendran vs Rasthali vs Grand Naine
  2. Ripeness Stage Classification (Unripe, Semi-Ripe, Ripe, Overripe)
  3. Ripening Shelf-Life Regressor (Days to Ripe & Remaining Shelf Life)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

VARIETY_SPECS = {
    "banana": {
        "varieties": [
            {
                "id": "sevazhai",
                "name": "Red Banana (Sevazhai)",
                "tamilName": "செவ்வாழை",
                # Rich reddish-maroon tones across stages and lighting
                "rgb_profiles": [
                    [76.0, 26.0, 22.0],  # Peak ripe dark maroon
                    [68.0, 32.0, 25.0],  # Semi-ripe crimson
                    [62.0, 35.0, 28.0],  # Early red-bronze
                    [72.0, 22.0, 20.0],  # Deep burgundy
                    [58.0, 38.0, 32.0],  # Ambient/indoor lighting
                    [66.3, 59.8, 59.1],  # Attenuated/camera-balanced red
                    [78.0, 25.0, 22.0]   # High red saturation
                ],
                "weight_range": (160.0, 220.0),
                "cap_range": (1.4, 2.4)
            },
            {
                "id": "poovan",
                "name": "Poovan",
                "tamilName": "பூவன் வாழைப்பழம்",
                # Bright golden yellow
                "rgb_profiles": [
                    [75.0, 70.0, 15.0],
                    [78.0, 72.0, 18.0],
                    [68.0, 64.0, 14.0],
                    [42.0, 65.0, 16.0]   # Unripe greenish
                ],
                "weight_range": (100.0, 135.0),
                "cap_range": (1.1, 1.9)
            },
            {
                "id": "nendran",
                "name": "Nendran",
                "tamilName": "நேந்திரன் வாழைப்பழம்",
                # Deep yellow-orange plantain
                "rgb_profiles": [
                    [78.0, 58.0, 12.0],
                    [80.0, 60.0, 14.0],
                    [72.0, 52.0, 10.0],
                    [45.0, 66.0, 14.0]   # Unripe plantain
                ],
                "weight_range": (180.0, 240.0),
                "cap_range": (1.2, 2.0)
            },
            {
                "id": "rasthali",
                "name": "Rasthali",
                "tamilName": "ரஸ்தாளி வாழைப்பழம்",
                # Creamy pale yellow
                "rgb_profiles": [
                    [78.0, 76.0, 24.0],
                    [82.0, 78.0, 26.0],
                    [72.0, 70.0, 20.0]
                ],
                "weight_range": (120.0, 150.0),
                "cap_range": (1.1, 1.8)
            },
            {
                "id": "grand_naine",
                "name": "Grand Naine (G9)",
                "tamilName": "கிராண்ட் நைன்",
                # Greenish-yellow commercial banana
                "rgb_profiles": [
                    [48.0, 68.0, 18.0],
                    [65.0, 68.0, 20.0],
                    [38.0, 72.0, 16.0]
                ],
                "weight_range": (150.0, 190.0),
                "cap_range": (1.2, 1.9)
            }
        ]
    },
    "mango": {
        "varieties": [
            {
                "id": "sindhoora",
                "name": "Sindhoora (Apple Mango)",
                "tamilName": "சிந்தூரா / ஆப்பிள் மாம்பழம்",
                "rgb_profiles": [
                    [76.0, 32.0, 24.0],  # Deep red blush
                    [72.0, 42.0, 26.0],  # Crimson-orange blush
                    [80.0, 36.0, 22.0],  # Vermillion red blush
                    [68.0, 48.0, 28.0]   # Red cheek with yellow shoulder
                ],
                "weight_range": (250.0, 350.0),
                "cap_range": (1.8, 2.5)
            },
            {
                "id": "alphonso",
                "name": "Alphonso",
                "tamilName": "அல்போன்சா",
                "rgb_profiles": [[78.0, 60.0, 12.0], [82.0, 64.0, 10.0], [52.0, 68.0, 18.0]],
                "weight_range": (240.0, 320.0),
                "cap_range": (1.6, 2.4)
            },
            {
                "id": "neelum",
                "name": "Neelum",
                "tamilName": "நீலம்",
                "rgb_profiles": [[74.0, 66.0, 15.0], [68.0, 62.0, 14.0], [48.0, 70.0, 18.0]],
                "weight_range": (200.0, 280.0),
                "cap_range": (1.5, 2.2)
            },
            {
                "id": "banganapalli",
                "name": "Banganapalli",
                "tamilName": "பங்கனபள்ளி",
                "rgb_profiles": [[76.0, 68.0, 18.0], [70.0, 64.0, 16.0]],
                "weight_range": (300.0, 420.0),
                "cap_range": (1.8, 2.6)
            },
            {
                "id": "malgova",
                "name": "Malgova",
                "tamilName": "மல்கோவா",
                "rgb_profiles": [[72.0, 58.0, 20.0], [65.0, 60.0, 22.0]],
                "weight_range": (350.0, 500.0),
                "cap_range": (2.0, 2.8)
            },
            {
                "id": "totapuri",
                "name": "Totapuri (Kilimookku)",
                "tamilName": "கிளிமூக்கு மாங்காய்",
                "rgb_profiles": [[55.0, 68.0, 22.0], [60.0, 64.0, 20.0]],
                "weight_range": (280.0, 380.0),
                "cap_range": (1.6, 2.4)
            },
            {
                "id": "kesar",
                "name": "Kesar",
                "tamilName": "கேசர்",
                "rgb_profiles": [[84.0, 54.0, 10.0], [78.0, 56.0, 12.0]],
                "weight_range": (220.0, 300.0),
                "cap_range": (1.6, 2.3)
            }
        ]
    },
    "papaya": {
        "varieties": [
            {
                "id": "red_lady",
                "name": "Red Lady",
                "tamilName": "ரெட் லேடி பப்பாளி",
                "rgb_profiles": [[84.0, 46.0, 12.0], [78.0, 48.0, 14.0], [45.0, 68.0, 18.0]],
                "weight_range": (1100.0, 1600.0),
                "cap_range": (3.2, 4.4)
            },
            {
                "id": "co7",
                "name": "Coimbatore Co-7",
                "tamilName": "கோ-7 பப்பாளி",
                "rgb_profiles": [[76.0, 58.0, 15.0], [72.0, 56.0, 16.0]],
                "weight_range": (1000.0, 1400.0),
                "cap_range": (3.0, 4.2)
            },
            {
                "id": "washington",
                "name": "Washington",
                "tamilName": "வாஷிங்டன் பப்பாளி",
                "rgb_profiles": [[70.0, 52.0, 20.0], [66.0, 54.0, 18.0]],
                "weight_range": (900.0, 1300.0),
                "cap_range": (2.8, 4.0)
            }
        ]
    },
    "apple": {
        "varieties": [
            {
                "id": "royal_delicious",
                "name": "Royal Delicious",
                "tamilName": "ரொயல் டெலிஷியஸ்",
                "rgb_profiles": [[82.0, 20.0, 18.0], [76.0, 22.0, 20.0], [55.0, 32.0, 24.0]],
                "weight_range": (170.0, 220.0),
                "cap_range": (1.6, 2.1)
            },
            {
                "id": "ambri",
                "name": "Kashmiri Ambri",
                "tamilName": "காஷ்மீரி அம்ரி",
                "rgb_profiles": [[72.0, 28.0, 22.0], [68.0, 35.0, 25.0]],
                "weight_range": (150.0, 195.0),
                "cap_range": (1.5, 2.0)
            },
            {
                "id": "granny_smith",
                "name": "Granny Smith",
                "tamilName": "கிரானி ஸ்மித்",
                "rgb_profiles": [[42.0, 72.0, 18.0], [38.0, 76.0, 16.0]],
                "weight_range": (160.0, 210.0),
                "cap_range": (1.5, 2.0)
            }
        ]
    }
}

def train_fruit_variety_model(fruit_name: str, spec: dict):
    print(f"\n[+] Training Variety Detector for {fruit_name.upper()}...")
    fruit_dir = os.path.join(MODELS_DIR, fruit_name)
    os.makedirs(fruit_dir, exist_ok=True)

    X_var, y_var = [], []
    for v_info in spec["varieties"]:
        v_name = f"{v_info['name']} ({v_info['tamilName']})"
        w_low, w_high = v_info["weight_range"]
        c_low, c_high = v_info["cap_range"]

        for profile in v_info["rgb_profiles"]:
            # Generate 80 variations around each profile
            for _ in range(80):
                r = np.clip(profile[0] + np.random.normal(0, 3.5), 5, 98)
                g = np.clip(profile[1] + np.random.normal(0, 3.5), 5, 98)
                b = np.clip(profile[2] + np.random.normal(0, 3.0), 5, 98)
                w = np.clip(np.random.uniform(w_low, w_high), 40, 2000)
                c = np.clip(np.random.uniform(c_low, c_high), 0.5, 6.0)
                X_var.append([r, g, b, w, c])
                y_var.append(v_name)

    clf_var = RandomForestClassifier(n_estimators=60, random_state=42)
    clf_var.fit(X_var, y_var)

    var_path = os.path.join(fruit_dir, f"{fruit_name}_variety_model.pkl")
    with open(var_path, "wb") as f:
        pickle.dump({
            "model": clf_var,
            "feature_names": ["R", "G", "B", "W", "C"],
            "classes": list(clf_var.classes_)
        }, f)
    print(f"  [OK] Saved Variety Classifier: {var_path} ({len(X_var)} samples, {len(clf_var.classes_)} classes)")

def train_banana_ripeness_and_timeline():
    print("\n[+] Training Banana Ripeness & Timeline ML Regressors (including Red Banana)...")
    b_dir = os.path.join(MODELS_DIR, "banana")
    os.makedirs(b_dir, exist_ok=True)

    # 1. Ripeness Stage Classifier for Banana
    X_rip, y_rip = [], []

    # Unripe Banana: (Green Cavendish, Green Poovan, or Dark Bronze/Greenish-Red Red Banana)
    for _ in range(120):
        # Yellow cultivar unripe: high green
        r = np.random.normal(35.0, 4.0)
        g = np.random.normal(68.0, 5.0)
        b = np.random.normal(16.0, 3.0)
        X_rip.append([r, g, b, 140.0, 1.2])
        y_rip.append("Unripe")

    for _ in range(80):
        # Red Banana unripe: bronze-green/dark reddish
        r = np.random.normal(46.0, 3.5)
        g = np.random.normal(38.0, 3.0)
        b = np.random.normal(26.0, 2.5)
        X_rip.append([r, g, b, 185.0, 1.3])
        y_rip.append("Unripe")

    # Semi-Ripe Banana:
    for _ in range(120):
        # Yellow cultivars semi-ripe
        r = np.random.normal(58.0, 4.0)
        g = np.random.normal(58.0, 4.0)
        b = np.random.normal(15.0, 3.0)
        X_rip.append([r, g, b, 135.0, 1.45])
        y_rip.append("Semi-Ripe")

    for _ in range(100):
        # Red Banana semi-ripe: crimson red
        r = np.random.normal(65.0, 4.0)
        g = np.random.normal(34.0, 3.5)
        b = np.random.normal(25.0, 3.0)
        X_rip.append([r, g, b, 180.0, 1.6])
        y_rip.append("Semi-Ripe")

    # Ripe Banana:
    for _ in range(120):
        # Yellow cultivars ripe: golden yellow
        r = np.random.normal(76.0, 3.5)
        g = np.random.normal(70.0, 3.5)
        b = np.random.normal(14.0, 2.5)
        X_rip.append([r, g, b, 130.0, 1.8])
        y_rip.append("Ripe")

    for _ in range(120):
        # Red Banana ripe: deep maroon/burgundy
        r = np.random.normal(74.0, 4.0)
        g = np.random.normal(26.0, 3.5)
        b = np.random.normal(22.0, 3.0)
        X_rip.append([r, g, b, 175.0, 1.95])
        y_rip.append("Ripe")

    # Overripe Banana:
    for _ in range(100):
        # Speckled/darkened
        r = np.random.normal(56.0, 5.0)
        g = np.random.normal(44.0, 4.5)
        b = np.random.normal(18.0, 3.0)
        X_rip.append([r, g, b, 120.0, 2.2])
        y_rip.append("Overripe")

    for _ in range(80):
        # Red Banana overripe
        r = np.random.normal(58.0, 4.0)
        g = np.random.normal(20.0, 3.0)
        b = np.random.normal(17.0, 2.5)
        X_rip.append([r, g, b, 165.0, 2.3])
        y_rip.append("Overripe")

    clf_rip = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_rip.fit(X_rip, y_rip)

    rip_path = os.path.join(b_dir, "banana_ripeness_model.pkl")
    with open(rip_path, "wb") as f:
        pickle.dump({
            "model": clf_rip,
            "feature_names": ["R", "G", "B", "W", "C"],
            "classes": list(clf_rip.classes_)
        }, f)
    print(f"  [OK] Saved Banana Ripeness Classifier: {rip_path}")

    # 2. Ripening Timeline Regressor for Banana
    stage_map = {"Unripe": 1, "Semi-Ripe": 2, "Ripe": 3, "Overripe": 4}
    X_reg, y_reg = [], []
    for temp in [22.0, 25.0, 28.0, 30.0, 32.0, 35.0]:
        for stage, s_num in stage_map.items():
            base_days = {"Unripe": 6.5, "Semi-Ripe": 3.2, "Ripe": 0.5, "Overripe": 0.0}[stage]
            # Higher temp accelerates ripening
            temp_factor = max(0.4, 1.0 - (temp - 28.0) * 0.08)
            days_to_ripe = round(max(0.0, base_days * temp_factor), 1)
            shelf_life = round(days_to_ripe + (4.5 if stage in ["Unripe", "Semi-Ripe"] else 2.5), 1)

            for _ in range(25):
                r = 70.0 + np.random.normal(0, 5)
                g = 50.0 + np.random.normal(0, 5)
                b = 20.0 + np.random.normal(0, 3)
                w = 160.0 + np.random.normal(0, 20)
                c = 1.6 + np.random.normal(0, 0.2)
                X_reg.append([s_num, temp, r, g, b, w, c])
                y_reg.append([days_to_ripe, shelf_life])

    reg = RandomForestRegressor(n_estimators=40, random_state=42)
    reg.fit(X_reg, y_reg)

    reg_path = os.path.join(b_dir, "banana_ripening_model.pkl")
    with open(reg_path, "wb") as f:
        pickle.dump(reg, f)
    print(f"  [OK] Saved Banana Ripening Regressor: {reg_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("FruitTwin Multi-Fruit AI Model & Variety Classifier Training")
    print("Calibrated for: Red Banana (Sevazhai), Poovan, Alphonso, Neelum, etc.")
    print("=" * 70)

    for fruit, spec in VARIETY_SPECS.items():
        train_fruit_variety_model(fruit, spec)

    train_banana_ripeness_and_timeline()
    print("\n[SUCCESS] All Multi-Fruit & Variety Models Trained & Saved Successfully!")
