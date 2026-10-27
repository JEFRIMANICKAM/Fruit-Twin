"""
Trainer script for extended fruit models:
- Orange (Citrus sinensis)
- Guava (Psidium guajava)
- Pomegranate (Punica granatum)
- Watermelon (Citrullus lanatus)

Generates:
  {fruit}/{fruit}_ripeness_model.pkl
  {fruit}/{fruit}_ripening_model.pkl
  {fruit}/{fruit}_variety_model.pkl
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

EXTENDED_FRUITS_SPECS = {
    "orange": {
        "varieties": [
            {
                "id": "nagpur",
                "name": "Nagpur Orange (நாக்பூர் ஆரஞ்சு)",
                "profiles": [[84.0, 52.0, 14.0], [80.0, 56.0, 16.0], [78.0, 50.0, 12.0]],
                "weight": (160.0, 220.0),
                "cap": (1.6, 2.4)
            },
            {
                "id": "sathgudi",
                "name": "Sathgudi (சாத்துக்குடி)",
                "profiles": [[74.0, 70.0, 20.0], [70.0, 68.0, 22.0], [62.0, 72.0, 24.0]],
                "weight": (140.0, 190.0),
                "cap": (1.5, 2.2)
            },
            {
                "id": "coorg_mandarin",
                "name": "Coorg Mandarin (கூர்க் மாண்டரின்)",
                "profiles": [[82.0, 48.0, 16.0], [86.0, 46.0, 14.0]],
                "weight": (120.0, 170.0),
                "cap": (1.4, 2.1)
            },
            {
                "id": "kinnow",
                "name": "Kinnow (கின்னோ)",
                "profiles": [[85.0, 55.0, 18.0], [80.0, 58.0, 16.0]],
                "weight": (150.0, 210.0),
                "cap": (1.7, 2.5)
            }
        ],
        "stages": {
            "Unripe": {"r": (28.0, 4.0), "g": (66.0, 4.0), "b": (20.0, 3.0), "days": 7.0, "shelf": 12.0},
            "Semi-Ripe": {"r": (64.0, 4.0), "g": (68.0, 4.0), "b": (22.0, 3.0), "days": 3.5, "shelf": 8.0},
            "Ripe": {"r": (82.0, 4.0), "g": (54.0, 4.0), "b": (15.0, 3.0), "days": 0.5, "shelf": 4.5},
            "Overripe": {"r": (68.0, 4.0), "g": (42.0, 4.0), "b": (16.0, 3.0), "days": 0.0, "shelf": 1.5}
        },
        "default_weight": 180.0,
        "default_cap": 2.0
    },
    "guava": {
        "varieties": [
            {
                "id": "allahabad_safeda",
                "name": "Allahabad Safeda (அலகாபாத் சபேடா)",
                "profiles": [[72.0, 76.0, 36.0], [68.0, 74.0, 32.0], [76.0, 78.0, 40.0]],
                "weight": (160.0, 240.0),
                "cap": (1.8, 2.6)
            },
            {
                "id": "lucknow_49",
                "name": "Lucknow 49 / Sardar (சர்தார் கொய்யா)",
                "profiles": [[62.0, 72.0, 28.0], [58.0, 70.0, 26.0], [66.0, 74.0, 30.0]],
                "weight": (180.0, 260.0),
                "cap": (1.9, 2.7)
            },
            {
                "id": "taiwan_pink",
                "name": "Taiwan Pink (தைவான் பிங்க் கொய்யா)",
                "profiles": [[70.0, 62.0, 32.0], [74.0, 58.0, 34.0], [72.0, 66.0, 30.0]],
                "weight": (220.0, 320.0),
                "cap": (2.1, 2.9)
            },
            {
                "id": "arka_kiran",
                "name": "Arka Kiran (அர்கா கிரண்)",
                "profiles": [[72.0, 54.0, 32.0], [76.0, 50.0, 30.0]],
                "weight": (170.0, 230.0),
                "cap": (1.9, 2.6)
            }
        ],
        "stages": {
            "Unripe": {"r": (34.0, 4.0), "g": (68.0, 4.0), "b": (24.0, 3.0), "days": 6.0, "shelf": 10.0},
            "Semi-Ripe": {"r": (60.0, 4.0), "g": (72.0, 4.0), "b": (28.0, 3.0), "days": 2.5, "shelf": 6.0},
            "Ripe": {"r": (74.0, 4.0), "g": (74.0, 4.0), "b": (34.0, 3.0), "days": 0.5, "shelf": 3.0},
            "Overripe": {"r": (64.0, 4.0), "g": (58.0, 4.0), "b": (30.0, 3.0), "days": 0.0, "shelf": 1.0}
        },
        "default_weight": 200.0,
        "default_cap": 2.2
    },
    "pomegranate": {
        "varieties": [
            {
                "id": "bhagwa",
                "name": "Bhagwa (பக்வா மாதுளை)",
                "profiles": [[82.0, 24.0, 22.0], [85.0, 22.0, 20.0], [78.0, 28.0, 24.0]],
                "weight": (240.0, 340.0),
                "cap": (1.8, 2.6)
            },
            {
                "id": "ganesh",
                "name": "Ganesh (கணேஷ் மாதுளை)",
                "profiles": [[72.0, 42.0, 26.0], [68.0, 46.0, 28.0], [74.0, 38.0, 24.0]],
                "weight": (220.0, 300.0),
                "cap": (1.7, 2.4)
            },
            {
                "id": "arakta",
                "name": "Arakta (அரக்டா)",
                "profiles": [[80.0, 28.0, 24.0], [83.0, 26.0, 22.0]],
                "weight": (230.0, 320.0),
                "cap": (1.8, 2.5)
            },
            {
                "id": "ruby",
                "name": "Ruby (ரூபி)",
                "profiles": [[86.0, 20.0, 20.0], [88.0, 18.0, 18.0]],
                "weight": (250.0, 350.0),
                "cap": (1.9, 2.7)
            }
        ],
        "stages": {
            "Unripe": {"r": (52.0, 4.0), "g": (62.0, 4.0), "b": (26.0, 3.0), "days": 10.0, "shelf": 18.0},
            "Semi-Ripe": {"r": (68.0, 4.0), "g": (44.0, 4.0), "b": (26.0, 3.0), "days": 5.0, "shelf": 12.0},
            "Ripe": {"r": (82.0, 4.0), "g": (24.0, 4.0), "b": (22.0, 3.0), "days": 0.5, "shelf": 7.0},
            "Overripe": {"r": (58.0, 4.0), "g": (22.0, 4.0), "b": (20.0, 3.0), "days": 0.0, "shelf": 2.0}
        },
        "default_weight": 280.0,
        "default_cap": 2.1
    },
    "watermelon": {
        "varieties": [
            {
                "id": "sugar_baby",
                "name": "Sugar Baby (சுகர் பேபி)",
                "profiles": [[32.0, 52.0, 28.0], [28.0, 48.0, 26.0]],
                "weight": (2800.0, 4200.0),
                "cap": (4.8, 7.2)
            },
            {
                "id": "kiran",
                "name": "Kiran (கிரண் தர்பூசணி)",
                "profiles": [[46.0, 64.0, 30.0], [42.0, 60.0, 28.0]],
                "weight": (3200.0, 5200.0),
                "cap": (5.2, 8.0)
            },
            {
                "id": "black_diamond",
                "name": "Black Diamond (பிளாக் டைமண்ட்)",
                "profiles": [[26.0, 44.0, 24.0], [24.0, 42.0, 22.0]],
                "weight": (4000.0, 6500.0),
                "cap": (6.0, 9.0)
            },
            {
                "id": "icebox",
                "name": "Icebox (ஐஸ்பாக்ஸ்)",
                "profiles": [[38.0, 58.0, 30.0], [35.0, 55.0, 28.0]],
                "weight": (1800.0, 3000.0),
                "cap": (3.8, 5.5)
            }
        ],
        "stages": {
            "Unripe": {"r": (34.0, 3.0), "g": (56.0, 4.0), "b": (30.0, 3.0), "days": 8.0, "shelf": 15.0},
            "Semi-Ripe": {"r": (42.0, 3.0), "g": (60.0, 4.0), "b": (28.0, 3.0), "days": 4.0, "shelf": 10.0},
            "Ripe": {"r": (50.0, 4.0), "g": (64.0, 4.0), "b": (26.0, 3.0), "days": 0.5, "shelf": 6.0},
            "Overripe": {"r": (44.0, 4.0), "g": (50.0, 4.0), "b": (24.0, 3.0), "days": 0.0, "shelf": 2.0}
        },
        "default_weight": 3500.0,
        "default_cap": 6.0
    }
}

def train_and_save_fruit(fruit_name, spec):
    fruit_dir = os.path.join(MODELS_DIR, fruit_name)
    os.makedirs(fruit_dir, exist_ok=True)
    print(f"\n---> Training AI Models for {fruit_name.upper()} ({fruit_dir})")

    # 1. Variety Model
    X_var, y_var = [], []
    for var in spec["varieties"]:
        v_name = var["name"]
        w_min, w_max = var["weight"]
        c_min, c_max = var["cap"]
        for prof in var["profiles"]:
            for _ in range(60):
                r = float(np.clip(np.random.normal(prof[0], 3.0), 0, 100))
                g = float(np.clip(np.random.normal(prof[1], 3.0), 0, 100))
                b = float(np.clip(np.random.normal(prof[2], 2.5), 0, 100))
                w = float(np.random.uniform(w_min, w_max))
                c = float(np.random.uniform(c_min, c_max))
                X_var.append([r, g, b, w, c])
                y_var.append(v_name)

    clf_var = RandomForestClassifier(n_estimators=45, random_state=42)
    clf_var.fit(X_var, y_var)
    var_path = os.path.join(fruit_dir, f"{fruit_name}_variety_model.pkl")
    with open(var_path, "wb") as f:
        pickle.dump({"model": clf_var, "classes": list(clf_var.classes_)}, f)
    print(f"  [OK] Saved Variety Model: {var_path}")

    # 2. Ripeness Model
    X_rip, y_rip = [], []
    for stage, s_spec in spec["stages"].items():
        r_mean, r_std = s_spec["r"]
        g_mean, g_std = s_spec["g"]
        b_mean, b_std = s_spec["b"]
        for _ in range(120):
            r = float(np.clip(np.random.normal(r_mean, r_std), 0, 100))
            g = float(np.clip(np.random.normal(g_mean, g_std), 0, 100))
            b = float(np.clip(np.random.normal(b_mean, b_std), 0, 100))
            w = float(np.random.normal(spec["default_weight"], spec["default_weight"] * 0.15))
            c = float(np.random.normal(spec["default_cap"], spec["default_cap"] * 0.12))
            X_rip.append([r, g, b, w, c])
            y_rip.append(stage)

    clf_rip = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_rip.fit(X_rip, y_rip)
    rip_path = os.path.join(fruit_dir, f"{fruit_name}_ripeness_model.pkl")
    with open(rip_path, "wb") as f:
        pickle.dump({"model": clf_rip, "classes": list(clf_rip.classes_)}, f)
    print(f"  [OK] Saved Ripeness Model: {rip_path}")

    # 3. Ripening Regressor (Days to Ripe & Shelf Life)
    stage_num_map = {"Unripe": 1, "Semi-Ripe": 2, "Ripe": 3, "Overripe": 4}
    X_reg, y_reg = [], []
    for temp in [18.0, 22.0, 25.0, 28.0, 32.0, 35.0]:
        temp_factor = max(0.4, 1.0 - (temp - 26.0) * 0.07)
        for stage, s_spec in spec["stages"].items():
            base_d = s_spec["days"]
            base_s = s_spec["shelf"]
            d_pred = round(max(0.0, base_d * temp_factor), 1)
            s_pred = round(max(0.5, base_s * temp_factor), 1)
            r_mean, _ = s_spec["r"]
            g_mean, _ = s_spec["g"]
            b_mean, _ = s_spec["b"]
            for _ in range(30):
                r = r_mean + np.random.normal(0, 3)
                g = g_mean + np.random.normal(0, 3)
                b = b_mean + np.random.normal(0, 2)
                w = spec["default_weight"] + np.random.normal(0, 20)
                c = spec["default_cap"] + np.random.normal(0, 0.2)
                X_reg.append([stage_num_map[stage], temp, r, g, b, w, c])
                y_reg.append([d_pred, s_pred])

    reg = RandomForestRegressor(n_estimators=40, random_state=42)
    reg.fit(X_reg, y_reg)
    reg_path = os.path.join(fruit_dir, f"{fruit_name}_ripening_model.pkl")
    with open(reg_path, "wb") as f:
        pickle.dump(reg, f)
    print(f"  [OK] Saved Ripening Regressor: {reg_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("FruitTwin: Training Extended Multi-Fruit AI Models")
    print("Adding: Orange, Guava, Pomegranate, Watermelon")
    print("=" * 60)
    for fruit, spec in EXTENDED_FRUITS_SPECS.items():
        train_and_save_fruit(fruit, spec)
    print("\n[COMPLETE] All 4 Extended Fruit Models Created Successfully!")
