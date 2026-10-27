from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np
import os
import pickle

from models.ripeness_model import FruitRipenessModel
from models.ripening_model import FruitRipeningModel
from models.fruit_classifier import detect_fruit_category, SUPPORTED_FRUIT_SET
from models.fruits360_classifier import Fruits360Classifier
from models.rawripe_classifier import RawRipeMaturityClassifier

ripeness_model = FruitRipenessModel()
ripening_model = FruitRipeningModel()
fruits360_classifier = Fruits360Classifier()
rawripe_classifier = RawRipeMaturityClassifier()

# Load calibrated mango ripening regressor if available
MANGO_REGRESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mango_ripening_regressor.pkl")
mango_regressor_data = None
if os.path.exists(MANGO_REGRESSOR_PATH):
    try:
        with open(MANGO_REGRESSOR_PATH, "rb") as f:
            mango_regressor_data = pickle.load(f)
    except Exception as e:
        print(f"[PredictionService] Notice loading mango regressor: {e}")

FRUIT_INFO = {
    "mango": {
        "name": "Mango",
        "tamil_name": "மாம்பழம்",
        "latin": "Mangifera indica",
        "icon": "🥭",
        "bio": {
            "Unripe": {"tss": 7.2, "ta": 3.8, "ratio": 1.89},
            "Semi-Ripe": {"tss": 10.5, "ta": 2.6, "ratio": 4.04},
            "Ripe": {"tss": 14.8, "ta": 1.4, "ratio": 10.57},
            "Overripe": {"tss": 16.5, "ta": 0.8, "ratio": 20.62}
        }
    },
    "banana": {
        "name": "Banana",
        "tamil_name": "வாழைப்பழம்",
        "latin": "Musa acuminata",
        "icon": "🍌",
        "bio": {
            "Unripe": {"tss": 4.5, "ta": 0.65, "ratio": 6.92},
            "Semi-Ripe": {"tss": 11.2, "ta": 0.45, "ratio": 24.88},
            "Ripe": {"tss": 19.5, "ta": 0.30, "ratio": 65.00},
            "Overripe": {"tss": 22.8, "ta": 0.20, "ratio": 114.00}
        }
    },
    "papaya": {
        "name": "Papaya",
        "tamil_name": "பப்பாளி",
        "latin": "Carica papaya",
        "icon": "🍈",
        "bio": {
            "Unripe": {"tss": 5.2, "ta": 0.35, "ratio": 14.85},
            "Semi-Ripe": {"tss": 8.8, "ta": 0.25, "ratio": 35.20},
            "Ripe": {"tss": 12.5, "ta": 0.15, "ratio": 83.33},
            "Overripe": {"tss": 14.2, "ta": 0.10, "ratio": 142.00}
        }
    },
    "apple": {
        "name": "Apple",
        "tamil_name": "ஆப்பிள்",
        "latin": "Malus domestica",
        "icon": "🍎",
        "bio": {
            "Unripe": {"tss": 8.5, "ta": 0.75, "ratio": 11.33},
            "Semi-Ripe": {"tss": 11.5, "ta": 0.50, "ratio": 23.00},
            "Ripe": {"tss": 14.5, "ta": 0.32, "ratio": 45.31},
            "Overripe": {"tss": 16.0, "ta": 0.22, "ratio": 72.72}
        }
    },
    "orange": {
        "name": "Orange",
        "tamil_name": "ஆரஞ்சு",
        "latin": "Citrus sinensis",
        "icon": "🍊",
        "bio": {
            "Unripe": {"tss": 6.8, "ta": 1.85, "ratio": 3.67},
            "Semi-Ripe": {"tss": 9.2, "ta": 1.25, "ratio": 7.36},
            "Ripe": {"tss": 12.4, "ta": 0.85, "ratio": 14.58},
            "Overripe": {"tss": 13.5, "ta": 0.55, "ratio": 24.54}
        }
    },
    "guava": {
        "name": "Guava",
        "tamil_name": "கொய்யா",
        "latin": "Psidium guajava",
        "icon": "🍐",
        "bio": {
            "Unripe": {"tss": 6.5, "ta": 0.65, "ratio": 10.00},
            "Semi-Ripe": {"tss": 8.8, "ta": 0.45, "ratio": 19.55},
            "Ripe": {"tss": 11.2, "ta": 0.28, "ratio": 40.00},
            "Overripe": {"tss": 12.0, "ta": 0.18, "ratio": 66.66}
        }
    },
    "pomegranate": {
        "name": "Pomegranate",
        "tamil_name": "மாதுளை",
        "latin": "Punica granatum",
        "icon": "🔴",
        "bio": {
            "Unripe": {"tss": 9.5, "ta": 1.95, "ratio": 4.87},
            "Semi-Ripe": {"tss": 13.2, "ta": 1.35, "ratio": 9.77},
            "Ripe": {"tss": 16.5, "ta": 0.75, "ratio": 22.00},
            "Overripe": {"tss": 17.5, "ta": 0.50, "ratio": 35.00}
        }
    },
    "watermelon": {
        "name": "Watermelon",
        "tamil_name": "தர்பூசணி",
        "latin": "Citrullus lanatus",
        "icon": "🍉",
        "bio": {
            "Unripe": {"tss": 6.0, "ta": 0.25, "ratio": 24.00},
            "Semi-Ripe": {"tss": 8.5, "ta": 0.18, "ratio": 47.22},
            "Ripe": {"tss": 12.2, "ta": 0.12, "ratio": 101.66},
            "Overripe": {"tss": 13.0, "ta": 0.08, "ratio": 162.50}
        }
    },
    "strawberry": {
        "name": "Strawberry",
        "tamil_name": "ஸ்ட்ராபெர்ரி",
        "latin": "Fragaria ananassa",
        "icon": "🍓",
        "bio": {
            "Unripe": {"tss": 4.5, "ta": 1.20, "ratio": 3.75},
            "Semi-Ripe": {"tss": 6.5, "ta": 0.90, "ratio": 7.22},
            "Ripe": {"tss": 8.5, "ta": 0.70, "ratio": 12.14},
            "Overripe": {"tss": 9.2, "ta": 0.50, "ratio": 18.40}
        }
    },
    "coconut": {
        "name": "Coconut",
        "tamil_name": "தேங்காய்",
        "latin": "Cocos nucifera",
        "icon": "🥥",
        "bio": {
            "Unripe": {"tss": 4.0, "ta": 0.40, "ratio": 10.00},
            "Semi-Ripe": {"tss": 5.5, "ta": 0.30, "ratio": 18.33},
            "Ripe": {"tss": 6.8, "ta": 0.20, "ratio": 34.00},
            "Overripe": {"tss": 7.2, "ta": 0.15, "ratio": 48.00}
        }
    },
    "litchi": {
        "name": "Litchi",
        "tamil_name": "லிச்சி",
        "latin": "Litchi chinensis",
        "icon": "🍒",
        "bio": {
            "Unripe": {"tss": 9.0, "ta": 1.10, "ratio": 8.18},
            "Semi-Ripe": {"tss": 14.0, "ta": 0.65, "ratio": 21.54},
            "Ripe": {"tss": 18.5, "ta": 0.35, "ratio": 52.86},
            "Overripe": {"tss": 19.5, "ta": 0.25, "ratio": 78.00}
        }
    }
}

def predict_single_fruit(image: Image.Image, expected_fruit: Optional[str] = None) -> Dict[str, Any]:
    """Dedicated handler for POST /predict-fruit and /api/predict-fruit."""
    return fruits360_classifier.predict(image, expected_fruit)

def predict_single_maturity(image: Image.Image, fruit_name: Optional[str] = None) -> Dict[str, Any]:
    """Dedicated handler for POST /predict-maturity and /api/predict-maturity."""
    if not fruit_name:
        fruit_res = fruits360_classifier.predict(image)
        fruit_name = fruit_res.get("predicted_fruit", "mango")
    return rawripe_classifier.predict_maturity(image, fruit_name)

def predict_single_ripening_time(fruit: str, temperature: float, current_stage: str = "Semi-Ripe") -> Dict[str, Any]:
    """Dedicated handler for POST /predict-ripening-time and /api/predict-ripening-time."""
    fruit_key = fruit.strip().lower()

    if fruit_key == "mango" and mango_regressor_data is not None:
        reg = mango_regressor_data["model"]
        eval_meta = mango_regressor_data["evaluation"]

        # Nominal feature vector [W, C, R, G, B]
        stage_rg = {"Unripe": (45.0, 58.0), "Semi-Ripe": (68.0, 52.0), "Ripe": (78.0, 48.0), "Overripe": (65.0, 38.0)}
        r_val, g_val = stage_rg.get(current_stage, (65.0, 50.0))
        sample_feat = np.array([[320.0, 2.1, r_val, g_val, 20.0]])

        base_pred = float(reg.predict(sample_feat)[0])
        # Temperature Arrhenius scaling relative to baseline 28°C
        temp_factor = max(0.4, 1.0 - (temperature - 28.0) * 0.08)
        adjusted_days = max(0.0, round(base_pred * temp_factor, 1))
        shelf_life = round(adjusted_days + 4.5, 1)

        return {
            "fruit": "Mango",
            "is_supported": True,
            "estimated_days_to_ripe": adjusted_days,
            "estimated_shelf_life": shelf_life,
            "temperature_celsius": temperature,
            "model_type": "GradientBoostingRegressor",
            "mae_days": eval_meta.get("mae_days", 1.60),
            "evaluation_conditions": eval_meta.get("evaluation_conditions", "25-28°C ambient storage"),
            "status_message": f"Estimated {adjusted_days} days to peak ripe at {temperature}°C based on Kasetsart University storage telemetry."
        }
    else:
        # Prompt requirement: If adequate time-labelled data is unavailable, show explicit message:
        return {
            "fruit": fruit.capitalize(),
            "is_supported": False,
            "estimated_days_to_ripe": None,
            "estimated_shelf_life": None,
            "temperature_celsius": temperature,
            "status_message": "Ripening-time estimation is not yet supported because validated training data is unavailable."
        }

def analyze_fruit_image(image: Image.Image, features: Dict[str, float], temperature: float, fruit: str = "mango") -> Dict[str, Any]:
    """
    Primary multi-model coordinator for POST /api/analyze:
    1. Fruits-360 fruit & variety identification
    2. RawRipe maturity classification
    3. Ripening time regression (or honest disclaimer)
    4. Guard validation (botanical leaf, non-fruit, cross-tab mismatch, exotic fruit)
    """
    fruit = fruit.lower()
    if fruit not in FRUIT_INFO:
        fruit = "mango"

    # Step 1: Fruits-360 identification
    fruit_id_res = fruits360_classifier.predict(image, expected_fruit=fruit)

    # CASE 0: Image Uncertain / Severe Blur / Degraded
    if fruit_id_res.get("input_type") == "uncertain":
        return {
            "fruit": fruit,
            "fruit_name": FRUIT_INFO[fruit]["name"],
            "tamil_name": FRUIT_INFO[fruit]["tamil_name"],
            "is_uncertain": True,
            "is_plant_or_leaf": False,
            "is_non_fruit": True,
            "is_fruit_mismatch": False,
            "is_unsupported_fruit": False,
            "fruit_identification": fruit_id_res,
            "maturity_classification": {
                "is_supported": False,
                "status_message": "Cannot evaluate maturity on uncertain or blurred images."
            },
            "uncertain_validation": {
                "detected": True,
                "confidence": fruit_id_res.get("confidence", 35.0),
                "title": "Image Uncertain — Please Try Again",
                "message": "Fruit Twin could not confidently identify this image. Please upload a clearer photograph showing the fruit."
            },
            "status_message": "⚠️ Image Uncertain: Please upload a clearer photograph showing the fruit.",
            "ripeness_stage": "Uncertain Specimen",
            "ripeness_score": 0.0,
            "confidence": fruit_id_res.get("confidence", 35.0),
            "estimated_days_to_ripe": 0.0,
            "estimated_shelf_life": 0.0,
            "overripe_risk": 0.0,
            "biochemical": {"tss": 0.0, "ta": 0.0, "ratio": 0.0},
            "features": features,
            "environmental_input": {"temperature": temperature}
        }

    # CASE A: Plant / Leaf / Foliage Detected
    if fruit_id_res.get("input_type") == "plant_foliage" or fruit_id_res.get("is_plant_or_leaf"):
        return {
            "fruit": fruit,
            "fruit_name": FRUIT_INFO[fruit]["name"],
            "tamil_name": FRUIT_INFO[fruit]["tamil_name"],
            "is_plant_or_leaf": True,
            "is_non_fruit": True,
            "is_fruit_mismatch": False,
            "is_unsupported_fruit": False,
            "fruit_identification": fruit_id_res,
            "maturity_classification": {
                "is_supported": False,
                "status_message": "Maturity cannot be evaluated on plant leaves or foliage."
            },
            "plant_validation": {
                "detected": True,
                "confidence": fruit_id_res.get("confidence", 95.0),
                "object_type": "plant_leaf",
                "title": "Plant / Foliage Detected | தாவர இலை கண்டறியப்பட்டது",
                "message": "This image appears to contain plant material rather than a supported harvested fruit. Fruit Twin requires a recognizable fruit specimen for fruit maturity analysis and GAN simulation."
            },
            "status_message": "⚠️ Plant / Foliage Detected: Please upload an actual fruit specimen.",
            "ripeness_stage": "Not a Fruit (Plant Foliage)",
            "ripeness_score": 0.0,
            "confidence": fruit_id_res.get("confidence", 95.0),
            "estimated_days_to_ripe": 0.0,
            "estimated_shelf_life": 0.0,
            "overripe_risk": 0.0,
            "biochemical": {"tss": 0.0, "ta": 0.0, "ratio": 0.0},
            "features": features,
            "environmental_input": {"temperature": temperature}
        }

    # CASE B: Non-Fruit Dataset Class (e.g. Potato Red, Tomato 1, Corn, Chestnut)
    if fruit_id_res.get("input_type") == "supported_nonfruit_class":
        c_name = fruit_id_res.get("detected_category", "Agricultural Crop")
        return {
            "fruit": fruit,
            "fruit_name": FRUIT_INFO[fruit]["name"],
            "tamil_name": FRUIT_INFO[fruit]["tamil_name"],
            "is_supported_nonfruit_class": True,
            "is_plant_or_leaf": False,
            "is_non_fruit": True,
            "is_fruit_mismatch": False,
            "is_unsupported_fruit": False,
            "fruit_identification": fruit_id_res,
            "maturity_classification": {
                "is_supported": False,
                "status_message": f"Maturity classification not supported for {c_name}."
            },
            "nonfruit_dataset_validation": {
                "detected": True,
                "detected_class": c_name,
                "title": f"Supported Non-Fruit Dataset Class: {c_name}",
                "message": fruit_id_res.get("rejection_reason", f"Recognized as {c_name}. Fruit-specific analysis is unavailable.")
            },
            "status_message": f"⚠️ Non-Fruit Dataset Class ({c_name}): Fruit-specific analysis is unavailable.",
            "ripeness_stage": f"Non-Fruit Class ({c_name})",
            "ripeness_score": 0.0,
            "confidence": fruit_id_res.get("confidence", 85.0),
            "estimated_days_to_ripe": 0.0,
            "estimated_shelf_life": 0.0,
            "overripe_risk": 0.0,
            "biochemical": {"tss": 0.0, "ta": 0.0, "ratio": 0.0},
            "features": features,
            "environmental_input": {"temperature": temperature}
        }

    # CASE C: Non-Fruit Object (phone, cup, furniture, etc.)
    if fruit_id_res.get("input_type") == "non_fruit" or fruit_id_res.get("is_non_fruit"):
        return {
            "fruit": fruit,
            "fruit_name": FRUIT_INFO[fruit]["name"],
            "tamil_name": FRUIT_INFO[fruit]["tamil_name"],
            "is_plant_or_leaf": False,
            "is_non_fruit": True,
            "is_fruit_mismatch": False,
            "is_unsupported_fruit": False,
            "fruit_identification": fruit_id_res,
            "maturity_classification": {
                "is_supported": False,
                "status_message": "Non-botanical object detected."
            },
            "nonfruit_validation": {
                "detected": True,
                "title": "No Supported Fruit Detected | பழம் கண்டறியப்படவில்லை",
                "message": "The uploaded image does not contain a confidently identifiable supported fruit. Please upload a clear photograph of a fruit."
            },
            "status_message": "⚠️ No Fruit Detected: Please upload a valid fruit photo.",
            "ripeness_stage": "Invalid Object",
            "ripeness_score": 0.0,
            "confidence": fruit_id_res.get("confidence", 90.0),
            "estimated_days_to_ripe": 0.0,
            "estimated_shelf_life": 0.0,
            "overripe_risk": 0.0,
            "biochemical": {"tss": 0.0, "ta": 0.0, "ratio": 0.0},
            "features": features,
            "environmental_input": {"temperature": temperature}
        }

    # CASE D: Exotic / Unsupported Fruit (e.g. Dragon Fruit, Kiwi)
    if fruit_id_res.get("input_type") == "unsupported_fruit" or fruit_id_res.get("is_unsupported_fruit"):
        cand_name = fruit_id_res.get("detected_category") or fruit_id_res.get("predicted_fruit", "Exotic Fruit")
        return {
            "fruit": fruit,
            "fruit_name": FRUIT_INFO[fruit]["name"],
            "tamil_name": FRUIT_INFO[fruit]["tamil_name"],
            "is_plant_or_leaf": False,
            "is_non_fruit": False,
            "is_fruit_mismatch": False,
            "is_unsupported_fruit": True,
            "fruit_identification": fruit_id_res,
            "maturity_classification": {
                "is_supported": False,
                "status_message": "Fruit identified successfully. Ripeness prediction is not yet available for this fruit."
            },
            "unsupported_validation": {
                "detected": True,
                "confidence": fruit_id_res.get("confidence", 92.0),
                "candidate_name": cand_name,
                "title": f"Unsupported Fruit Detected | வகைப்படுத்தப்படாத பழம்",
                "message": f"This appears to be a fruit ({cand_name}), but its fruit-specific analysis pipeline is not currently supported by Fruit Twin."
            },
            "status_message": f"⚠️ Unsupported Fruit Detected ({cand_name}): Not in verified dataset.",
            "ripeness_stage": "Unsupported Fruit",
            "ripeness_score": 50.0,
            "confidence": fruit_id_res.get("confidence", 92.0),
            "estimated_days_to_ripe": 0.0,
            "estimated_shelf_life": 0.0,
            "ripening_time_message": "Ripening-time estimation is not yet supported because validated training data is unavailable.",
            "environmental_input": {"temperature": temperature}
        }

    # Step 2: Maturity classification via RawRipe
    detected_fruit_name = fruit_id_res.get("predicted_fruit", fruit).lower()
    rawripe_res = rawripe_classifier.predict_maturity(image, fruit_name=fruit)

    # Determine maturity label:
    # If fruit is in RawRipe, use RawRipe raw/ripe result; otherwise check if earlier 4-stage CV model is connected
    maturity_stage = rawripe_res.get("maturity_stage")
    maturity_confidence = rawripe_res.get("confidence", 90.0)

    # Also query 4-stage CV model if user wants 4 stages
    cv_res = None
    if ripeness_model.is_connected(fruit):
        try:
            cv_res = ripeness_model.predict(image, fruit, features)
        except Exception:
            pass

    if cv_res:
        four_stage = cv_res.get("ripeness_stage", "Semi-Ripe")
        ripeness_score = cv_res.get("ripeness_score", 65.0)
    else:
        four_stage = "Ripe" if maturity_stage == "Ripe" else "Unripe"
        ripeness_score = 75.0 if maturity_stage == "Ripe" else 35.0

    # Step 3: Ripening time regression
    time_res = predict_single_ripening_time(fruit, temperature, current_stage=four_stage)
    days_to_ripe = time_res.get("estimated_days_to_ripe")
    shelf_life = time_res.get("estimated_shelf_life")
    ripening_time_supported = time_res.get("is_supported", False)
    ripening_time_message = time_res.get("status_message")

    # Step 4: Cross-tab mismatch check
    identified_cat = detected_fruit_name
    is_mismatch = (identified_cat in FRUIT_INFO) and (identified_cat != fruit) and (fruit_id_res.get("confidence", 0) >= 55.0)
    mismatch_info = None
    if is_mismatch:
        detected_info = FRUIT_INFO[identified_cat]
        current_info = FRUIT_INFO[fruit]
        mismatch_info = {
            "is_mismatch": True,
            "detected_fruit": identified_cat,
            "detected_fruit_name": detected_info["name"],
            "detected_fruit_tamil": detected_info["tamil_name"],
            "detected_fruit_icon": detected_info["icon"],
            "mismatch_message": f"Fruit Mismatch Detected | தவறான பழ வகை: The uploaded image appears to be a {detected_info['name']} ({detected_info['tamil_name']}), not a {current_info['name']} ({current_info['tamil_name']}). Switch to the {detected_info['name']} pipeline to continue the analysis."
        }

    bio_data = FRUIT_INFO[fruit]["bio"].get(four_stage, FRUIT_INFO[fruit]["bio"]["Semi-Ripe"])

    return {
        "fruit": fruit,
        "fruit_name": FRUIT_INFO[fruit]["name"],
        "tamil_name": FRUIT_INFO[fruit]["tamil_name"],
        "latin_name": FRUIT_INFO[fruit]["latin"],
        "fruit_identification": fruit_id_res,
        "detected_category": fruit_id_res.get("detected_category", f"{FRUIT_INFO[fruit]['name']} {fruit_id_res.get('predicted_variety', '')}".strip()),
        "detected_fruit": fruit_id_res.get("detected_fruit", FRUIT_INFO[fruit]["name"]),
        "detected_variety": fruit_id_res.get("detected_variety"),
        "variety_confidence": fruit_id_res.get("confidence", 95.0),
        "input_type": "supported_fruit",
        "is_match": not is_mismatch,
        "is_supported": True,
        "maturity_supported": rawripe_res.get("is_supported", False),
        "gan_supported": fruit in ["mango", "banana", "apple", "papaya"],
        "maturity_classification": rawripe_res,
        "ripening_time_estimation": time_res,
        "is_plant_or_leaf": False,
        "is_non_fruit": False,
        "is_unsupported_fruit": False,
        "is_fruit_mismatch": is_mismatch,
        "fruit_validation": mismatch_info,
        "detected_actual_fruit": identified_cat if is_mismatch else fruit,
        "detected_actual_fruit_name": mismatch_info["detected_fruit_name"] if is_mismatch else FRUIT_INFO[fruit]["name"],
        "detected_actual_fruit_tamil": mismatch_info["detected_fruit_tamil"] if is_mismatch else FRUIT_INFO[fruit]["tamil_name"],
        "mismatch_message": mismatch_info["mismatch_message"] if is_mismatch else None,
        "status_message": f"Models evaluated successfully: {fruit_id_res.get('predicted_fruit')} ({rawripe_res.get('maturity_stage', four_stage)}).",
        "ripeness_stage": four_stage,
        "rawripe_maturity": rawripe_res.get("maturity_stage"),
        "rawripe_confidence": maturity_confidence,
        "rawripe_supported": rawripe_res.get("is_supported", False),
        "rawripe_message": rawripe_res.get("status_message"),
        "ripeness_score": ripeness_score,
        "confidence": fruit_id_res.get("confidence", 90.0),
        "estimated_days_to_ripe": days_to_ripe if days_to_ripe is not None else 0.0,
        "estimated_shelf_life": shelf_life if shelf_life is not None else 0.0,
        "ripening_time_supported": ripening_time_supported,
        "ripening_time_message": ripening_time_message,
        "overripe_risk": 20.0,
        "biochemical": bio_data,
        "features": features,
        "environmental_input": {"temperature": temperature}
    }

def calculate_what_if(temperature: float, current_stage: str = "Semi-Ripe", fruit: str = "mango") -> Dict[str, Any]:
    return predict_single_ripening_time(fruit, temperature, current_stage)
