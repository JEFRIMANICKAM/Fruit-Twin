"""
Class Registry and Capability Matrix for FruitTwin
Dynamically generated from the actual trained Fruits-360 model and RawRipe classifier.
"""

import os
import pickle
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruits360_model.pkl")

# RawRipe supported fruits
RAWRIPE_SUPPORTED_FRUITS = {
    "apple", "banana", "coconut", "guava", "litchi", 
    "mango", "orange", "papaya", "pomegranate", "strawberry"
}

# GAN future simulation supported fruits across all 8 core agricultural fruits
GAN_SUPPORTED_FRUITS = {
    "mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon"
}

# Ripening time telemetry regression supported fruits across all 8 core agricultural fruits
TELEMETRY_SUPPORTED_FRUITS = {
    "mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon"
}

# Primary dashboard pipeline fruits (8 agricultural categories)
PIPELINE_SUPPORTED_FRUITS = {
    "mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon"
}

# Non-fruit dataset class classification
NON_FRUIT_DATASET_CLASSES = {
    "potato red": {"type": "Vegetable / Tuber", "is_fruit": False},
    "potato": {"type": "Vegetable / Tuber", "is_fruit": False},
    "tomato 1": {"type": "Vegetable / Culinary", "is_fruit": False},
    "tomato": {"type": "Vegetable / Culinary", "is_fruit": False},
    "corn": {"type": "Grain / Cereal", "is_fruit": False},
    "chestnut": {"type": "Nut / Seed", "is_fruit": False},
    "hazelnut": {"type": "Nut / Seed", "is_fruit": False},
    "walnut": {"type": "Nut / Seed", "is_fruit": False},
    "ginger": {"type": "Root / Spice", "is_fruit": False},
    "onion": {"type": "Vegetable / Bulb", "is_fruit": False},
    "pepper": {"type": "Vegetable / Spice", "is_fruit": False},
    "eggplant": {"type": "Vegetable", "is_fruit": False},
    "cucumber": {"type": "Vegetable / Gourd", "is_fruit": False}
}

class ClassRegistry:
    """
    Manages the discovered class registry, category metadata, and per-class feature support matrix.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.classes: List[str] = []
        self.class_details: Dict[str, Dict[str, Any]] = {}
        self.evaluation_meta: Dict[str, Any] = {}
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.classes = data.get("classes", [])
                    self.evaluation_meta = data.get("evaluation", {})
            except Exception as e:
                print(f"[ClassRegistry] Notice loading model: {e}")

        # If model failed or empty, fallback to discovering from dataset folder
        if not self.classes:
            f360_train = os.path.join(BASE_DIR, "data", "fruits-360", "Training")
            if os.path.exists(f360_train):
                self.classes = sorted([d for d in os.listdir(f360_train) if os.path.isdir(os.path.join(f360_train, d))])

        # Build comprehensive metadata for every single class
        for cname in self.classes:
            c_lower = cname.lower()
            
            # 1. Determine if fruit or non-fruit dataset class
            is_fruit = True
            category_type = "Fruit"
            for nf_key, nf_meta in NON_FRUIT_DATASET_CLASSES.items():
                if nf_key in c_lower:
                    is_fruit = False
                    category_type = nf_meta["type"]
                    break

            # 2. Extract fruit/item group and variety
            parts = cname.split()
            if len(parts) > 1 and is_fruit:
                group = parts[0]
                variety = " ".join(parts[1:])
            elif not is_fruit:
                group = parts[0]
                variety = " ".join(parts[1:]) if len(parts) > 1 else "Standard"
            else:
                group = cname
                variety = "Standard"

            group_lower = group.lower()

            # 3. Feature availability indicators
            id_supported = True
            maturity_supported = (group_lower in RAWRIPE_SUPPORTED_FRUITS)
            gan_supported = (group_lower in GAN_SUPPORTED_FRUITS)
            telemetry_supported = (group_lower in TELEMETRY_SUPPORTED_FRUITS)
            has_pipeline = (group_lower in PIPELINE_SUPPORTED_FRUITS)

            # Botanical / Tamil translations
            tamil_names = {
                "apple": "ஆப்பிள்",
                "banana": "வாழைப்பழம்",
                "mango": "மாம்பழம்",
                "orange": "ஆரஞ்சு",
                "papaya": "பப்பாளி",
                "guava": "கொய்யா",
                "pomegranate": "மாதுளை",
                "watermelon": "தர்பூசணி",
                "strawberry": "ஸ்ட்ராபெர்ரி",
                "coconut": "தேங்காய்",
                "litchi": "லிச்சி",
                "potato": "உருளைக்கிழங்கு",
                "tomato": "தக்காளி",
                "corn": "மக்காச்சோளம்",
                "chestnut": "செஸ்ட்நட்"
            }
            tamil_name = tamil_names.get(group_lower, group)

            self.class_details[cname] = {
                "class_name": cname,
                "group": group,
                "variety": variety,
                "tamil_name": tamil_name,
                "category_type": category_type,
                "is_fruit": is_fruit,
                "identification_supported": id_supported,
                "maturity_supported": maturity_supported,
                "gan_supported": gan_supported,
                "ripening_time_supported": telemetry_supported,
                "has_dedicated_pipeline": has_pipeline,
                "maturity_dataset": "RawRipe" if maturity_supported else None,
                "gan_architecture": "Conditional GAN Generator" if gan_supported else None
            }

    def get_catalog(self) -> Dict[str, Any]:
        """Returns the full class catalog with actual counts and capability indicators."""
        return {
            "total_classes": len(self.classes),
            "classes": self.classes,
            "class_details": list(self.class_details.values()),
            "fruit_classes_count": sum(1 for c in self.class_details.values() if c["is_fruit"]),
            "non_fruit_classes_count": sum(1 for c in self.class_details.values() if not c["is_fruit"]),
            "fruits_count": sum(1 for c in self.class_details.values() if c["is_fruit"]),
            "non_fruits_count": sum(1 for c in self.class_details.values() if not c["is_fruit"]),
            "evaluation": self.evaluation_meta
        }

    def get_class_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        return self.class_details.get(class_name)

# Singleton instance
class_registry = ClassRegistry()
