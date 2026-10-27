import os
import pickle
import numpy as np
from typing import Dict, Any, Optional
from models.base import get_fruit_model_paths, SUPPORTED_FRUITS

class FruitRipeningModel:
    """
    Timeline & Shelf-Life Regressor supporting Mango, Banana, Papaya, and Apple.
    """
    def __init__(self):
        self.models = {}
        self._load_all_models()

    def _load_all_models(self):
        for fruit in SUPPORTED_FRUITS:
            paths = get_fruit_model_paths(fruit)
            if os.path.exists(paths["ripening_pkl"]):
                try:
                    with open(paths["ripening_pkl"], "rb") as f:
                        self.models[fruit] = pickle.load(f)
                    print(f"[RipeningModel] Regressor loaded for {fruit}")
                except Exception as e:
                    print(f"[RipeningModel] Load error for {fruit}: {e}")

    def is_connected(self, fruit: str = "mango") -> bool:
        return fruit.lower() in self.models

    def predict(self, stage: str, temperature: float, features: Dict[str, float], fruit: str = "mango") -> Optional[Dict[str, Any]]:
        fruit = fruit.lower()
        if not self.is_connected(fruit):
            return None

        try:
            stage_map = {"Unripe": 1, "Semi-Ripe": 2, "Ripe": 3, "Overripe": 4}
            stage_num = stage_map.get(stage, 2)
            
            r = features.get("R", 43.5)
            g = features.get("G", 41.5)
            b = features.get("B", 14.5)
            w = features.get("W_grams", 320.0)
            c = features.get("C_pF", 2.05)
            
            feature_vector = [[stage_num, temperature, r, g, b, w, c]]
            
            preds = self.models[fruit].predict(feature_vector)[0]
            if isinstance(preds, (list, tuple, np.ndarray)):
                days_to_ripe = float(preds[0])
                shelf_life = float(preds[1]) if len(preds) > 1 else max(0.5, days_to_ripe + 4.0)
            else:
                days_to_ripe = float(preds)
                shelf_life = max(0.5, days_to_ripe + 4.0)
            
            overripe_risk = min(99.0, max(5.0, (temperature - 20) * 2.5 + (4 - days_to_ripe) * 10))
            
            return {
                "fruit": fruit,
                "estimated_days_to_ripe": round(max(0.0, days_to_ripe), 1),
                "estimated_shelf_life": round(max(0.5, shelf_life), 1),
                "overripe_risk": round(overripe_risk, 1)
            }
        except Exception as e:
            print(f"[RipeningModel] Inference error for {fruit}: {e}")
            return None
