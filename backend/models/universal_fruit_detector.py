"""
Universal Multi-Fruit Detector for FruitTwin
Features:
1. Deep CNN Architecture (MobileNetV2) checkpoint loader ('models/universal_fruit_cnn.pth').
2. Precision Optical-Morphological Classification Engine calibrated for real-world Google images:
   - Banana: High aspect ratio (elongation > 1.45 or aspect > 1.35) and yellow/green dominance;
             Red Banana (Sevazhai) via elevated crimson-to-green ratio (R > 1.15 * G, B > 0.28) with crescent elongation.
   - Papaya: Oblong/pyriform geometry (elongation > 1.18 or aspect > 1.20) with salmon-orange pulp or
             yellow-green longitudinal skin streaks. Never blocked simply because orange color is present.
   - Orange: Strictly spherical geometry (elongation < 1.25, aspect between 0.78 and 1.28) with uniform citrus peel texture.
   - Watermelon: Dark green rind with distinct striping (or crimson core) and high foreground mass.
   - Pomegranate: Hexagonal/crenulated silhouette with deep crimson/ruby pericarp (R > 1.4 * G).
   - Guava: Spherical/pyriform, pale lime-green to yellowish-white with low red pigment and moderate surface texture.
   - Apple: Spheroid with central stem indentation, deep crimson red or Granny Smith green.
   - Mango: Asymmetrical kidney/ovoid curve with yellow-orange or green blush.
3. Botanical Guards: Plant foliage/leaves, non-fruit objects, and exotic unsupported fruits.
"""

import os
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

from models.fruit_classifier import detect_fruit_category, SUPPORTED_FRUITS, FRUIT_METADATA

CORE_FRUITS = SUPPORTED_FRUITS
FRUIT_SPECS = FRUIT_METADATA

class UniversalFruitDetector:
    """
    Robust fruit detector supporting all 8 core agricultural fruits with real-world Google image resilience.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.cnn_model = None
        self.has_cnn = False
        
        # Check for PyTorch deep CNN weights using MobileNetV2 architecture
        default_pth = os.path.join(os.path.dirname(__file__), "universal_fruit_cnn.pth")
        target_path = model_path or default_pth
        
        if os.path.exists(target_path):
            try:
                import torch
                import torch.nn as nn
                import torchvision.models as models
                
                mobilenet = models.mobilenet_v2(weights=None)
                in_feat = mobilenet.classifier[1].in_features
                mobilenet.classifier[1] = nn.Linear(in_feat, len(CORE_FRUITS))
                
                state_dict = torch.load(target_path, map_location=torch.device('cpu'))
                mobilenet.load_state_dict(state_dict)
                mobilenet.eval()
                self.cnn_model = mobilenet
                self.has_cnn = True
                print(f"[UniversalFruitDetector] MobileNetV2 PyTorch weights loaded from {target_path}")
            except Exception as e:
                print(f"[UniversalFruitDetector] PyTorch load notice: {e}. Operating in optical-morphological mode.")

    def detect_category(self, image: Image.Image, expected_fruit: Optional[str] = None) -> Tuple[Optional[str], float, Dict[str, Any]]:
        return detect_fruit_category(image, expected_fruit=expected_fruit)

    def predict(self, image: Image.Image) -> Tuple[str, float, str]:
        """
        Classifies real-world Google images across all 8 agricultural fruits.
        Returns:
            (fruit_name, confidence_percentage, variety_name)
        """
        cat_id, conf, meta = self.detect_category(image)
        
        # If guard triggered, return informative botanical status
        if cat_id in ["plant_leaf", "non_fruit", "unsupported_fruit"]:
            cand = meta.get("candidate_fruit", cat_id.replace("_", " ").title())
            return cand, conf, "Non-Supported"

        if cat_id is None:
            # Fallback to Fruits360 model if available
            try:
                from models.fruits360_classifier import fruits360_classifier
                res = fruits360_classifier.predict(image)
                return res.get("predicted_fruit", "Unknown"), res.get("confidence", 50.0), res.get("predicted_variety", "Standard")
            except Exception:
                return "Unknown", 50.0, "Standard"

        fruit_name = FRUIT_SPECS.get(cat_id, {}).get("name", (cat_id or "Mango").capitalize())
        variety_name = meta.get("detected_variety", FRUIT_SPECS.get(cat_id, {}).get("default_variety", "Standard"))
        return fruit_name, conf, variety_name

# Global singleton
universal_fruit_detector = UniversalFruitDetector()
