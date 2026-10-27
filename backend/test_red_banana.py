from PIL import Image
import numpy as np
from services.prediction_service import analyze_fruit_image

arr = np.zeros((150, 150, 3), dtype=np.uint8)
arr[:, :, 0] = 185  # High Red
arr[:, :, 1] = 55   # Low Green
arr[:, :, 2] = 50   # Low Blue

img = Image.fromarray(arr)

features = {
    "green_ratio": 0.05,
    "yellow_ratio": 0.10,
    "red_ratio": 0.85,
    "brightness": 0.45
}

res = analyze_fruit_image(img, features, 28.0, "banana")
print("\n--- RED BANANA IDENTIFICATION TEST ---")
print("Fruit:", res["fruit_name"])
print("Detected Variety:", res["detected_variety"].encode("ascii", "replace").decode("ascii"))
print("Variety Confidence:", f"{res['variety_confidence']}%")
print("Ripeness Stage:", res["ripeness_stage"])
print("Ripeness Score:", f"{res['ripeness_score']}%")
print("Estimated Shelf Life:", f"{res['estimated_shelf_life']} Days")
print("Recommendation Category:", res["recommendation"]["category"])
print("Recommendation Action:", res["recommendation"]["action"])
print("--------------------------------------\n")
