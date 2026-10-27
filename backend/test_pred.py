from PIL import Image
import traceback
from services.prediction_service import analyze_mango_image

try:
    print("Testing analyze_mango_image...")
    img = Image.new("RGB", (100, 100), (100, 150, 50))
    res = analyze_mango_image(img, {"green_ratio": 0.5, "yellow_ratio": 0.3, "red_ratio": 0.2}, 28.0)
    print("Success! Output:", res)
except Exception as e:
    traceback.print_exc()
