from PIL import Image
import traceback
from services.gan_service import simulate_future_stage

try:
    print("Testing GAN Simulation Service...")
    img = Image.new("RGB", (200, 200), (40, 160, 50))
    res1 = simulate_future_stage(img, 28.0, "Semi-Ripe")
    print("Semi-Ripe result status:", res1["status_message"])
    print("Base64 string length:", len(res1["generated_image"]))

    res2 = simulate_future_stage(img, 28.0, "Ripe")
    print("Ripe result status:", res2["status_message"])

    res3 = simulate_future_stage(img, 28.0, "Overripe")
    print("Overripe result status:", res3["status_message"])

    print("\nGAN Simulation Test SUCCESSFUL!")
except Exception as e:
    traceback.print_exc()
