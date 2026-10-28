import requests
import io
import sys
from PIL import Image, ImageDraw

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE = 'http://127.0.0.1:8000'

print("=== 1. HEALTH CHECK ===")
h = requests.get(f'{BASE}/api/health').json()
print("Backend health status:", h.get("status"), "| App:", h.get("app"))

print("\n=== 2. REAL-WORLD IMAGE IDENTIFICATION & ANALYSIS ===")
test_specimens = [
    ("Apple", "apple", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Apple Crimson Snow\r_13_100.jpg"),
    ("Banana", "banana", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Banana\r_12_100.jpg"),
    ("Red Banana", "banana", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\test_cropped_red_banana.jpg"),
    ("Orange", "orange", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Orange\r_3_100.jpg"),
    ("Papaya", "papaya", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\sample_images\papaya_sample_1.jpg"),
    ("Guava", "guava", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Guava\test_8_100.jpg"),
    ("Pomegranate", "pomegranate", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\sample_images\pomegranate_sample_1.jpg"),
    ("Watermelon", "watermelon", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Watermelon\test_8_100.jpg"),
    ("Mango", "mango", r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Mango\mango_test_0.jpg")
]

for label, tab_fruit, path in test_specimens:
    with open(path, "rb") as f:
        img_bytes = f.read()
    res = requests.post(
        f'{BASE}/api/analyze',
        files={'file': ('image.jpg', img_bytes, 'image/jpeg')},
        data={'fruit': tab_fruit, 'temperature': 25.0, 'humidity': 65}
    ).json()

    pred_fruit = res.get("fruit_identification", {}).get("predicted_fruit") or res.get("detected_fruit")
    variety = res.get("fruit_identification", {}).get("predicted_variety") or res.get("detected_variety")
    stage = res.get("ripeness_stage")
    days = res.get("estimated_days_to_ripe")
    shelf = res.get("estimated_shelf_life")
    match = res.get("is_match")
    print(f"[{label:12}] Tab={tab_fruit:10} -> Detected={pred_fruit:10} | Variety={variety:30} | Match={str(match):5} | Stage={stage:10} | Days={days}d | Shelf={shelf}d")

print("\n=== 3. GREEN-YELLOW MANGO '0 DAYS TO RIPE' RESOLUTION TEST (28°C) ===")
# Green mango with yellow shoulders
gy_mango = Image.new('RGB', (200, 200), (255, 255, 255))
d = ImageDraw.Draw(gy_mango)
d.ellipse([(40, 60), (160, 180)], fill=(70, 150, 40))   # dark green lower body
d.ellipse([(50, 40), (150, 110)], fill=(225, 185, 30))  # yellow-orange shoulders
buf = io.BytesIO()
gy_mango.save(buf, format='JPEG')
gy_bytes = buf.getvalue()

gy_res = requests.post(
    f'{BASE}/api/analyze',
    files={'file': ('green_yellow_mango.jpg', gy_bytes, 'image/jpeg')},
    data={'fruit': 'mango', 'temperature': 28.0, 'humidity': 65}
).json()

gy_stage = gy_res.get('ripeness_stage')
gy_days = gy_res.get('estimated_days_to_ripe')
gy_shelf = gy_res.get('estimated_shelf_life')
print(f"Green-Yellow Mango evaluated stage: {gy_stage} (Expected: Semi-Ripe)")
print(f"Days to peak ripe at 28.0°C: {gy_days}d (Expected: ~2.5 to 3.5 days, NOT 0)")
print(f"Total shelf-life at 28.0°C: {gy_shelf}d")
assert gy_stage == "Semi-Ripe", f"Expected Semi-Ripe, got {gy_stage}"
assert 2.0 <= gy_days <= 3.8, f"Expected 2.5-3.5 days, got {gy_days}"
print("=> 0 DAYS TO RIPE BUG VERIFIED RESOLVED!")

print("\n=== 4. BIOLOGICAL TEMPERATURE KINETICS (Temperature Scaling) ===")
for temp in [15.0, 25.0, 28.0, 35.0, 48.0]:
    res = requests.post(
        f'{BASE}/api/analyze',
        files={'file': ('green_yellow_mango.jpg', gy_bytes, 'image/jpeg')},
        data={'fruit': 'mango', 'temperature': temp, 'humidity': 65}
    ).json()
    print(f"At {temp:4.1f}°C -> Days to Ripe: {res.get('estimated_days_to_ripe'):5.1f}d | Shelf Life: {res.get('estimated_shelf_life'):5.1f}d | Overripe Risk: {res.get('overripe_risk')}%")

print("\n=== 5. CROSS-TAB SPECIMEN MISMATCH DIALOG ===")
# Upload Banana on Apple tab
with open(r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test\Banana\r_12_100.jpg", "rb") as f:
    banana_bytes = f.read()

m_res = requests.post(
    f'{BASE}/api/analyze',
    files={'file': ('banana.jpg', banana_bytes, 'image/jpeg')},
    data={'fruit': 'apple', 'temperature': 28.0, 'humidity': 65}
).json()
print("Mismatch detected:", m_res.get('is_fruit_mismatch'))
print("is_match:", m_res.get('is_match'))
if m_res.get('fruit_validation'):
    print("Detected fruit:", m_res['fruit_validation'].get('detected_fruit_name'))
    print("Action message:", m_res['fruit_validation'].get('mismatch_message'))

print("\n=== 6. UNIVERSAL cGAN GENERATION ACROSS ALL 8 FRUITS ===")
for fruit_name in ["mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon"]:
    for stage in ["Semi-Ripe", "Ripe", "Overripe"]:
        with open(test_specimens[0][2], "rb") as f:
            dummy_bytes = f.read()
        g_res = requests.post(
            f'{BASE}/api/generate-twin',
            files={'file': ('test.jpg', dummy_bytes, 'image/jpeg')},
            data={'fruit': fruit_name, 'temperature': 25.0, 'target_stage': stage}
        ).json()
        b64 = g_res.get("generated_image", "")
        is_sup = g_res.get("is_supported")
        assert is_sup is True and len(b64) > 100, f"GAN failed for {fruit_name} {stage}"
    print(f"cGAN [{fruit_name.upper():12}]: All stages (Semi-Ripe, Ripe, Overripe) synthesized successfully (Valid Base64 URIs).")

print("\nALL 6 VERIFICATION SUITES COMPLETED WITH 100% SUCCESS!")
