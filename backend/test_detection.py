import io
import sys
import requests
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')

API_URL = "http://127.0.0.1:8000/api/analyze"

def make_test_leaf_image():
    # Make a green leaf-like image with high green ratio, texture lines, veins
    im = Image.new("RGB", (300, 300), color=(15, 20, 35))
    draw = ImageDraw.Draw(im)
    # Draw green leaf polygon
    points = [(150, 30), (220, 100), (240, 180), (150, 270), (60, 180), (80, 100)]
    draw.polygon(points, fill=(25, 130, 35), outline=(50, 200, 60))
    # Draw central and lateral veins
    draw.line([(150, 30), (150, 280)], fill=(120, 230, 130), width=3)
    draw.line([(150, 90), (210, 120)], fill=(100, 210, 110), width=2)
    draw.line([(150, 130), (230, 170)], fill=(100, 210, 110), width=2)
    draw.line([(150, 170), (210, 220)], fill=(100, 210, 110), width=2)
    draw.line([(150, 90), (90, 120)], fill=(100, 210, 110), width=2)
    draw.line([(150, 130), (70, 170)], fill=(100, 210, 110), width=2)
    draw.line([(150, 170), (90, 220)], fill=(100, 210, 110), width=2)
    
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()

def make_test_dragon_fruit_image():
    # Make a magenta dragon fruit with green tipped scales
    im = Image.new("RGB", (300, 300), color=(15, 20, 35))
    draw = ImageDraw.Draw(im)
    # Magenta/pink oval fruit body
    draw.ellipse([(60, 50), (240, 260)], fill=(210, 25, 110), outline=(240, 40, 130))
    # Green scales/bracts protruding
    scales = [
        [(50, 90), (30, 75), (60, 70)],
        [(250, 90), (270, 75), (240, 70)],
        [(40, 150), (20, 140), (50, 130)],
        [(260, 150), (280, 140), (250, 130)],
        [(150, 30), (130, 10), (170, 10)]
    ]
    for sc in scales:
        draw.polygon(sc, fill=(110, 190, 50))
    
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()

def make_test_orange_image():
    im = Image.new("RGB", (300, 300), color=(15, 20, 35))
    draw = ImageDraw.Draw(im)
    draw.ellipse([(60, 60), (240, 240)], fill=(240, 120, 10))
    draw.ellipse([(145, 52), (155, 62)], fill=(25, 110, 30))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return buf.getvalue()

print("\n--- Test 1: Uploading Plant Leaf ---")
leaf_bytes = make_test_leaf_image()
res1 = requests.post(API_URL, files={"file": ("leaf.jpg", leaf_bytes, "image/jpeg")}, data={"fruit": "mango", "temperature": 28.0})
data1 = res1.json()
print("is_plant_or_leaf:", data1.get("is_plant_or_leaf"))
print("plant_validation title:", data1.get("plant_validation", {}).get("title"))
print("status_message:", data1.get("status_message"))

print("\n--- Test 2: Uploading Dragon Fruit (Unsupported Fruit) ---")
df_bytes = make_test_dragon_fruit_image()
res2 = requests.post(API_URL, files={"file": ("dragon.jpg", df_bytes, "image/jpeg")}, data={"fruit": "mango", "temperature": 28.0})
data2 = res2.json()
print("is_unsupported_fruit:", data2.get("is_unsupported_fruit"))
print("candidate_fruit:", data2.get("unsupported_validation", {}).get("candidate_name"))
print("status_message:", data2.get("status_message"))

print("\n--- Test 3: Uploading Verified Orange ---")
orange_bytes = make_test_orange_image()
res3 = requests.post(API_URL, files={"file": ("orange.jpg", orange_bytes, "image/jpeg")}, data={"fruit": "orange", "temperature": 24.0})
data3 = res3.json()
print("fruit:", data3.get("fruit_name"))
print("stage:", data3.get("ripeness_stage"))
print("TSS Brix:", data3.get("biochemical", {}).get("tss"))
print("Days to ripe:", data3.get("estimated_days_to_ripe"))
print("is_mismatch:", data3.get("is_fruit_mismatch"))

print("\n--- Test 4: Uploading Orange into Mango Tab (Cross-Fruit Mismatch) ---")
res4 = requests.post(API_URL, files={"file": ("orange.jpg", orange_bytes, "image/jpeg")}, data={"fruit": "mango", "temperature": 28.0})
data4 = res4.json()
print("is_mismatch:", data4.get("is_fruit_mismatch"))
print("detected_actual_fruit_name:", data4.get("detected_actual_fruit_name"))
print("mismatch_message:", data4.get("mismatch_message"))
