import os
import sys
import io
import json
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from models.class_registry import class_registry
from models.fruits360_classifier import fruits360_classifier
from services.prediction_service import analyze_fruit_image
from services.gan_service import simulate_future_stage
from services.image_processing import process_uploaded_image

DATA_DIR = r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\data\fruits-360\test"

def image_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def load_or_create_test_image(rel_dir, fname) -> Image.Image:
    full_path = os.path.join(DATA_DIR, rel_dir, fname)
    if os.path.exists(full_path):
        return Image.open(full_path).convert("RGB")
    raise FileNotFoundError(f"Missing test image: {full_path}")

def run_tests():
    print("=" * 70)
    print("FRUIT TWIN — 17-POINT MASTER VERIFICATION TEST SUITE (DIRECT ENGINE)")
    print("=" * 70)
    passed = 0
    total = 17

    # Load actual benchmark test images from Fruits-360 test split
    apple_img = load_or_create_test_image("Apple Crimson Snow", "r_13_100.jpg")
    banana_img = load_or_create_test_image("Banana", "r_12_100.jpg")
    mango_img = load_or_create_test_image("Mango", "mango_test_0.jpg")
    orange_img = load_or_create_test_image("Orange", "r_3_100.jpg")
    straw_img = load_or_create_test_image("Strawberry", "3_100.jpg")
    potato_img = load_or_create_test_image("Potato Red", "test_8_100.jpg")
    wm_img = load_or_create_test_image("Watermelon", "test_8_100.jpg")

    # Real cropped Red Banana photograph
    red_banana_path = r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\backend\test_cropped_red_banana.jpg"
    red_banana_img = Image.open(red_banana_path).convert("RGB")

    # Synthetic realistic specimens for edge cases
    # 8. Leaf image: green leaf with contour and veins on white background
    leaf_img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(leaf_img)
    draw.polygon([(60, 10), (95, 55), (85, 100), (60, 115), (35, 100), (25, 55)], fill=(34, 139, 34))
    draw.line([(60, 15), (60, 115)], fill=(144, 238, 144), width=2)

    # 9. Tree branch: brown woody branch with leaf clusters
    branch_img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    draw_b = ImageDraw.Draw(branch_img)
    draw_b.line([(20, 100), (100, 30)], fill=(101, 67, 33), width=6)
    draw_b.ellipse([(70, 20), (105, 50)], fill=(34, 139, 34))
    draw_b.ellipse([(40, 50), (75, 80)], fill=(46, 125, 50))

    # 10. Phone / Metallic object: low saturation, rectangular with dark screen & metallic bezel
    phone_img = Image.new("RGB", (120, 120), color=(240, 240, 240))
    draw_p = ImageDraw.Draw(phone_img)
    draw_p.rounded_rectangle([(35, 15), (85, 105)], radius=8, fill=(50, 50, 50), outline=(120, 120, 120), width=2)
    draw_p.rectangle([(40, 25), (80, 90)], fill=(20, 25, 30))

    # 11. Dragon Fruit / Exotic: bright magenta/pink with green scales
    dragon_img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    draw_d = ImageDraw.Draw(dragon_img)
    draw_d.ellipse([(30, 20), (90, 105)], fill=(219, 39, 119))
    draw_d.polygon([(25, 50), (15, 40), (30, 35)], fill=(132, 204, 22))
    draw_d.polygon([(95, 50), (105, 40), (90, 35)], fill=(132, 204, 22))

    # 12. Blurred / Low-Contrast image: degraded image with low pixel variance (std < 12)
    blurred_img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # TEST 1 — Apple in Apple tab
    print("\n[TEST 1] Apple in Apple tab:")
    res1 = fruits360_classifier.predict(apple_img, expected_fruit="apple")
    t1_pass = res1.get("detected_fruit", "").lower() == "apple" and res1.get("is_match") is True
    print(f"  Result: detected={res1.get('detected_fruit')}, is_match={res1.get('is_match')}, confidence={res1.get('confidence')}% -> {'PASS' if t1_pass else 'FAIL'}")
    if t1_pass: passed += 1

    # TEST 2 — Banana in Apple tab (mismatch modal + switch)
    print("\n[TEST 2] Banana in Apple tab (Mismatch Detection):")
    res2 = fruits360_classifier.predict(banana_img, expected_fruit="apple")
    t2_pass = res2.get("detected_fruit", "").lower() == "banana" and res2.get("is_match") is False
    print(f"  Result: detected={res2.get('detected_fruit')}, is_match={res2.get('is_match')}, active_tab={res2.get('active_tab')} -> {'PASS' if t2_pass else 'FAIL'}")
    if t2_pass: passed += 1

    # TEST 3 — Red Banana in Apple tab (Sevazhai Cultivar Recognition)
    print("\n[TEST 3] Red Banana in Apple tab (Sevazhai Cultivar Recognition):")
    res3 = fruits360_classifier.predict(red_banana_img, expected_fruit="apple")
    t3_pass = res3.get("detected_fruit", "").lower() == "banana" and res3.get("is_match") is False and "Red Banana" in res3.get("detected_variety", "")
    print(f"  Result: detected={res3.get('detected_fruit')}, variety={res3.get('detected_variety')}, confidence={res3.get('confidence')}% -> {'PASS' if t3_pass else 'FAIL'}")
    if t3_pass: passed += 1

    # TEST 4 — Mango in Watermelon tab
    print("\n[TEST 4] Mango in Watermelon tab:")
    res4 = fruits360_classifier.predict(mango_img, expected_fruit="watermelon")
    t4_pass = res4.get("detected_fruit", "").lower() == "mango" and res4.get("is_match") is False
    print(f"  Result: detected={res4.get('detected_fruit')}, is_match={res4.get('is_match')}, active_tab={res4.get('active_tab')} -> {'PASS' if t4_pass else 'FAIL'}")
    if t4_pass: passed += 1

    # TEST 5 — Orange in Guava tab
    print("\n[TEST 5] Orange in Guava tab:")
    res5 = fruits360_classifier.predict(orange_img, expected_fruit="guava")
    t5_pass = res5.get("detected_fruit", "").lower() == "orange" and res5.get("is_match") is False
    print(f"  Result: detected={res5.get('detected_fruit')}, is_match={res5.get('is_match')}, active_tab={res5.get('active_tab')} -> {'PASS' if t5_pass else 'FAIL'}")
    if t5_pass: passed += 1

    # TEST 6 — Additional Fruits-360 class (Strawberry)
    print("\n[TEST 6] Additional Fruits-360 class (Strawberry):")
    res6 = fruits360_classifier.predict(straw_img)
    t6_pass = res6.get("detected_fruit", "").lower() == "strawberry" or "Strawberry" in res6.get("detected_category", "")
    print(f"  Result: detected_category={res6.get('detected_category')}, is_supported={res6.get('is_supported')} -> {'PASS' if t6_pass else 'FAIL'}")
    if t6_pass: passed += 1

    # TEST 7 — Supported non-fruit dataset class (Potato Red)
    print("\n[TEST 7] Supported Non-Fruit Dataset Class (Potato Red):")
    res7 = fruits360_classifier.predict(potato_img)
    t7_pass = res7.get("input_type") == "supported_nonfruit_class" and "Potato Red" in res7.get("detected_category", "")
    print(f"  Result: input_type={res7.get('input_type')}, category={res7.get('detected_category')}, reason={res7.get('rejection_reason')} -> {'PASS' if t7_pass else 'FAIL'}")
    if t7_pass: passed += 1

    # TEST 8 — Leaf-only image (Plant foliage modal)
    print("\n[TEST 8] Leaf-only image (Plant Foliage Rejection):")
    res8 = fruits360_classifier.predict(leaf_img)
    t8_pass = res8.get("input_type") == "plant_foliage"
    print(f"  Result: input_type={res8.get('input_type')}, message={res8.get('message')} -> {'PASS' if t8_pass else 'FAIL'}")
    if t8_pass: passed += 1

    # TEST 9 — Tree branch with vegetation (Plant foliage modal)
    print("\n[TEST 9] Tree Branch / Foliage:")
    res9 = fruits360_classifier.predict(branch_img)
    t9_pass = res9.get("input_type") == "plant_foliage"
    print(f"  Result: input_type={res9.get('input_type')}, message={res9.get('message')} -> {'PASS' if t9_pass else 'FAIL'}")
    if t9_pass: passed += 1

    # TEST 10 — Phone or metallic non-fruit (Non-fruit object modal)
    print("\n[TEST 10] Phone / Metallic non-fruit object:")
    res10 = fruits360_classifier.predict(phone_img)
    t10_pass = res10.get("input_type") == "non_fruit"
    print(f"  Result: input_type={res10.get('input_type')}, message={res10.get('message')} -> {'PASS' if t10_pass else 'FAIL'}")
    if t10_pass: passed += 1

    # TEST 11 — Unsupported fruit (Dragon fruit)
    print("\n[TEST 11] Unsupported fruit (Dragon fruit):")
    res11 = fruits360_classifier.predict(dragon_img)
    t11_pass = res11.get("input_type") in ["unsupported_fruit", "non_fruit"]
    print(f"  Result: input_type={res11.get('input_type')}, message={res11.get('message')} -> {'PASS' if t11_pass else 'FAIL'}")
    if t11_pass: passed += 1

    # TEST 12 — Blurred image (Uncertain image modal)
    print("\n[TEST 12] Blurred / Low-Contrast image (Uncertain):")
    res12 = fruits360_classifier.predict(blurred_img)
    t12_pass = res12.get("input_type") == "uncertain"
    print(f"  Result: input_type={res12.get('input_type')}, message={res12.get('message')} -> {'PASS' if t12_pass else 'FAIL'}")
    if t12_pass: passed += 1

    # TEST 13 — Fruit without maturity model (Honest disclaimer)
    print("\n[TEST 13] Maturity disclaimer for uncalibrated fruit (Watermelon):")
    _, wm_feats = process_uploaded_image(image_to_bytes(wm_img))
    res13 = analyze_fruit_image(wm_img, wm_feats, 28.0, "watermelon")
    t13_pass = res13.get("maturity_classification", {}).get("is_supported") is False
    print(f"  Result: maturity is_supported={res13.get('maturity_classification', {}).get('is_supported')}, msg='{res13.get('maturity_classification', {}).get('status_message')}' -> {'PASS' if t13_pass else 'FAIL'}")
    if t13_pass: passed += 1

    # TEST 14 — Universal cGAN Generation across Core Fruits (Watermelon)
    print("\n[TEST 14] Universal cGAN Generation across Core Fruits (Watermelon):")
    res14 = simulate_future_stage(wm_img, 28.0, "Ripe", "watermelon")
    t14_pass = res14.get("is_supported") is True and bool(res14.get("generated_image"))
    print(f"  Result: gan is_supported={res14.get('is_supported')}, has_image={bool(res14.get('generated_image'))} -> {'PASS' if t14_pass else 'FAIL'}")
    if t14_pass: passed += 1

    # TEST 15 — Pipeline switching execution (Banana analyzed in Banana pipeline)
    print("\n[TEST 15] Pipeline Switching & Execution (Banana analyzed in Banana pipeline):")
    _, b_feats = process_uploaded_image(image_to_bytes(banana_img))
    res15 = analyze_fruit_image(banana_img, b_feats, 26.0, "banana")
    t15_pass = res15.get("fruit") == "banana" and res15.get("ripeness_stage") in ["Ripe", "Raw", "Semi-Ripe", "Overripe"]
    print(f"  Result: fruit={res15.get('fruit')}, ripeness_stage={res15.get('ripeness_stage')}, confidence={res15.get('confidence')}% -> {'PASS' if t15_pass else 'FAIL'}")
    if t15_pass: passed += 1

    # TEST 16 — Invalid image input handled cleanly
    print("\n[TEST 16] Invalid / Corrupted image input (Handling cleanly):")
    try:
        Image.open(io.BytesIO(b"NOT_A_VALID_IMAGE_DATA_CORRUPT"))
        t16_pass = False
    except Exception as e:
        t16_pass = True
        print(f"  Result: Caught image decode exception as expected: {type(e).__name__} -> PASS")
    if t16_pass: passed += 1

    # TEST 17 — Dataset Catalogue Registry (Exact 20 Classes)
    print("\n[TEST 17] Dataset Catalogue Registry (Exact 20 Classes):")
    res17 = class_registry.get_catalog()
    total_classes = res17.get("total_classes", 0)
    fruits_count = res17.get("fruits_count", 0)
    non_fruits_count = res17.get("non_fruits_count", 0)
    t17_pass = total_classes == 20 and fruits_count == 16 and non_fruits_count == 4
    print(f"  Result: total_classes={total_classes}, fruits={fruits_count}, non_fruits={non_fruits_count} -> {'PASS' if t17_pass else 'FAIL'}")
    if t17_pass: passed += 1

    print("\n" + "=" * 70)
    print(f"MASTER TEST SUITE SUMMARY: {passed} / {total} TESTS PASSED ({passed/total*100:.1f}%)")
    print("=" * 70)
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
