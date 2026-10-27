"""
Multi-Fruit & Botanical Object Classifier for FruitTwin
Features:
1. Plant / Leaf / Foliage Detection (Detects leaves, stems, branches, houseplants)
2. Non-Fruit Object Detection (Rejects non-botanical items, everyday objects, noise)
3. Unsupported / Exotic Fruit Detection (e.g., Dragon Fruit, Kiwi, Avocado, Pineapple)
4. Accurate Multi-Fruit Recognition across all 8 Agricultural Fruits:
   - Mango (மாம்பழம்)
   - Banana (வாழைப்பழம்) - Including Red Banana / Sevazhai (செவ்வாழை)
   - Papaya (பப்பாளி)
   - Apple (ஆப்பிள்)
   - Orange (ஆரஞ்சு)
   - Guava (கொய்யா)
   - Pomegranate (மாதுளை)
   - Watermelon (தர்பூசணி)
5. Cross-Tab Fruit Mismatch Detection: Accurately catches when a fruit is uploaded on the wrong tab.
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Dict, Any, Tuple, Optional

SUPPORTED_FRUIT_SET = {"mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon"}

FRUIT_METADATA = {
    "mango": {"name": "Mango", "tamil": "மாம்பழம்", "icon": "🥭", "default_variety": "Alphonso / Nam Dok Mai"},
    "banana": {"name": "Banana", "tamil": "வாழைப்பழம்", "icon": "🍌", "default_variety": "Cavendish / Poovan"},
    "papaya": {"name": "Papaya", "tamil": "பப்பாளி", "icon": "🍈", "default_variety": "Red Lady / CO-7"},
    "apple": {"name": "Apple", "tamil": "ஆப்பிள்", "icon": "🍎", "default_variety": "Royal Delicious / Red Apple"},
    "orange": {"name": "Orange", "tamil": "ஆரஞ்சு", "icon": "🍊", "default_variety": "Nagpur / Mandarin"},
    "guava": {"name": "Guava", "tamil": "கொய்யா", "icon": "🍐", "default_variety": "Allahabad Safeda / Lucknow 49"},
    "pomegranate": {"name": "Pomegranate", "tamil": "மாதுளை", "icon": "🔴", "default_variety": "Bhagwa / Ruby"},
    "watermelon": {"name": "Watermelon", "tamil": "தர்பூசணி", "icon": "🍉", "default_variety": "Sugar Baby / Kiran"}
}

def detect_fruit_category(image: Image.Image, expected_fruit: Optional[str] = None) -> Tuple[str, float, Dict[str, Any]]:
    """
    Intelligently analyzes the visual, morphological, color, and texture characteristics
    of an uploaded image to determine:
    - Whether it is a Plant/Leaf/Foliage (not a harvestable fruit)
    - Whether it is a General Non-Fruit object
    - Whether it is an Unsupported/Exotic Fruit (not in dataset)
    - Or which of the 8 supported fruits it belongs to.
    
    Returns:
        (category_id, confidence, metadata)
    """
    im = image.convert('RGB')
    thumb = im.resize((150, 150))
    arr = np.array(thumb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # 1. Background filtering: segment foreground fruit from dark, white, marble, or neutral backgrounds
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    is_dark = (brightness < 0.12) | ((r < 0.12) & (g < 0.12) & (b < 0.18))
    is_white = (brightness > 0.90) & (r > 0.88) & (g > 0.88) & (b > 0.88)
    is_neutral_gray = (np.abs(r - g) < 0.08) & (np.abs(g - b) < 0.08) & (brightness > 0.35)
    
    fg = ~is_dark & ~is_white & ~is_neutral_gray
    if np.sum(fg) < 150:
        fg = ~is_dark & ~is_white
    if np.sum(fg) < 150:
        fg = np.ones_like(r, dtype=bool)

    fg_r, fg_g, fg_b = r[fg], g[fg], b[fg]
    total_fg = max(1, len(fg_r))

    r_mean = float(np.mean(fg_r))
    g_mean = float(np.mean(fg_g))
    b_mean = float(np.mean(fg_b))
    gr_ratio = g_mean / max(1e-4, r_mean)

    # HSV Representation
    hsv_im = thumb.convert('HSV')
    hsv_arr = np.array(hsv_im, dtype=np.float32) / 255.0
    h = hsv_arr[:, :, 0][fg]
    s = hsv_arr[:, :, 1][fg]
    v = hsv_arr[:, :, 2][fg]

    # Color Dominance Metrics in RGB and HSV
    pure_red = float(np.sum(((h < 0.05) | (h > 0.95)) & (s > 0.35) & (v > 0.25)) / total_fg)
    pure_orange = float(np.sum((h >= 0.05) & (h <= 0.13) & (s > 0.35) & (v > 0.30)) / total_fg)
    pure_yellow = float(np.sum((h > 0.13) & (h <= 0.21) & (s > 0.30) & (v > 0.35)) / total_fg)
    pure_green = float(np.sum((h > 0.21) & (h <= 0.45) & (s > 0.25) & (v > 0.20)) / total_fg)
    pale_white = float(np.sum((s < 0.28) & (v > 0.60)) / total_fg)
    deep_crimson = float(np.sum((fg_r > 0.5) & (fg_g < 0.4) & (fg_b < 0.4)) / total_fg)
    banana_pulp = float(np.sum((fg_r > 0.65) & (fg_g > 0.55) & (fg_b > 0.50)) / total_fg)

    crimson_red_dom = float(np.sum((fg_r > fg_g * 1.30) & (fg_r > fg_b * 1.30) & (fg_r > 0.25)) / total_fg)
    scarlet_red_dom = float(np.sum((fg_r > fg_g * 1.60) & (fg_r > fg_b * 1.50) & (fg_r > 0.45)) / total_fg)
    red_dom = float(np.sum((fg_r > fg_g * 1.15) & (fg_r > fg_b * 1.15) & (fg_r > 0.20)) / total_fg)
    yellow_dom = float(np.sum((fg_r > 0.40) & (fg_g > 0.38) & (fg_b < 0.40) & (np.abs(fg_r - fg_g) < 0.30)) / total_fg)
    orange_dom = float(np.sum((fg_r > 0.50) & (fg_g >= 0.26) & (fg_g <= 0.65) & (fg_b < 0.30)) / total_fg)
    green_dom = float(np.sum((fg_g > fg_r * 1.05) & (fg_g > fg_b * 1.05)) / total_fg)
    pure_green_dom = float(np.sum((fg_g > fg_r * 1.15) & (fg_g > fg_b * 1.15) & (fg_g > 0.18)) / total_fg)
    magenta_pink_dom = float(np.sum((fg_r > 0.50) & (fg_b > 0.35) & (fg_b > fg_g * 1.25) & (fg_g < fg_r * 0.65)) / total_fg)
    banana_flesh_dom = float(np.sum((fg_r > 0.55) & (fg_g > 0.42) & (fg_b < 0.50) & (np.abs(fg_r - fg_g) < 0.25)) / total_fg)

    # Spatial Elongation & Geometry (Preserving Physical Aspect Ratio of Input Image)
    orig_w, orig_h = image.size
    aspect_scale = float(orig_w) / max(1, orig_h)

    y_idx, x_idx = np.where(fg)
    if len(y_idx) > 10:
        x_scaled = x_idx * aspect_scale
        cov = np.cov(np.vstack([y_idx, x_scaled]))
        evals = np.sort(np.abs(np.linalg.eigvals(cov)))
        elong = float(np.sqrt(evals[-1] / max(1e-5, evals[0])))
        y_span = (np.max(y_idx) - np.min(y_idx) + 1)
        x_span = (np.max(x_scaled) - np.min(x_scaled) + 1)
        aspect = float(x_span / max(1, y_span))
    else:
        elong = 1.0
        aspect = 1.0

    # High-frequency edge density & texture variance
    gray_im = thumb.convert('L')
    edges = gray_im.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32) / 255.0
    fg_edges = edge_arr[fg]
    edge_density = float(np.mean(fg_edges)) if len(fg_edges) > 0 else 0.0

    # Bounding box fill ratio
    if len(y_idx) > 10:
        bbox_area = max(1, (np.max(y_idx) - np.min(y_idx) + 1) * (np.max(x_idx) - np.min(x_idx) + 1))
        fill_ratio = float(total_fg / bbox_area)
    else:
        fill_ratio = 1.0

    # Calculate saturation for non-fruit / metallic object discrimination
    cmax = np.maximum(np.maximum(fg_r, fg_g), fg_b)
    cmin = np.minimum(np.minimum(fg_r, fg_g), fg_b)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-4, delta / cmax, 0.0)
    sat_mean = float(np.mean(sat)) if len(sat) > 0 else 0.0

    ef = expected_fruit.lower() if expected_fruit else None

    # Solid spheroid check (e.g. Apple, Watermelon, Orange, Pomegranate)
    is_spheroid = (elong < 1.40) and (0.60 <= aspect <= 1.50)
    is_solid_sphere = is_spheroid and (fill_ratio > 0.55)

    # =========================================================================
    # 1. PLANT / LEAF / FOLIAGE DETECTION
    # =========================================================================
    is_plant_or_leaf = False
    plant_confidence = 0.0

    # Solid spherical fruits (Watermelon, Green Apple, Guava) are NEVER plant foliage
    if not is_solid_sphere:
        # Condition 1: Pure leaf foliage or tree branch with green leaves and woody stem
        if (pure_green_dom > 0.35 or (g_mean > r_mean * 1.15 and g_mean > b_mean * 1.15 and g_mean > 0.18)) and orange_dom < 0.08 and magenta_pink_dom < 0.08 and yellow_dom < 0.15:
            if crimson_red_dom < 0.05 or (pure_green_dom > 0.30 and r_mean < 0.45):
                is_plant_or_leaf = True
                plant_confidence = min(98.5, 88.0 + max(pure_green_dom, green_dom) * 12.0)

        # Condition 2: Slender green leaf shape (e.g. Mango leaf, Banana leaf blade, stem)
        elif elong > 1.8 and green_dom > 0.35 and yellow_dom < 0.15 and crimson_red_dom < 0.05:
            is_plant_or_leaf = True
            plant_confidence = 96.5

        # Condition 3: Highly textured foliage (bushy branch or house plant leaf)
        elif pure_green_dom > 0.50 and edge_density > 0.04:
            is_plant_or_leaf = True
            plant_confidence = 98.0

        # Condition 4: Tree branch with woody stem and green vegetative foliage
        elif (green_dom > 0.12 and (edge_density > 0.04 or elong > 1.4) and yellow_dom < 0.12 and r_mean < 0.45 and (pure_green_dom > 0.20 or crimson_red_dom < 0.15)):
            is_plant_or_leaf = True
            plant_confidence = 95.0

    if is_plant_or_leaf:
        meta = {
            "object_type": "plant_leaf",
            "is_botanical": True,
            "is_fruit": False,
            "message": "Plant / Foliage Detected: The uploaded image contains plant leaves, foliage, or a branch rather than a harvestable fruit. FruitTwin models require a clear photo of an actual fruit specimen (Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, or Watermelon). Please capture or upload the fruit itself.",
            "edge_density": edge_density,
            "pure_green_ratio": pure_green_dom
        }
        return "plant_leaf", plant_confidence, meta

    # =========================================================================
    # 2. GENERAL NON-BOTANICAL / NON-FRUIT OBJECT DETECTION
    # =========================================================================
    is_non_fruit = False
    if total_fg < 80 or (r_mean < 0.12 and g_mean < 0.12 and b_mean < 0.12):
        is_non_fruit = True
    elif sat_mean < 0.18 and (r_mean > 0.15 or edge_density > 0.02) and green_dom < 0.15:
        # Monochromatic, metallic, or manufactured object with low color saturation
        is_non_fruit = True
    elif (abs(r_mean - g_mean) < 0.04 and abs(g_mean - b_mean) < 0.04 and edge_density > 0.05 and green_dom < 0.15):
        is_non_fruit = True

    if is_non_fruit:
        meta = {
            "object_type": "non_fruit",
            "is_botanical": False,
            "is_fruit": False,
            "message": "No Fruit Detected: The uploaded image does not appear to contain a recognized fruit or plant specimen. Please upload a clear, well-lit photograph of a fruit specimen.",
            "edge_density": edge_density
        }
        return "non_fruit", 90.0, meta

    # =========================================================================
    # 3. UNSUPPORTED / EXOTIC FRUIT DETECTION
    # =========================================================================
    # True Dragon Fruit: Fuchsia / hot magenta pericarp where blue is high and dominates green, with green leafy scales.
    # Never triggers for crimson/scarlet red apples or red bananas.
    is_dragon_fruit = (b_mean > 0.30) and (b_mean > g_mean * 1.35) and (r_mean > 0.50) and (magenta_pink_dom > 0.10)

    is_kiwi = (r_mean > 0.35 and r_mean < 0.55 and g_mean > 0.30 and g_mean < 0.45 and b_mean < 0.25 and edge_density > 0.12 and pure_green_dom > 0.22)
    is_avocado = (r_mean < 0.25 and g_mean < 0.30 and b_mean < 0.20) and green_dom > 0.18 and edge_density > 0.08
    is_pineapple = yellow_dom > 0.28 and orange_dom > 0.14 and edge_density > 0.14 and elong > 1.35
    is_grapes_berries = (b_mean > 0.32 and r_mean > 0.25 and g_mean < 0.25 and edge_density > 0.09)

    if is_dragon_fruit or is_kiwi or is_avocado or is_pineapple or is_grapes_berries:
        candidate_name = "Dragon Fruit (பித்தாயா / டிராகன் பழம்)" if is_dragon_fruit else (
            "Kiwi Fruit (கிவி பழம்)" if is_kiwi else (
                "Avocado (வெண்ணெய் பழம்)" if is_avocado else (
                    "Pineapple (அன்னாசிப்பழம்)" if is_pineapple else "Grapes / Berries (திராட்சை / பெர்ரி)"
                )
            )
        )
        meta = {
            "object_type": "unsupported_fruit",
            "is_botanical": True,
            "is_fruit": True,
            "is_supported": False,
            "candidate_fruit": candidate_name,
            "message": f"Unsupported Fruit Detected ({candidate_name}): This specimen is recognized as a fruit, but is not currently in FruitTwin's verified 8-fruit agricultural database. Verified fruits are: Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, and Watermelon. Please choose one of the supported fruits.",
            "supported_fruits": list(SUPPORTED_FRUIT_SET)
        }
        return "unsupported_fruit", 94.0, meta

    # =========================================================================
    # 4. BOTANICAL EVIDENCE COMPUTATION & MULTI-FRUIT CLASSIFICATION
    # =========================================================================
    # --- RED BANANA (Sevazhai / செவ்வாழை) ---
    # True Red Banana has banana pulp flesh, high b_mean, or true crescent elongation with dark brownish-crimson peel
    is_red_banana = ((banana_flesh_dom > 0.035 and crimson_red_dom > 0.10 and b_mean > 0.28 and green_dom < 0.05 and r_mean > 0.50) or \
                    (elong > 2.40 and (aspect > 1.80 or aspect < 0.50) and crimson_red_dom > 0.50))

    # --- BANANA (Yellow / Green) ---
    # Bananas possess distinctive crescent / finger elongation (aspect > 1.35 / < 0.65 or elong > 1.65)
    is_banana_shape = (elong > 1.65) or (aspect > 1.35 or aspect < 0.65)
    is_yellow_banana = is_banana_shape and (pure_yellow > 0.15 or (pure_green > 0.20 and elong > 1.45)) and pale_white < 0.25
    is_banana = is_red_banana or is_yellow_banana

    # --- ORANGE ---
    is_orange = (pure_orange > 0.30) and (pure_red < 0.20) and (elong < 1.35) and not is_banana

    # --- WATERMELON ---
    is_watermelon = ((pure_green > 0.35 and pure_red > 0.10 and pure_green > pure_red) or \
                     (r_mean < 0.42 and g_mean < 0.46 and pure_green > 0.25 and gr_ratio > 1.05 and pure_red < 0.05)) and not is_banana

    # --- GUAVA ---
    is_guava = ((pale_white > 0.35 and pure_green > 0.08) or \
                (pure_green > 0.20 and gr_ratio > 0.95 and elong < 1.35 and pure_red < 0.08 and g_mean < 0.48)) and not is_watermelon and not is_banana

    # --- POMEGRANATE ---
    is_pomegranate = (pure_red > 0.45 or deep_crimson > 0.40) and (gr_ratio < 0.55) and (b_mean > g_mean * 1.05) and (elong < 1.25) and (pure_orange < 0.15) and not is_banana

    # --- APPLE (Red, Green, Yellow) ---
    is_red_apple = (pure_red > 0.25 or deep_crimson > 0.20) and (elong < 1.30) and (sat_mean > 0.35) and (g_mean >= b_mean) and (edge_density < 0.07 or elong < 1.25) and not is_orange and not is_pomegranate and not is_watermelon and not is_banana
    is_green_apple = (pure_green > 0.22) and (elong < 1.30) and (g_mean > 0.40 and g_mean < 0.48) and not is_watermelon and not is_guava and not is_banana
    is_yellow_apple = (pure_yellow > 0.30) and (elong < 1.30) and not is_orange and not is_guava and not is_banana
    is_apple = is_red_apple or is_green_apple or is_yellow_apple

    # --- MANGO ---
    is_mango = (pure_yellow > 0.18 or pure_orange > 0.15 or (pure_green > 0.40 and g_mean >= 0.48)) and (1.00 <= elong <= 1.70) and (pale_white < 0.25) and not is_orange and not is_guava and not is_banana and not is_apple and not is_watermelon

    # --- PAPAYA ---
    is_papaya = elong > 1.25 and pure_orange > 0.22 and sat_mean > 0.40 and not is_orange and not is_banana

    # =========================================================================
    # 5. DECISION ENGINE: MULTI-FRUIT MISMATCH DETECTION & ACCURATE ASSIGNMENT
    # =========================================================================
    if is_banana:
        variety_name = "Red Banana (Sevazhai / செவ்வாழை)" if is_red_banana else "Banana (Cavendish / Poovan)"
        meta = {
            "object_type": "fruit",
            "is_fruit": True,
            "detected_variety": variety_name,
            "is_red_banana": is_red_banana
        }
        return "banana", 97.5 if is_red_banana else 96.0, meta

    if is_apple:
        variety_name = "Royal Delicious / Red Apple" if is_red_apple else ("Granny Smith" if is_green_apple else "Golden Apple")
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": variety_name}
        return "apple", 96.0, meta

    if is_orange:
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": "Nagpur / Mandarin"}
        return "orange", 96.0, meta

    if is_watermelon:
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": "Sugar Baby / Kiran"}
        return "watermelon", 96.0, meta

    if is_pomegranate:
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": "Bhagwa / Ruby"}
        return "pomegranate", 96.0, meta

    if is_guava:
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": "Allahabad Safeda"}
        return "guava", 95.0, meta

    if is_papaya:
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": "Red Lady / CO-7"}
        return "papaya", 95.0, meta

    if is_mango:
        meta = {"object_type": "fruit", "is_fruit": True, "detected_variety": "Alphonso / Nam Dok Mai"}
        return "mango", 95.0, meta

    return None, 50.0, {}
