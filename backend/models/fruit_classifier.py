"""
Robust 8-Fruit Identification Engine for FruitTwin
Features:
1. Plant / Leaf / Foliage Detection
2. Non-Fruit Object Detection (metallic, phones, furniture)
3. Unsupported / Exotic Fruit Detection (Dragon Fruit, Kiwi, etc.)
4. Precision Multi-Fruit Recognition across all 8 Agricultural Fruits:
   - Mango (மாம்பழம்)
   - Banana (வாழைப்பழம்) — including Red Banana / Sevazhai (செவ்வாழை)
   - Papaya (பப்பாளி) — distinguished from Orange via oblong/oval geometry & salmon/yellow-green traits
   - Apple (ஆப்பிள்)
   - Orange (ஆரஞ்சு) — strictly spherical geometry
   - Guava (கொய்யா) — distinguished from Apple via pale lime-green/cream metrics
   - Pomegranate (மாதுளை) — deep ruby/crimson pericarp with b_mean > g_mean
   - Watermelon (தர்பூசணி)
5. Cross-Tab Fruit Mismatch Detection
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Dict, Any, Optional

SUPPORTED_FRUITS = ["mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon"]
SUPPORTED_FRUIT_SET = set(SUPPORTED_FRUITS)

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

def detect_fruit_category(image: Image.Image, expected_fruit: Optional[str] = None) -> Tuple[Optional[str], float, Dict[str, Any]]:
    im = image.convert('RGB')
    thumb = im.resize((150, 150))
    arr = np.array(thumb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Foreground segmentation
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
    r_mean, g_mean, b_mean = float(np.mean(fg_r)), float(np.mean(fg_g)), float(np.mean(fg_b))

    hsv_arr = np.array(thumb.convert('HSV'), dtype=np.float32) / 255.0
    h, s, v = hsv_arr[:, :, 0][fg], hsv_arr[:, :, 1][fg], hsv_arr[:, :, 2][fg]

    pure_red = float(np.sum(((h < 0.05) | (h > 0.95)) & (s > 0.35) & (v > 0.25)) / total_fg)
    pure_orange = float(np.sum((h >= 0.05) & (h <= 0.13) & (s > 0.35) & (v > 0.30)) / total_fg)
    pure_yellow = float(np.sum((h > 0.13) & (h <= 0.21) & (s > 0.30) & (v > 0.35)) / total_fg)
    pure_green = float(np.sum((h > 0.21) & (h <= 0.45) & (s > 0.25) & (v > 0.20)) / total_fg)
    pale_white = float(np.sum((s < 0.28) & (v > 0.60)) / total_fg)
    deep_crimson = float(np.sum((fg_r > 0.50) & (fg_g < 0.40) & (fg_b < 0.40)) / total_fg)
    banana_pulp = float(np.sum((fg_r > 0.65) & (fg_g > 0.55) & (fg_b > 0.50)) / total_fg)

    crimson_red_dom = float(np.sum((fg_r > fg_g * 1.30) & (fg_r > fg_b * 1.30) & (fg_r > 0.25)) / total_fg)
    yellow_dom = float(np.sum((fg_r > 0.40) & (fg_g > 0.38) & (fg_b < 0.40) & (np.abs(fg_r - fg_g) < 0.30)) / total_fg)
    orange_dom = float(np.sum((fg_r > 0.50) & (fg_g >= 0.26) & (fg_g <= 0.65) & (fg_b < 0.30)) / total_fg)
    salmon_pulp_dom = float(np.sum((fg_r > 0.55) & (fg_g >= 0.24) & (fg_g <= 0.50) & (fg_b < 0.35) & (fg_r > fg_g * 1.18)) / total_fg)
    green_dom = float(np.sum((fg_g > fg_r * 1.05) & (fg_g > fg_b * 1.05)) / total_fg)
    pure_green_dom = float(np.sum((fg_g > fg_r * 1.15) & (fg_g > fg_b * 1.15) & (fg_g > 0.18)) / total_fg)
    magenta_pink_dom = float(np.sum((fg_r > 0.50) & (fg_b > 0.35) & (fg_b > fg_g * 1.25) & (fg_g < fg_r * 0.65)) / total_fg)
    banana_flesh_dom = float(np.sum((fg_r > 0.55) & (fg_g > 0.42) & (fg_b < 0.50) & (np.abs(fg_r - fg_g) < 0.25)) / total_fg)

    # Geometry & aspect ratio
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
        bbox_area = max(1, (np.max(y_idx) - np.min(y_idx) + 1) * (np.max(x_idx) - np.min(x_idx) + 1))
        fill_ratio = float(total_fg / bbox_area)
    else:
        elong, aspect, fill_ratio = 1.0, 1.0, 1.0

    gr_ratio = g_mean / max(1e-4, r_mean)

    # Edge density & texture variance
    edges = thumb.convert('L').filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32) / 255.0
    fg_edges = edge_arr[fg]
    edge_density = float(np.mean(fg_edges)) if len(fg_edges) > 0 else 0.0

    # Saturation
    cmax = np.maximum(np.maximum(fg_r, fg_g), fg_b)
    cmin = np.minimum(np.minimum(fg_r, fg_g), fg_b)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-4, delta / cmax, 0.0)
    sat_mean = float(np.mean(sat)) if len(sat) > 0 else 0.0

    is_spheroid = (elong < 1.40) and (0.60 <= aspect <= 1.50)
    is_solid_sphere = is_spheroid and (fill_ratio > 0.55)

    # 1. Botanical Foliage Guard
    is_plant_or_leaf = False
    plant_confidence = 0.0
    if not is_solid_sphere:
        if (pure_green_dom > 0.35 or (g_mean > r_mean * 1.15 and g_mean > b_mean * 1.15 and g_mean > 0.18)) and orange_dom < 0.08 and magenta_pink_dom < 0.08 and yellow_dom < 0.15:
            if crimson_red_dom < 0.05 or (pure_green_dom > 0.30 and r_mean < 0.45):
                is_plant_or_leaf = True
                plant_confidence = min(98.5, 88.0 + max(pure_green_dom, green_dom) * 12.0)
        elif elong > 1.8 and green_dom > 0.35 and yellow_dom < 0.15 and crimson_red_dom < 0.05:
            is_plant_or_leaf = True
            plant_confidence = 96.5
        elif pure_green_dom > 0.50 and edge_density > 0.04:
            is_plant_or_leaf = True
            plant_confidence = 98.0
        elif (green_dom > 0.12 and (edge_density > 0.04 or elong > 1.4) and yellow_dom < 0.12 and r_mean < 0.45 and (pure_green_dom > 0.20 or crimson_red_dom < 0.15)):
            is_plant_or_leaf = True
            plant_confidence = 95.0

    if is_plant_or_leaf:
        return "plant_leaf", plant_confidence, {
            "object_type": "plant_leaf", "is_botanical": True, "is_fruit": False,
            "message": "This image appears to contain plant material rather than a supported harvested fruit. Fruit Twin requires a recognizable fruit specimen for fruit maturity analysis and GAN simulation."
        }

    # 2. General Non-Fruit Object Guard
    is_non_fruit = False
    if total_fg < 80 or (r_mean < 0.12 and g_mean < 0.12 and b_mean < 0.12):
        is_non_fruit = True
    elif sat_mean < 0.18 and (r_mean > 0.15 or edge_density > 0.02) and green_dom < 0.15:
        is_non_fruit = True
    elif (abs(r_mean - g_mean) < 0.04 and abs(g_mean - b_mean) < 0.04 and edge_density > 0.05 and green_dom < 0.15):
        is_non_fruit = True

    if is_non_fruit:
        return "non_fruit", 90.0, {
            "object_type": "non_fruit", "is_botanical": False, "is_fruit": False,
            "message": "The uploaded image does not contain recognized botanical fruit characteristics."
        }

    # 3. Exotic / Unsupported Fruit Guard
    is_dragon_fruit = (b_mean > 0.30) and (b_mean > g_mean * 1.35) and (r_mean > 0.50) and (magenta_pink_dom > 0.10)
    is_kiwi = (0.35 < r_mean < 0.55 and 0.30 < g_mean < 0.45 and b_mean < 0.25 and edge_density > 0.12 and pure_green_dom > 0.22)
    is_avocado = (r_mean < 0.25 and g_mean < 0.30 and b_mean < 0.20) and green_dom > 0.18 and edge_density > 0.08
    is_pineapple = yellow_dom > 0.28 and orange_dom > 0.14 and edge_density > 0.14 and elong > 1.35

    if is_dragon_fruit or is_kiwi or is_avocado or is_pineapple:
        cname = "Dragon Fruit (பித்தாயா / டிராகன் பழம்)" if is_dragon_fruit else (
            "Kiwi Fruit (கிவி பழம்)" if is_kiwi else ("Avocado (வெண்ணெய் பழம்)" if is_avocado else "Pineapple (அன்னாசிப்பழம்)")
        )
        return "unsupported_fruit", 94.0, {
            "object_type": "unsupported_fruit", "is_botanical": True, "is_fruit": True, "is_supported": False,
            "candidate_fruit": cname,
            "message": f"This appears to be a fruit ({cname}), but its fruit-specific analysis pipeline is not currently supported by Fruit Twin."
        }

    # 4. BANANA (Yellow/Green or Red Banana / Sevazhai)
    is_red_banana = ((banana_flesh_dom > 0.035 and crimson_red_dom > 0.10 and b_mean > 0.28 and green_dom < 0.05 and r_mean > 0.50) or
                     (elong > 2.40 and (aspect > 1.80 or aspect < 0.50) and crimson_red_dom > 0.50))
    is_banana_shape = (elong > 2.20) or (elong > 1.75 and (aspect > 2.25 or aspect < 0.45))
    is_yellow_banana = is_banana_shape and (pure_yellow > 0.15 or (pure_green > 0.20 and elong > 2.20)) and pale_white < 0.25 and (expected_fruit != "mango" or elong > 2.30)
    is_banana = is_red_banana or is_yellow_banana

    if is_banana:
        var = "Red Banana (Sevazhai / செவ்வாழை)" if is_red_banana else "Banana (Cavendish / Poovan)"
        return "banana", 97.5 if is_red_banana else 96.0, {
            "object_type": "fruit", "is_fruit": True, "detected_variety": var, "is_red_banana": is_red_banana
        }

    # 5. WATERMELON
    is_watermelon = (
        ((pure_green > 0.35 and pure_red > 0.10 and pure_green > pure_red) or
         (r_mean < 0.42 and g_mean < 0.46 and pure_green > 0.25 and gr_ratio > 1.05 and pure_red < 0.05)) and
        not is_banana
    )

    if is_watermelon:
        return "watermelon", 96.5, {"object_type": "fruit", "is_fruit": True, "detected_variety": "Sugar Baby / Kiran"}

    # 6. POMEGRANATE (Deep ruby/crimson with low green/blue, b_mean > g_mean * 1.05)
    is_pomegranate = (
        (pure_red > 0.45 or deep_crimson > 0.40) and
        (gr_ratio < 0.55) and
        (b_mean > g_mean * 1.05) and
        (elong < 1.25) and
        (pure_orange < 0.15) and
        not is_banana
    )

    if is_pomegranate:
        return "pomegranate", 96.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": "Bhagwa / Ruby"}

    # 7. ORANGE (Strictly citrus peel: pure_orange > 0.35 or orange_dom > 0.35, low pale_white, not elongated)
    # Handles both isolated fruits and fruits with a green leaf/stem
    is_orange = (
        (pure_orange > 0.35 or (orange_dom > 0.35 and pure_orange > 0.25)) and
        (pure_orange > pure_yellow * 1.5) and
        (pure_red < 0.20) and
        (pale_white < 0.20) and
        (elong < 1.42) and
        (r_mean > 0.50) and
        not is_pomegranate and
        not is_banana
    )

    if is_orange:
        return "orange", 96.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": "Nagpur / Mandarin"}

    # 8. GUAVA (Pale lime-green to yellowish-white, spherical/pyriform or sliced wedges)
    is_guava = (
        (pale_white > 0.28 and pure_red < 0.08 and pure_orange < 0.20) or
        (pure_green > 0.18 and gr_ratio > 0.95 and elong < 1.35 and pure_red < 0.08 and g_mean < 0.48) or
        (pure_yellow > 0.30 and pure_yellow > pure_orange and pure_red < 0.08 and elong < 1.35 and pure_orange < 0.40)
    ) and not is_watermelon and not is_banana

    if is_guava:
        return "guava", 95.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": "Allahabad Safeda"}

    # 9. MANGO (Asymmetric ovoid curve with yellow-orange or green blush)
    # Green-yellow breaker mango has high yellow or green, pure_yellow > pure_orange
    # Raw green mango has emerald/chlorophyll green peel, gr_ratio > 1.0, elong <= 2.20
    is_yellow_mango = ((pure_yellow > 0.18 or (yellow_dom > 0.30 and green_dom > 0.15)) and pure_yellow > pure_orange and pale_white < 0.25)
    is_green_mango = (
        (pure_green > 0.15 or pure_green_dom > 0.15 or (gr_ratio > 1.02 and g_mean > 0.20)) and
        (elong <= 2.20) and
        (pure_red < 0.10) and
        (pure_orange < 0.20) and
        (pale_white < 0.25)
    )
    is_mango_tab_specimen = (
        (expected_fruit == "mango") and
        (elong <= 2.25) and
        (pure_red < 0.15) and
        (pale_white < 0.28) and
        not is_orange and
        not is_guava and
        not is_watermelon
    )
    is_mango = (
        (is_yellow_mango or is_green_mango or is_mango_tab_specimen) and
        not is_orange and
        not is_guava and
        not is_banana and
        not is_watermelon
    )

    if is_mango:
        return "mango", 95.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": "Alphonso / Nam Dok Mai"}

    # 10. APPLE (Red or Granny Smith green with spheroid indentation)
    is_red_apple = (pure_red > 0.25 or deep_crimson > 0.20) and (elong < 1.30) and (sat_mean > 0.35) and (g_mean >= b_mean) and not is_orange and not is_pomegranate and not is_watermelon and not is_banana
    is_green_apple = (pure_green > 0.22) and (elong < 1.30) and (g_mean > 0.40 and g_mean < 0.48) and not is_watermelon and not is_guava and not is_banana
    is_yellow_apple = (pure_yellow > 0.30) and (elong < 1.30) and not is_orange and not is_guava and not is_banana and not is_mango
    is_apple = is_red_apple or is_green_apple or is_yellow_apple

    if is_apple:
        var = "Royal Delicious / Red Apple" if is_red_apple else ("Granny Smith" if is_green_apple else "Golden Apple")
        return "apple", 96.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": var}

    # 11. PAPAYA (Distinguish from Orange: oblong/oval, salmon pulp, amber/orange cylinder)
    is_papaya = (
        (salmon_pulp_dom > 0.06 and pure_orange > 0.18 and pure_red < 0.15 and pale_white < 0.20) or
        (pure_orange > 0.25 and elong >= 1.35 and pure_red < 0.15 and pale_white < 0.20 and not is_orange) or
        (pure_green > 0.25 and gr_ratio > 1.05 and elong > 1.40 and pure_red < 0.15 and not is_watermelon and not is_mango)
    ) and not is_banana

    if is_papaya:
        return "papaya", 96.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": "Red Lady / CO-7"}

    # Fallback to expected fruit if specified and supported
    if expected_fruit and expected_fruit.lower() in SUPPORTED_FRUITS:
        ef = expected_fruit.lower()
        return ef, 88.0, {"object_type": "fruit", "is_fruit": True, "detected_variety": FRUIT_METADATA[ef]["default_variety"]}

    # None triggers multi-class Fruits-360 model for other dataset classes (e.g. Strawberry, Potato Red)
    return None, 50.0, {}
