"""
Universal Multi-Stage Ripeness Prediction Engine for FruitTwin
Calibrated across all 8 core agricultural fruits:
- Mango (மாம்பழம்)
- Banana (வாழைப்பழம்) - with dedicated handling for Red Banana (Sevazhai / செவ்வாழை)
- Papaya (பப்பாளி)
- Apple (ஆப்பிள்)
- Orange (ஆரஞ்சு)
- Guava (கொய்யா)
- Pomegranate (மாதுளை)
- Watermelon (தர்பூசணி)

Supports 4-stage classification: 'Unripe', 'Semi-Ripe', 'Ripe', 'Overripe'
Returns calibrated percentage probabilities, color metrics, and ripeness_index (0.0 to 1.0).
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Dict, Any, Tuple, Optional


def extract_optical_metrics(image: Image.Image) -> Dict[str, Any]:
    """
    Extracts foreground colorimetric (RGB & HSV), textural, and morphological metrics.
    Preserves original image physical aspect ratio.
    """
    orig_w, orig_h = image.size
    aspect_scale = float(orig_w) / max(1, orig_h)

    thumb = image.resize((128, 128))
    arr = np.array(thumb, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Background filtering (white backdrop, dark shadows, neutral gray table)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    is_dark = (brightness < 0.10) | ((r < 0.10) & (g < 0.10) & (b < 0.15))
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
    bright_mean = float(np.mean(brightness[fg]))
    gr_ratio = g_mean / max(1e-4, r_mean)
    rg_ratio = r_mean / max(1e-4, g_mean)

    # HSV Representation
    hsv_im = thumb.convert("HSV")
    hsv_arr = np.array(hsv_im, dtype=np.float32) / 255.0
    h = hsv_arr[:, :, 0][fg]
    s = hsv_arr[:, :, 1][fg]
    v = hsv_arr[:, :, 2][fg]

    cmax = np.maximum(np.maximum(fg_r, fg_g), fg_b)
    cmin = np.minimum(np.minimum(fg_r, fg_g), fg_b)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-4, delta / cmax, 0.0)
    sat_mean = float(np.mean(sat)) if len(sat) > 0 else 0.0

    # Color space dominances
    pure_red = float(np.sum(((h < 0.05) | (h > 0.95)) & (s > 0.30) & (v > 0.20)) / total_fg)
    pure_orange = float(np.sum((h >= 0.05) & (h <= 0.13) & (s > 0.32) & (v > 0.25)) / total_fg)
    pure_yellow = float(np.sum((h > 0.13) & (h <= 0.21) & (s > 0.28) & (v > 0.30)) / total_fg)
    pure_green = float(np.sum((h > 0.21) & (h <= 0.45) & (s > 0.22) & (v > 0.18)) / total_fg)
    pale_white = float(np.sum((s < 0.25) & (v > 0.60)) / total_fg)
    deep_crimson = float(np.sum((fg_r > 0.45) & (fg_r > fg_g * 1.35) & (fg_r > fg_b * 1.35)) / total_fg)

    # Senescence / dark necrotic patches (browning / soft decay)
    dark_senescence = float(np.sum((v < 0.22) | ((fg_r > fg_g) & (fg_g > fg_b) & (v < 0.35) & (s < 0.45))) / total_fg)

    # Texture & surface gradient
    gray_im = thumb.convert("L")
    edges = gray_im.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32) / 255.0
    fg_edges = edge_arr[fg]
    edge_density = float(np.mean(fg_edges)) if len(fg_edges) > 0 else 0.0

    # Spatial geometry
    y_idx, x_idx = np.where(fg)
    if len(y_idx) > 10:
        x_scaled = x_idx * aspect_scale
        cov = np.cov(np.vstack([y_idx, x_scaled]))
        evals = np.sort(np.abs(np.linalg.eigvals(cov)))
        elong = float(np.sqrt(evals[-1] / max(1e-5, evals[0])))
        y_span = np.max(y_idx) - np.min(y_idx) + 1
        x_span = np.max(x_scaled) - np.min(x_scaled) + 1
        aspect = float(x_span / max(1, y_span))
    else:
        elong = 1.0
        aspect = 1.0

    return {
        "r_mean": round(r_mean, 3),
        "g_mean": round(g_mean, 3),
        "b_mean": round(b_mean, 3),
        "brightness": round(bright_mean, 3),
        "sat_mean": round(sat_mean, 3),
        "gr_ratio": round(gr_ratio, 3),
        "rg_ratio": round(rg_ratio, 3),
        "pure_green": round(pure_green, 3),
        "pure_yellow": round(pure_yellow, 3),
        "pure_orange": round(pure_orange, 3),
        "pure_red": round(pure_red, 3),
        "deep_crimson": round(deep_crimson, 3),
        "pale_white": round(pale_white, 3),
        "dark_senescence": round(dark_senescence, 3),
        "edge_density": round(edge_density, 3),
        "elong": round(elong, 3),
        "aspect": round(aspect, 3),
    }


def _softmax(scores: np.ndarray, temp: float = 1.0) -> np.ndarray:
    exps = np.exp((scores - np.max(scores)) / temp)
    return exps / np.sum(exps)


def predict_multi_fruit_ripeness(image: Image.Image, fruit: str = "mango") -> Dict[str, Any]:
    """
    Universal multi-stage ripeness prediction engine for all 8 core fruits:
    - Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, Watermelon.

    Returns:
    {
      "fruit": fruit,
      "stage": "Ripe" | "Semi-Ripe" | "Unripe" | "Overripe",
      "confidence": float,
      "probabilities": {"Unripe": float, "Semi-Ripe": float, "Ripe": float, "Overripe": float},
      "color_metrics": {...},
      "ripeness_index": float (0.0 to 1.0)
    }
    """
    fruit = fruit.lower().strip()
    m = extract_optical_metrics(image)

    p_green = m["pure_green"]
    p_yellow = m["pure_yellow"]
    p_orange = m["pure_orange"]
    p_red = m["pure_red"]
    deep_crim = m["deep_crimson"]
    r_m = m["r_mean"]
    g_m = m["g_mean"]
    b_m = m["b_mean"]
    gr = m["gr_ratio"]
    rg = m["rg_ratio"]
    edge = m["edge_density"]
    sen = m["dark_senescence"]
    bright = m["brightness"]

    # Scores: [Unripe, Semi-Ripe, Ripe, Overripe]
    raw_scores = np.zeros(4, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 1. BANANA (Cavendish, Poovan, Nendran & Red Banana / Sevazhai)
    # -------------------------------------------------------------------------
    if fruit == "banana":
        # Check if Red Banana (Sevazhai)
        is_red_banana = (p_red > 0.20 or deep_crim > 0.15 or (r_m > g_m * 1.25 and b_m > 0.25))

        if is_red_banana:
            # Red Banana (Sevazhai): Unripe (purplish-green) -> Semi-Ripe -> Ripe (deep burgundy) -> Overripe
            unripe_w = p_green * 4.0 + (1.0 if g_m > r_m * 0.9 else 0.0)
            semi_w = (1.5 if 0.10 <= p_red <= 0.35 else 0.5) + (1.0 if p_green > 0.05 else 0.0)
            ripe_w = (deep_crim * 4.0 + p_red * 2.5) if (edge < 0.08 and sen < 0.15) else 1.0
            overripe_w = sen * 5.0 + (edge > 0.07) * 3.0 + (bright < 0.40) * 1.5
            raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])
        else:
            # Standard Cavendish / Yellow Banana:
            has_banana_breaker = (p_green > 0.06 and (p_yellow > 0.08 or p_orange > 0.08))
            if has_banana_breaker:
                semi_w = 5.2 + p_yellow * 2.0
                unripe_w = p_green * 2.0
                ripe_w = p_yellow * 1.8
            else:
                # Unripe: Deep chlorophyll green dominance (pure_green > 0.25, g_mean > r_mean * 1.05)
                unripe_w = p_green * 4.5 + (2.5 if (g_m > r_m * 1.05 and p_green > 0.20) else 0.0)
                # Semi-Ripe baseline
                semi_w = 1.0
                # Ripe: Bright yellow (pure_yellow > 0.22)
                ripe_w = p_yellow * 4.5 + p_orange * 2.0 + (2.0 if (rg > 1.15 and p_green < 0.08) else 0.0)
            # Overripe: Sugar spots / senescence flecks or peel browning
            overripe_w = sen * 5.5 + (edge > 0.065) * 3.0 + (1.5 if bright < 0.42 and p_yellow > 0.10 else 0.0)
            raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 2. MANGO (Alphonso, Banganapalli, Neelum, Sindhoora)
    # -------------------------------------------------------------------------
    elif fruit == "mango":
        is_sindhoora = (p_red > 0.15 and r_m > g_m * 1.10)
        # Breaker stage: green and yellow co-exist, or peel still has green tones (gr > 0.94 or rg < 1.08 or p_green > 0.03)
        # with developing yellow/orange shoulders (p_yellow > 0.12 or p_orange > 0.10 or (p_yellow + p_orange) > 0.14)
        has_mango_breaker = (
            (p_yellow > 0.12 or p_orange > 0.10 or (p_yellow + p_orange) > 0.14) and
            (p_green > 0.03 or gr > 0.94 or rg < 1.08)
        )
        
        if has_mango_breaker:
            # Green mango with yellow shoulders evaluates as Semi-Ripe (~2.5-3.5 days to peak ripe at 28°C)
            semi_w = 5.5 + (p_yellow + p_orange) * 2.0
            unripe_w = (1.5 if gr > 1.0 else 0.5) + p_green * 2.0
            ripe_w = 1.0 + (p_orange * 2.0) + (p_red * 2.0)
        else:
            # Unripe: Dark green, firm peel (gr_ratio > 1.05)
            unripe_w = p_green * 4.5 + (3.0 if gr > 1.05 else 0.0)
            semi_w = 1.0
            # Ripe: Vibrant yellow, saffron, or reddish blush (Sindhoora)
            if is_sindhoora:
                ripe_w = p_red * 3.5 + p_yellow * 2.5 + p_orange * 2.5 + (2.0 if edge < 0.075 else 0.0)
            else:
                ripe_w = p_yellow * 4.0 + p_orange * 3.0 + (2.5 if (rg > 1.10 and p_green < 0.08) else 0.0)
        # Overripe: Dark senescence patches, peel wrinkling
        overripe_w = sen * 5.0 + (edge > 0.075) * 2.5 + (1.5 if bright < 0.42 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 3. PAPAYA (Red Lady, Co-7, Washington)
    # -------------------------------------------------------------------------
    elif fruit == "papaya":
        has_papaya_breaker = (p_green > 0.08 and (p_yellow + p_orange >= 0.14))
        if has_papaya_breaker:
            semi_w = 5.2 + (p_yellow + p_orange) * 2.0
            unripe_w = p_green * 2.0
            ripe_w = (p_yellow + p_orange) * 1.8
        else:
            # Unripe: Solid dark green peel
            unripe_w = p_green * 4.5 + (3.0 if (p_green > 0.30 and p_yellow < 0.15) else 0.0)
            semi_w = 1.0
            # Ripe: >75% yellow-orange coverage
            ripe_w = (p_orange * 3.5 + p_yellow * 3.0) + (2.5 if (p_orange + p_yellow > 0.45 and p_green < 0.08) else 0.0)
        # Overripe: Softening, dark brown necrotic circular spots
        overripe_w = sen * 5.0 + (edge > 0.075) * 3.0 + (1.5 if bright < 0.45 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 4. APPLE (Royal Delicious / Kashmiri Ambri / Granny Smith)
    # -------------------------------------------------------------------------
    elif fruit == "apple":
        is_granny_smith = (p_green > 0.35 and p_red < 0.10)
        if is_granny_smith:
            unripe_w = 4.0 if (p_green > 0.45 and bright < 0.50) else 1.0
            semi_w = 3.0 if (p_green > 0.30 and bright >= 0.50 and p_yellow < 0.15) else 1.0
            ripe_w = 4.0 if (p_green > 0.30 and (p_yellow > 0.12 or bright >= 0.55)) else 1.2
            overripe_w = sen * 5.0 + (edge > 0.07) * 3.0
        else:
            # Red Apple (Royal Delicious / Red Chief / Crimson Snow):
            # Unripe: Pale green / starchy dull background
            unripe_w = p_green * 4.0 + (3.0 if (p_green > 0.25 and p_red < 0.15) else 0.0)
            # Semi-Ripe: Partial red/yellow blush development
            semi_w = 3.2 if (0.10 <= p_red <= 0.35 and p_green > 0.08) else (1.5 if (p_yellow > 0.15 and p_red < 0.30) else 0.3)
            # Ripe: Deep crimson red (Royal Delicious: pure_red > 0.35 or deep_crimson > 0.25)
            ripe_w = (p_red * 3.5 + deep_crim * 3.0) + (2.5 if (p_red > 0.35 and edge < 0.075) else 0.0)
            # Overripe: Dull wrinkled peel, bruising, loss of gloss
            overripe_w = sen * 5.0 + (edge > 0.075) * 3.0 + (1.5 if bright < 0.40 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 5. ORANGE (Nagpur / Mandarin / Mosambi)
    # -------------------------------------------------------------------------
    elif fruit == "orange":
        # Unripe: Greenish-yellow rind
        unripe_w = p_green * 4.5 + (3.0 if (gr > 0.95 and p_orange < 0.15) else 0.0)
        # Semi-Ripe: Pale yellow-orange rind
        semi_w = 3.5 if (p_green > 0.08 and (p_orange > 0.12 or p_yellow > 0.18)) else (1.5 if (0.12 <= p_orange <= 0.28) else 0.3)
        # Ripe: Full deep citrus orange (pure_orange > 0.30)
        ripe_w = p_orange * 4.5 + (3.0 if (p_orange > 0.28 and p_green < 0.08) else (p_orange * 2.0))
        # Overripe: Brown peel softening, desiccated rind
        overripe_w = sen * 5.0 + (edge > 0.08) * 3.0 + (1.5 if bright < 0.42 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 6. GUAVA (Allahabad Safeda / Lucknow 49)
    # -------------------------------------------------------------------------
    elif fruit == "guava":
        # Unripe: Deep green, hard texture
        unripe_w = p_green * 4.5 + (3.0 if (p_green > 0.30 and p_yellow < 0.10) else 0.0)
        # Semi-Ripe: Light green to yellowish-green transition
        semi_w = 3.5 if (p_green > 0.15 and (p_yellow > 0.10 or m["pale_white"] > 0.15)) else 1.0
        # Ripe: Pale greenish-yellow / cream with aromatic blush
        ripe_w = (m["pale_white"] * 3.0 + p_yellow * 3.5) + (2.5 if (p_yellow > 0.18 and p_green < 0.22) else 0.0)
        # Overripe: Pulp browning, soft surface abrasions
        overripe_w = sen * 5.0 + (edge > 0.075) * 3.0 + (1.5 if bright < 0.40 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 7. POMEGRANATE (Bhagwa / Ruby / Ganesh)
    # -------------------------------------------------------------------------
    elif fruit == "pomegranate":
        # Unripe: Pale yellowish-pink or green tint, round shape
        unripe_w = p_green * 4.0 + (3.0 if (p_green > 0.10 or (p_yellow > 0.20 and p_red < 0.18)) else 0.0)
        # Semi-Ripe: Pinkish-red coloration with developing ridges
        semi_w = 3.2 if (0.18 <= p_red <= 0.40 and deep_crim < 0.25) else 1.0
        # Ripe: Deep ruby/crimson pericarp with flattened hexagonal sides
        ripe_w = (p_red * 3.5 + deep_crim * 4.0) + (3.0 if (p_red > 0.45 or deep_crim > 0.35) else 0.0)
        # Overripe: Hardened brown aril cracking, dark skin discoloration
        overripe_w = sen * 5.0 + (edge > 0.08) * 3.0 + (1.5 if bright < 0.38 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # -------------------------------------------------------------------------
    # 8. WATERMELON (Sugar Baby / Kiran)
    # -------------------------------------------------------------------------
    elif fruit == "watermelon":
        # Unripe: White/pale ground spot, dull stripes
        unripe_w = (m["pale_white"] * 3.5) + (2.5 if (m["pale_white"] > 0.22 and p_yellow < 0.08) else 0.8)
        # Semi-Ripe: Faint yellow ground spot
        semi_w = 3.0 if (0.06 <= p_yellow <= 0.14 and p_green > 0.20) else 1.2
        # Ripe: Deep butter-yellow ground spot, dull rind, crisp dark stripes
        ripe_w = (p_green * 3.0 + p_yellow * 3.0) + (2.5 if (p_green > 0.25 and (p_yellow > 0.12 or bright > 0.40)) else 0.0)
        # Overripe: Hollow spongy texture, brownish-yellow rind decay
        overripe_w = sen * 4.5 + (edge > 0.08) * 3.0 + (1.5 if bright < 0.35 else 0.0)
        raw_scores = np.array([unripe_w, semi_w, ripe_w, overripe_w])

    # Fallback default
    else:
        raw_scores = np.array([p_green * 3.0, (p_green + p_yellow) * 1.5, (p_yellow + p_orange + p_red) * 2.5, sen * 3.0])

    # Ensure baseline positive mass
    raw_scores = np.maximum(raw_scores, 0.05)

    # Convert to calibrated probabilities via temperature-scaled softmax
    probs = _softmax(raw_scores, temp=0.85)

    stage_names = ["Unripe", "Semi-Ripe", "Ripe", "Overripe"]
    pred_idx = int(np.argmax(probs))
    pred_stage = stage_names[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100.0, 1)

    prob_dict = {
        "Unripe": round(float(probs[0]) * 100.0, 1),
        "Semi-Ripe": round(float(probs[1]) * 100.0, 1),
        "Ripe": round(float(probs[2]) * 100.0, 1),
        "Overripe": round(float(probs[3]) * 100.0, 1),
    }

    # Continuous Ripeness Index (0.0 = completely unripe, 1.0 = completely overripe)
    # Stage weights: Unripe=0.15, Semi-Ripe=0.50, Ripe=0.82, Overripe=0.98
    stage_centers = np.array([0.15, 0.50, 0.82, 0.98], dtype=np.float32)
    ripeness_index = round(float(np.dot(probs, stage_centers)), 3)

    return {
        "fruit": fruit,
        "stage": pred_stage,
        "confidence": confidence,
        "probabilities": prob_dict,
        "color_metrics": m,
        "ripeness_index": ripeness_index,
    }
