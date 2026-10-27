from PIL import Image
import numpy as np

def detect_fruit(path):
    im = Image.open(path).convert('RGB')
    arr = np.array(im.resize((150, 150)), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    fg = ((r > 0.12) | (g > 0.12) | (b > 0.12)) & ~((r > 0.96) & (g > 0.96) & (b > 0.96))
    if np.sum(fg) < 200:
        fg = np.ones_like(r, dtype=bool)

    fg_r, fg_g, fg_b = r[fg], g[fg], b[fg]
    total_fg = len(fg_r)

    r_mean = float(np.mean(fg_r))
    g_mean = float(np.mean(fg_g))
    b_mean = float(np.mean(fg_b))

    red_dom = float(np.sum((fg_r > fg_g * 1.15) & (fg_r > fg_b * 1.15) & (fg_r > 0.22)) / total_fg)
    yellow_dom = float(np.sum((fg_r > 0.42) & (fg_g > 0.42) & (fg_b < 0.45) & (np.abs(fg_r - fg_g) < 0.18)) / total_fg)
    green_dom = float(np.sum((fg_g > fg_r * 1.05) & (fg_g > fg_b * 1.05)) / total_fg)

    # Spatial elongation
    y_idx, x_idx = np.where(fg)
    if len(y_idx) > 10:
        cov = np.cov(np.vstack([y_idx, x_idx]))
        evals = np.linalg.eigvals(cov)
        evals = np.sort(np.abs(evals))
        elong = float(np.sqrt(evals[-1] / max(1e-5, evals[0])))
    else:
        elong = 1.0

    # Rule-based fruit category detection:
    # 1. Classic Yellow Banana:
    if yellow_dom > 0.30 and abs(r_mean - g_mean) < 0.10 and b_mean < 0.35:
        return "banana", 96.0

    # 2. Classic Green Banana:
    if green_dom > 0.35 and elong > 1.3:
        return "banana", 95.0

    # 3. Red Banana vs Red Mango / Apple:
    if red_dom > 0.15:
        # Check if Red Mango: warm orange-yellow undertones (yellow_dom > 0.08 or r_mean > g_mean * 1.35)
        if (yellow_dom > 0.08 or r_mean > g_mean * 1.35) and r_mean > 0.50:
            return "mango", 95.0
        # If pure crimson red with very low yellow and spherical:
        elif elong < 1.15 and b_mean < 0.25:
            return "apple", 92.0
        # Else if dark maroon tones with low yellow:
        elif elong > 1.35:
            return "banana", 94.0
        else:
            return "mango", 85.0

    # 4. Standard Ripe Mango (Alphonso, Banganapalli):
    if r_mean > 0.65 and g_mean > 0.45 and yellow_dom > 0.15:
        return "mango", 94.0

    # 5. Apple (Crimson or Granny Smith):
    if elong < 1.20 and (red_dom > 0.40 or green_dom > 0.40):
        return "apple", 91.0

    return "mango", 80.0

p_b = r"C:/Users/JEFRI/.gemini/antigravity/brain/8704a43d-8bc3-4f2f-934b-888b3e99f73a/.user_uploaded/media_1791561159886.png"
p_m = r"C:\Users\JEFRI\.gemini\antigravity\scratch\fruittwin\test_red_mango.jpg"

print("Detect fruit for Banana image:", detect_fruit(p_b))
print("Detect fruit for Mango image:", detect_fruit(p_m))
