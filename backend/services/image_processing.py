import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

def process_uploaded_image(file_bytes: bytes) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Decodes uploaded image bytes into Pillow Image object and extracts color/visual features,
    with intelligent foreground fruit segmentation to isolate fruit skin from container/background borders.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # Resize thumbnail for feature analysis
    img_resized = img.resize((150, 150))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Foreground mask: filter out dark background (< 0.10), pure white (> 0.94),
    # and neutral gray/checkerboard backgrounds (|r-g| < 0.06 and |g-b| < 0.06 and r > 0.35)
    is_dark = (r < 0.10) & (g < 0.10) & (b < 0.10)
    is_white = (r > 0.94) & (g > 0.94) & (b > 0.94)
    is_neutral_gray = (np.abs(r - g) < 0.06) & (np.abs(g - b) < 0.06) & (r > 0.35)
    
    fg_mask = ~is_dark & ~is_white & ~is_neutral_gray
    if np.sum(fg_mask) < 200:
        fg_mask = ~is_dark & ~is_white
    if np.sum(fg_mask) < 200:
        fg_mask = np.ones_like(r, dtype=bool)
        
    fg_r = r[fg_mask]
    fg_g = g[fg_mask]
    fg_b = b[fg_mask]
    total_fg_pixels = len(fg_r)
    
    # Calculate color dominance on foreground fruit pixels
    green_mask = (fg_g > fg_r * 1.05) & (fg_g > fg_b * 1.05)
    yellow_mask = (fg_r > 0.42) & (fg_g > 0.42) & (fg_b < 0.45) & (abs(fg_r - fg_g) < 0.18)
    # Red dominance: distinct anthocyanin/carotenoid pigmentation (Red Banana, Red Apple)
    red_mask = (fg_r > fg_g * 1.15) & (fg_r > fg_b * 1.15) & (fg_r > 0.22)
    
    green_ratio = float(np.sum(green_mask) / total_fg_pixels)
    yellow_ratio = float(np.sum(yellow_mask) / total_fg_pixels)
    red_ratio = float(np.sum(red_mask) / total_fg_pixels)
    
    r_mean = float(np.mean(fg_r)) * 100.0
    g_mean = float(np.mean(fg_g)) * 100.0
    b_mean = float(np.mean(fg_b)) * 100.0
    brightness = float(np.mean(arr))
    
    features = {
        "green_ratio": round(green_ratio, 3),
        "yellow_ratio": round(yellow_ratio, 3),
        "red_ratio": round(red_ratio, 3),
        "brightness": round(brightness, 3),
        "R_fg": round(r_mean, 2),
        "G_fg": round(g_mean, 2),
        "B_fg": round(b_mean, 2)
    }
    
    return img, features
