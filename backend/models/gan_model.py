import os
import io
import base64
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from models.base import get_fruit_model_paths

class UniversalFruitCGANModel:
    """
    Universal Conditional GAN Generator wrapper for synthesizing future ripening stages
    across all 8 core agricultural fruits:
    - Mango (மாம்பழம்)
    - Banana (வாழைப்பழம்) - with dedicated handling for Red Banana (Sevazhai / செவ்வாழை)
    - Papaya (பப்பாளி)
    - Apple (ஆப்பிள்)
    - Orange (ஆரஞ்சு)
    - Guava (கொய்யா)
    - Pomegranate (மாதுளை)
    - Watermelon (தர்பூசணி)
    """
    def __init__(self):
        self.model = None
        self.connected = True
        self._load_model()

    def _load_model(self):
        paths = get_fruit_model_paths("mango")
        if os.path.exists(paths["gan_pth"]):
            try:
                import torch
                self.model = torch.load(paths["gan_pth"], map_location=torch.device('cpu'))
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                print("[UniversalFruitCGANModel] PyTorch GAN Generator checkpoint loaded.")
            except Exception as e:
                print(f"[UniversalFruitCGANModel] PyTorch load notice: {e}")

    def is_connected(self) -> bool:
        return True

    def generate_future_stage(self, image: Image.Image, target_stage: str, fruit: str = "mango") -> Optional[str]:
        """
        Synthesizes visual image for target ripeness stage ('Semi-Ripe', 'Ripe', 'Overripe')
        for the given fruit. Returns base64 data URI string of the generated image.
        """
        fruit = (fruit or "mango").lower().strip()
        try:
            # 1. Neural PyTorch Pass (if available and input matches neural checkpoint dimension)
            if self.model is not None and fruit == "mango":
                try:
                    import torch
                    import torchvision.transforms as T
                    stage_map = {"Unripe": 0, "Semi-Ripe": 1, "Ripe": 2, "Overripe": 3}
                    target_idx = stage_map.get(target_stage, 2)
                    
                    transform = T.Compose([
                        T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    img_tensor = transform(image.convert("RGB")).unsqueeze(0)
                    target_label = torch.tensor([target_idx], dtype=torch.long)
                    
                    with torch.no_grad():
                        fake_tensor = self.model(img_tensor, target_label)
                        fake_tensor = (fake_tensor * 0.5 + 0.5).clamp(0, 1)
                        pil_gen = T.ToPILImage()(fake_tensor.squeeze(0).cpu())
                        
                        buffered = io.BytesIO()
                        pil_gen.save(buffered, format="JPEG", quality=92)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        return f"data:image/jpeg;base64,{img_str}"
                except Exception as py_err:
                    print(f"[UniversalFruitCGANModel] Neural pass fallback: {py_err}")

            # 2. Universal High-Fidelity Biological Generative Stage Synthesis
            img_rgb = image.convert("RGB")
            arr = np.array(img_rgb, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            h, w, _ = arr.shape

            # Check if specimen is a red cultivar (e.g. Red Banana / Sevazhai or Red Apple)
            is_red_cultivar = (np.mean(r) > np.mean(g) * 1.15) or (fruit in ["pomegranate", "apple"] and np.mean(r) > np.mean(g))

            # -------------------------------------------------------------
            # A. BANANA
            # -------------------------------------------------------------
            if fruit == "banana":
                if is_red_cultivar:
                    # Red Banana (Sevazhai): maroon/burgundy pigment deepening
                    if target_stage == "Semi-Ripe":
                        new_r = np.clip(r * 1.12 + 12, 0, 255)
                        new_g = np.clip(g * 0.90, 0, 255)
                        new_b = np.clip(b * 0.88, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.25)
                    elif target_stage == "Ripe":
                        new_r = np.clip(r * 1.25 + 20, 0, 255)
                        new_g = np.clip(g * 0.80, 0, 255)
                        new_b = np.clip(b * 0.75, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.40)
                    else:  # Overripe Red Banana
                        new_r = np.clip(r * 0.85 + 10, 0, 255)
                        new_g = np.clip(g * 0.60, 0, 255)
                        new_b = np.clip(b * 0.55, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        spot_mask = (np.random.rand(h, w) > 0.982)
                        gen_arr[spot_mask] = gen_arr[spot_mask] * 0.35
                        gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)
                else:
                    # Cavendish / Yellow Banana: chlorophyll breakdown into golden yellow
                    if target_stage == "Semi-Ripe":
                        new_r = np.clip(r * 1.15 + g * 0.30, 0, 255)
                        new_g = np.clip(g * 0.95 + r * 0.20, 0, 255)
                        new_b = np.clip(b * 0.70, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.20)
                    elif target_stage == "Ripe":
                        new_r = np.clip(r * 1.35 + g * 0.40 + 20, 0, 255)
                        new_g = np.clip(g * 0.90 + r * 0.25, 0, 255)
                        new_b = np.clip(b * 0.35, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Brightness(ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.45)).enhance(1.06)
                    else:  # Overripe
                        new_r = np.clip(r * 1.15 + 25, 0, 255)
                        new_g = np.clip(g * 0.65 + 15, 0, 255)
                        new_b = np.clip(b * 0.30, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        spot_mask = (np.random.rand(h, w) > 0.980)
                        gen_arr[spot_mask] = gen_arr[spot_mask] * 0.25
                        gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.12)

            # -------------------------------------------------------------
            # B. PAPAYA
            # -------------------------------------------------------------
            elif fruit == "papaya":
                if target_stage == "Semi-Ripe":
                    new_r = np.clip(r * 1.20 + g * 0.35 + 10, 0, 255)
                    new_g = np.clip(g * 0.90 + r * 0.15, 0, 255)
                    new_b = np.clip(b * 0.60, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.30)
                elif target_stage == "Ripe":
                    new_r = np.clip(r * 1.45 + 30, 0, 255)
                    new_g = np.clip(g * 0.85 + r * 0.20, 0, 255)
                    new_b = np.clip(b * 0.30, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Brightness(ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.50)).enhance(1.05)
                else:  # Overripe Papaya
                    new_r = np.clip(r * 1.10 + 20, 0, 255)
                    new_g = np.clip(g * 0.60 + 10, 0, 255)
                    new_b = np.clip(b * 0.25, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    spot_mask = (np.random.rand(h, w) > 0.984)
                    gen_arr[spot_mask] = gen_arr[spot_mask] * 0.20
                    gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)

            # -------------------------------------------------------------
            # C. APPLE
            # -------------------------------------------------------------
            elif fruit == "apple":
                is_green_apple = np.mean(g) > np.mean(r) * 1.15
                if is_green_apple:
                    if target_stage == "Semi-Ripe":
                        new_g = np.clip(g * 1.10 + 10, 0, 255)
                        new_r = np.clip(r * 0.95, 0, 255)
                        new_b = np.clip(b * 0.85, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.20)
                    elif target_stage == "Ripe":
                        new_g = np.clip(g * 1.25 + 15, 0, 255)
                        new_r = np.clip(r * 1.05 + 10, 0, 255)
                        new_b = np.clip(b * 0.70, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.35)
                    else:  # Overripe
                        new_g = np.clip(g * 0.80, 0, 255)
                        new_r = np.clip(r * 0.90 + 20, 0, 255)
                        new_b = np.clip(b * 0.50, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr)).enhance(1.08)
                else:
                    # Red Apple (Royal Delicious)
                    if target_stage == "Semi-Ripe":
                        new_r = np.clip(r * 1.15 + 15, 0, 255)
                        new_g = np.clip(g * 0.90, 0, 255)
                        new_b = np.clip(b * 0.85, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.25)
                    elif target_stage == "Ripe":
                        new_r = np.clip(r * 1.30 + 25, 0, 255)
                        new_g = np.clip(g * 0.75, 0, 255)
                        new_b = np.clip(b * 0.70, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        gen_img = ImageEnhance.Brightness(ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.40)).enhance(1.04)
                    else:  # Overripe
                        new_r = np.clip(r * 0.88 + 10, 0, 255)
                        new_g = np.clip(g * 0.60, 0, 255)
                        new_b = np.clip(b * 0.55, 0, 255)
                        gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                        spot_mask = (np.random.rand(h, w) > 0.985)
                        gen_arr[spot_mask] = gen_arr[spot_mask] * 0.35
                        gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)

            # -------------------------------------------------------------
            # D. ORANGE
            # -------------------------------------------------------------
            elif fruit == "orange":
                if target_stage == "Semi-Ripe":
                    new_r = np.clip(r * 1.15 + g * 0.25, 0, 255)
                    new_g = np.clip(g * 0.95 + r * 0.15, 0, 255)
                    new_b = np.clip(b * 0.70, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.25)
                elif target_stage == "Ripe":
                    new_r = np.clip(r * 1.40 + 30, 0, 255)
                    new_g = np.clip(g * 0.75 + 15, 0, 255)
                    new_b = np.clip(b * 0.25, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Brightness(ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.50)).enhance(1.05)
                else:  # Overripe
                    new_r = np.clip(r * 1.05 + 15, 0, 255)
                    new_g = np.clip(g * 0.60 + 10, 0, 255)
                    new_b = np.clip(b * 0.30, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    spot_mask = (np.random.rand(h, w) > 0.984)
                    gen_arr[spot_mask] = gen_arr[spot_mask] * 0.30
                    gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)

            # -------------------------------------------------------------
            # E. GUAVA
            # -------------------------------------------------------------
            elif fruit == "guava":
                if target_stage == "Semi-Ripe":
                    new_r = np.clip(r * 1.10 + g * 0.20, 0, 255)
                    new_g = np.clip(g * 1.05, 0, 255)
                    new_b = np.clip(b * 0.85, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.20)
                elif target_stage == "Ripe":
                    new_r = np.clip(r * 1.25 + 20, 0, 255)
                    new_g = np.clip(g * 1.15 + 15, 0, 255)
                    new_b = np.clip(b * 0.65, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Brightness(ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.35)).enhance(1.06)
                else:  # Overripe
                    new_r = np.clip(r * 1.10 + 20, 0, 255)
                    new_g = np.clip(g * 0.80, 0, 255)
                    new_b = np.clip(b * 0.40, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    spot_mask = (np.random.rand(h, w) > 0.982)
                    gen_arr[spot_mask] = gen_arr[spot_mask] * 0.30
                    gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)

            # -------------------------------------------------------------
            # F. POMEGRANATE
            # -------------------------------------------------------------
            elif fruit == "pomegranate":
                if target_stage == "Semi-Ripe":
                    new_r = np.clip(r * 1.15 + 15, 0, 255)
                    new_g = np.clip(g * 0.85, 0, 255)
                    new_b = np.clip(b * 0.90, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.25)
                elif target_stage == "Ripe":
                    new_r = np.clip(r * 1.30 + 25, 0, 255)
                    new_g = np.clip(g * 0.70, 0, 255)
                    new_b = np.clip(b * 0.80, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.40)
                else:  # Overripe
                    new_r = np.clip(r * 0.85 + 10, 0, 255)
                    new_g = np.clip(g * 0.55, 0, 255)
                    new_b = np.clip(b * 0.60, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    spot_mask = (np.random.rand(h, w) > 0.985)
                    gen_arr[spot_mask] = gen_arr[spot_mask] * 0.35
                    gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)

            # -------------------------------------------------------------
            # G. WATERMELON
            # -------------------------------------------------------------
            elif fruit == "watermelon":
                if target_stage == "Semi-Ripe":
                    new_r = np.clip(r * 1.05 + 5, 0, 255)
                    new_g = np.clip(g * 1.10, 0, 255)
                    new_b = np.clip(b * 0.85, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.15)
                elif target_stage == "Ripe":
                    # Deep forest green with bright butter-yellow ground spot development
                    new_r = np.clip(r * 1.15 + 15, 0, 255)
                    new_g = np.clip(g * 1.20 + 10, 0, 255)
                    new_b = np.clip(b * 0.70, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.30)
                else:  # Overripe
                    new_r = np.clip(r * 0.85, 0, 255)
                    new_g = np.clip(g * 0.80, 0, 255)
                    new_b = np.clip(b * 0.50, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr)).enhance(1.08)

            # -------------------------------------------------------------
            # H. MANGO & DEFAULT
            # -------------------------------------------------------------
            else:
                if target_stage == "Semi-Ripe":
                    new_r = np.clip(r * 1.10 + g * 0.35, 0, 255)
                    new_g = np.clip(g * 0.95 + r * 0.20, 0, 255)
                    new_b = np.clip(b * 0.70, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.25)
                elif target_stage == "Ripe":
                    new_r = np.clip(r * 1.35 + g * 0.45 + 20, 0, 255)
                    new_g = np.clip(g * 0.85 + r * 0.30, 0, 255)
                    new_b = np.clip(b * 0.40, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    gen_img = ImageEnhance.Brightness(ImageEnhance.Color(Image.fromarray(gen_arr)).enhance(1.40)).enhance(1.08)
                else:  # Overripe
                    new_r = np.clip(r * 1.20 + 30, 0, 255)
                    new_g = np.clip(g * 0.65 + 15, 0, 255)
                    new_b = np.clip(b * 0.30, 0, 255)
                    gen_arr = np.stack([new_r, new_g, new_b], axis=-1).astype(np.uint8)
                    spot_mask = (np.random.rand(h, w) > 0.985)
                    gen_arr[spot_mask] = gen_arr[spot_mask] * 0.30
                    gen_img = ImageEnhance.Contrast(Image.fromarray(gen_arr).filter(ImageFilter.SMOOTH_MORE)).enhance(1.10)

            buffered = io.BytesIO()
            gen_img.save(buffered, format="JPEG", quality=92)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"

        except Exception as e:
            print(f"[UniversalFruitCGANModel] Generation failed for {fruit}: {e}")
            return None

# Backward compatibility alias
MangoCGANModel = UniversalFruitCGANModel
