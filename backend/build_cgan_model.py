"""
Build & Save PyTorch Conditional GAN Generator Checkpoint for FruitTwin Mango Models.
Saves to models/mango/mango_cgan.pth
"""

import os
import torch
import torch.nn as nn

MANGO_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "mango")
os.makedirs(MANGO_MODELS_DIR, exist_ok=True)
GAN_MODEL_PTH = os.path.join(MANGO_MODELS_DIR, "mango_cgan.pth")

class MangoCGANGenerator(nn.Module):
    """
    Conditional GAN Generator for Mango Ripening Stage Image Synthesis.
    Takes input image tensor (B, 3, H, W) and target stage label tensor (B, 1),
    and synthesizes target ripening stage visual appearance.
    """
    def __init__(self):
        super().__init__()
        self.stage_embed = nn.Embedding(4, 16)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, img, stage_label):
        b, c, h, w = img.size()
        emb = self.stage_embed(stage_label).view(b, 16, 1, 1).expand(b, 16, h, w)
        x = torch.cat([img, emb], dim=1)
        feat = self.encoder(x)
        out = self.decoder(feat)
        # Residual blend with input image for realistic texture preservation
        return torch.clamp(img * 0.4 + out * 0.6, -1.0, 1.0)

def generate_and_save_cgan():
    print("Building PyTorch Conditional GAN Model for Mango...")
    model = MangoCGANGenerator()
    torch.save(model, GAN_MODEL_PTH)
    print(f"Saved Conditional GAN model to: {GAN_MODEL_PTH}")

if __name__ == "__main__":
    generate_and_save_cgan()
