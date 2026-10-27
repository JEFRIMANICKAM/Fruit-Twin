"""
Train / Generate Lightweight Model Weight Checkpoints for FruitTwin Mango Models.

Run:
  python train_demo_models.py

This script creates:
  1. models/mango/mango_ripeness_model.pth  (PyTorch CNN classifier for Mango ripeness)
  2. models/mango/mango_ripening_model.pkl  (Scikit-Learn Random Forest regressor for ripening shelf-life)
  3. models/mango/mango_cgan.pth            (PyTorch Conditional GAN Generator for future stage visuals)
"""

import os
import pickle
import numpy as np

MANGO_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "mango")
os.makedirs(MANGO_MODELS_DIR, exist_ok=True)

def train_ripeness_cnn():
    print("[1/3] Building PyTorch Mango Ripeness Classifier...")
    try:
        import torch
        import torch.nn as nn
        
        class MangoCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16 * 7 * 7, 32),
                    nn.ReLU(),
                    nn.Linear(32, 4) # 4 classes: Unripe, Semi-Ripe, Ripe, Overripe
                )
                
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)
                
        model = MangoCNN()
        model_path = os.path.join(MANGO_MODELS_DIR, "mango_ripeness_model.pth")
        torch.save(model, model_path)
        print(f" Saved: {model_path}")
    except Exception as e:
        print(f" Failed to create CNN model: {e}")

def train_ripening_regressor():
    print("[2/3] Training Scikit-Learn Mango Ripening ML Regressor...")
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Synthetic dataset based on temperature, stage, and color features
        # Features: [stage_num, temp, green_ratio, yellow_ratio, red_ratio]
        # Targets: [days_to_ripe, total_shelf_life]
        X = []
        y = []
        
        for temp in [25, 27, 28, 30, 32]:
            for stage in [1, 2, 3, 4]:
                # Stage 1: Unripe, 2: Semi-Ripe, 3: Ripe, 4: Overripe
                if stage == 1:
                    days_ripe = max(0.5, 6.0 - (temp - 25) * 0.4)
                    shelf = days_ripe + 4.0
                    g, y_ratio, r = 0.7, 0.2, 0.1
                elif stage == 2:
                    days_ripe = max(0.5, 3.0 - (temp - 25) * 0.3)
                    shelf = days_ripe + 3.0
                    g, y_ratio, r = 0.3, 0.5, 0.2
                elif stage == 3:
                    days_ripe = 0.0
                    shelf = max(0.5, 3.0 - (temp - 25) * 0.2)
                    g, y_ratio, r = 0.1, 0.6, 0.3
                else:
                    days_ripe = 0.0
                    shelf = 0.5
                    g, y_ratio, r = 0.0, 0.3, 0.7
                    
                X.append([stage, temp, g, y_ratio, r])
                y.append([days_ripe, shelf])
                
        X = np.array(X)
        y = np.array(y)
        
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        model_path = os.path.join(MANGO_MODELS_DIR, "mango_ripening_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f" Saved: {model_path}")
    except Exception as e:
        print(f" Failed to train regressor: {e}")

def train_cgan_generator():
    print("[3/3] Building PyTorch Conditional GAN Generator...")
    try:
        import torch
        import torch.nn as nn
        
        class MangoCGANGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(4, 16)
                self.conv1 = nn.Conv2d(3 + 16, 32, kernel_size=3, padding=1)
                self.act = nn.ReLU()
                self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
                self.tanh = nn.Tanh()
                
            def forward(self, img, stage_label):
                b, c, h, w = img.size()
                emb = self.embed(stage_label).view(b, 16, 1, 1).expand(b, 16, h, w)
                x = torch.cat([img, emb], dim=1)
                x = self.act(self.conv1(x))
                x = self.tanh(self.conv2(x))
                return x
                
        generator = MangoCGANGenerator()
        model_path = os.path.join(MANGO_MODELS_DIR, "mango_cgan.pth")
        torch.save(generator, model_path)
        print(f" Saved: {model_path}")
    except Exception as e:
        print(f" Failed to create CGAN model: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("FruitTwin Model Weight Generator")
    print("=" * 60)
    train_ripeness_cnn()
    train_ripening_regressor()
    train_cgan_generator()
    print("Done! Model weight files initialized in models/mango/")
