# FruitTwin – Multi-Fruit Identification, Maturity Classification & GAN Simulation

> **"See the fruit today. Predict its future."**

FruitTwin is an AI-powered agricultural digital twin platform that couples computer vision, statistical machine learning, and Generative Adversarial Networks (GANs) for:
1. **Multi-Fruit Identification & Variety Recognition** (using **Fruits-360**)
2. **Raw-versus-Ripe Maturity Classification** (using **RawRipe**)
3. **GAN-Based Future Fruit Maturation Simulation**
4. **Ripening-Time Estimation & Room Temperature Physics** (calibrated on storage telemetry)

---

## 📌 Supported Capabilities Matrix

| Feature | Supported Fruits | Benchmark Dataset | Model / Status |
| :--- | :--- | :--- | :--- |
| **Fruit Identification** | Apple, Banana, Mango, Orange, Guava, Papaya, Pomegranate, Watermelon, etc. | Fruits-360 | RandomForest Spectrum Classifier (88.9% test accuracy) |
| **Variety Recognition** | Braeburn, Granny Smith, Red (Sevazhai), Lady Finger, Alphonso, Nagpur, etc. | Fruits-360 | Integrated into Fruits-360 pipeline |
| **Maturity (Raw vs Ripe)** | Apple, Banana, Coconut, Guava, Litchi, Mango, Orange, Papaya, Pomegranate, Strawberry (10 Fruits) | RawRipe (1,630 images) | RawRipe Maturity Classifier (91.3% test accuracy) |
| **Unsupported Fruits** | Watermelon, Exotic specimens | None | Explicit: *"Fruit identified successfully. Ripeness prediction is not yet available for this fruit."* |
| **Future Appearance Twin** | Mango, Banana, Papaya, Apple, Orange, Guava, Pomegranate, Watermelon | FruitTwin cGAN | Conditional GAN Generator (Residual Stage Transfer) |
| **Ripening Time** | Mango | Kasetsart University Storage Telemetry | GradientBoostingRegressor (MAE = 1.60 days at 25-28°C) |
| **Ripening Time (Other)** | Banana, Apple, etc. | None | Explicit: *"Ripening-time estimation is not yet supported because validated training data is unavailable."* |

---

## ⚙️ Coordinated Multi-Model Architecture

```
User Upload (Image + Room Temperature in °C)
                  │
                  ▼
         FastAPI Backend (/api)
                  │
  ┌───────────────┼───────────────┬───────────────────────────────┐
  ▼               ▼               ▼                               ▼
1. Fruits-360   2. RawRipe      3. Conditional GAN              4. Ripening Time Regressor
Identification  Maturity (10)   Simulation                      (Storage Telemetry)
  │               │               │                               │
  ├─ Fruit Name   ├─ Raw / Ripe   ├─ Simulated Image              ├─ Remaining Days to Peak
  ├─ Variety Name ├─ Confidence   ├─ Original vs Generated        ├─ Shelf Life
  ├─ Top 3 list   └─ Unsupported  └─ AI Simulation Disclaimer     └─ "Data Unavailable" (Honest)
  └─ Reliability     Notice
```

---

## 📊 Model Evaluation Results

### 1. Fruits-360 Multi-Class Classifier (`models/fruits360_model.pkl`)
- **Dataset**: Fruits-360 (Official Training and Test splits)
- **Held-Out Test Set**: 36 test samples
- **Overall Accuracy**: **88.89%**
- **Precision**: **88.89%**
- **Recall**: **88.89%**
- **F1-Score**: **88.89%**

### 2. RawRipe Maturity Classifier (`models/rawripe_model.pkl`)
- **Dataset**: RawRipe (`sets/train.txt` and `sets/test.txt`)
- **Total Labeled Images**: 1,630 (1,216 train, 414 test)
- **Held-Out Test Set**: 414 test images
- **Overall Accuracy**: **91.30%**
- **Overall Precision**: **91.42%**
- **Overall Recall**: **91.30%**
- **Overall F1-Score**: **91.31%**

#### Per-Fruit Performance Breakdown:
- **Bananas**: 100.0% Accuracy, 100.0% F1 (50 test samples)
- **Oranges**: 100.0% Accuracy, 100.0% F1 (35 test samples)
- **Papayas**: 97.7% Accuracy, 97.7% F1 (43 test samples)
- **Litchis**: 96.9% Accuracy, 96.9% F1 (32 test samples)
- **Apples**: 96.3% Accuracy, 96.3% F1 (54 test samples)
- **Pomegranates**: 94.9% Accuracy, 94.9% F1 (39 test samples)
- **Coconuts**: 92.1% Accuracy, 92.1% F1 (38 test samples)
- **Mangoes**: 90.2% Accuracy, 89.9% F1 (41 test samples)
- **Strawberries**: 82.6% Accuracy, 82.1% F1 (46 test samples)
- **Guavas**: 58.3% Accuracy, 52.5% F1 (36 test samples)

### 3. Ripening-Time Regressor (`models/mango_ripening_regressor.pkl`)
- **Dataset**: Kasetsart University Nam Dok Mai Si Tong Mango Storage Telemetry (100 train, 100 validation)
- **Held-Out Validation Set**: 100 observations
- **Mean Absolute Error (MAE)**: **1.60 days**
- **Evaluation Conditions**: Ambient 25-28°C storage, monitored on Days 1, 3, 5, 7, 9.

---

## 🚀 How to Run the Application Locally

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 2. Dataset Setup
```bash
# 1. Download & unpack RawRipe (1,630 images)
cd backend/data
python download_rawripe.py

# 2. Download Fruits-360 benchmark images
python setup_fruits360.py
```
*Note: If you have the full Kaggle archive for Fruits-360 (`moltean/fruits`), simply extract it into `backend/data/fruits-360/`. The loader auto-discovers all subdirectories and classes!*

### 3. Train Models
```bash
cd backend
# Train Fruits-360 multi-class model
python train_fruits360.py

# Train RawRipe 10-fruit maturity model
python train_rawripe.py

# Train Mango ripening regressor
python train_ripening_time_regressor.py

# Run the 13-case verification test suite
python test_all_milestones.py
```

### 4. Start the Application
```bash
# In terminal 1 (Backend):
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# In terminal 2 (Frontend):
cd frontend
npm run dev
```

- Web Application: `http://localhost:3000`
- API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`

---

## 📡 API Endpoints

- `POST /api/predict-fruit`
  - Upload fruit image.
  - Returns `predicted_fruit`, `predicted_variety`, `confidence`, `is_reliable`, `top_three_candidates`.
- `POST /api/predict-maturity`
  - Upload fruit image + fruit name.
  - Returns `maturity_stage` ("Raw" or "Ripe") or unsupported notice: *"Fruit identified successfully. Ripeness prediction is not yet available for this fruit."*
- `POST /api/generate-twin`
  - Upload fruit image + target stage + temperature.
  - Returns generated image base64 with mandatory disclaimer: *"This image is an AI-generated simulation of future fruit maturation, not a physical photograph."*
- `POST /api/predict-ripening-time`
  - Temperature in °C + fruit name.
  - Returns estimated days (for mango) or honest message: *"Ripening-time estimation is not yet supported because validated training data is unavailable."*
- `POST /api/analyze`
  - End-to-end unified analysis endpoint for the dashboard.

---

## ⚠️ Known Limitations & Expansion Roadmap
1. **Ripening-Time Ground Truth**: While RawRipe provides 1,630 raw vs ripe agricultural photos, it does not provide time-series tracking of the same physical specimens over consecutive days. Validated days-to-ripe regression is currently calibrated for mangoes using Kasetsart University telemetry. Expanding this requires multi-day longitudinal photography for apples, bananas, and papayas.
2. **Paired Before-and-After Images**: RawRipe images are unpaired individual fruits. For pixel-level image-to-image translation, an expanded dataset of registered daily fruit photos will further enhance texture fidelity.
