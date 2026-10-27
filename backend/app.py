import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from PIL import Image

from models.base import get_models_status
from services.image_processing import process_uploaded_image
from services.prediction_service import (
    analyze_fruit_image,
    calculate_what_if,
    predict_single_fruit,
    predict_single_maturity,
    predict_single_ripening_time
)
from services.gan_service import simulate_future_stage
from data.dataset_loader import get_dataset_summary
from data.rawripe_loader import get_rawripe_summary
from data.fruits360_loader import find_fruits360_root, discover_dataset_classes
from models.class_registry import class_registry

app = FastAPI(
    title="FruitTwin Multi-Fruit AI Platform",
    description="Multi-Fruit Identification (Fruits-360), Maturity Classification (RawRipe), and GAN-Based Future Appearance Simulation",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 12 * 1024 * 1024  # 12 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

def validate_image_file(file: UploadFile, file_bytes: bytes) -> Image.Image:
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Image file exceeds maximum 12MB limit.")
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()
        # Re-open after verify()
        pil_img = Image.open(io.BytesIO(file_bytes))
        return pil_img
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid or readable image.")

def validate_temperature(temperature: float) -> float:
    if temperature < 5.0 or temperature > 50.0:
        raise HTTPException(status_code=422, detail="Temperature must be between 5.0°C and 50.0°C.")
    return temperature

class RipeningTimeRequest(BaseModel):
    fruit: str
    temperature: float = Field(..., ge=5.0, le=50.0)
    current_stage: Optional[str] = "Semi-Ripe"

class WhatIfRequest(BaseModel):
    temperature: float = Field(..., ge=5.0, le=50.0)
    current_stage: Optional[str] = "Semi-Ripe"
    fruit: Optional[str] = "mango"

# =============================================================================
# HEALTH & DATASET METRICS ENDPOINTS
# =============================================================================
@app.get("/api/health")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "app": "FruitTwin Multi-Fruit AI Platform",
        "supported_fruits": ["mango", "banana", "papaya", "apple", "orange", "guava", "pomegranate", "watermelon", "strawberry", "coconut", "litchi"],
        "models": {
            "fruit_identification": "Fruits-360 Multi-Class Classifier",
            "maturity_classification": "RawRipe 10-Fruit Maturity Classifier",
            "future_simulation": "Conditional GAN Generator",
            "ripening_regression": "Gradient Boosting Regressor (Kasetsart University Telemetry)"
        },
        "version": "2.1.0"
    }

@app.get("/api/status")
def model_status(fruit: str = Query("mango")):
    return get_models_status(fruit)

@app.get("/api/dataset")
def dataset_info():
    real_summary = get_dataset_summary()
    rawripe_summary = get_rawripe_summary()
    fruits360_root = find_fruits360_root()
    fruits360_summary = discover_dataset_classes(fruits360_root) if fruits360_root else {"status": "not_found"}

    return {
        "real_mango_dataset": real_summary,
        "rawripe_dataset": rawripe_summary,
        "fruits360_dataset": {
            "num_classes": fruits360_summary.get("num_classes", 0),
            "classes": list(fruits360_summary.get("classes", {}).keys())
        }
    }

# =============================================================================
# DATASET CLASS CATALOGUE ENDPOINT
# =============================================================================
@app.get("/api/catalog")
@app.get("/catalog")
@app.get("/api/classes")
def dataset_catalog_endpoint():
    """
    Returns the complete, dynamically discovered class catalog from the trained Fruits-360 model,
    distinguishing fruit vs non-fruit dataset classes and detailing per-class capabilities:
    identification, maturity classification, GAN generation, and shelf-life regression.
    """
    return class_registry.get_catalog()

# =============================================================================
# UNIVERSAL FRUIT PRE-SCAN & VALIDATION ENDPOINTS
# =============================================================================
@app.post("/api/validate-fruit")
@app.post("/validate-fruit")
@app.post("/api/predict-fruit")
@app.post("/predict-fruit")
async def validate_or_predict_fruit_endpoint(
    file: UploadFile = File(...),
    active_tab: Optional[str] = Form(None),
    expected_fruit: Optional[str] = Form(None),
    fruit: Optional[str] = Form(None)
):
    """
    Universal image identification & validation endpoint running independently of the active tab.
    Identifies every class supported by the trained model, detects plant leaves,
    rejects non-fruit objects, identifies exotic unsupported fruits, and detects cross-fruit mismatches.
    """
    try:
        file_bytes = await file.read()
        pil_img = validate_image_file(file, file_bytes)
        tab_context = active_tab or expected_fruit or fruit
        result = predict_single_fruit(pil_img, expected_fruit=tab_context)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fruit identification failed: {str(e)}")

# =============================================================================
# MILESTONE 2: RAWRIPE MATURITY CLASSIFICATION ENDPOINT
# =============================================================================
@app.post("/api/predict-maturity")
@app.post("/predict-maturity")
async def predict_maturity_endpoint(
    file: UploadFile = File(...),
    fruit: Optional[str] = Form(None)
):
    """
    Classifies maturity stage as Raw or Ripe for the 10 RawRipe supported fruits.
    If the fruit is outside RawRipe, returns unsupported status with required message.
    """
    try:
        file_bytes = await file.read()
        pil_img = validate_image_file(file, file_bytes)
        result = predict_single_maturity(pil_img, fruit_name=fruit)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Maturity classification failed: {str(e)}")

# =============================================================================
# MILESTONE 3: RIPENING-TIME ESTIMATION ENDPOINT
# =============================================================================
@app.post("/api/predict-ripening-time")
@app.post("/predict-ripening-time")
def predict_ripening_time_endpoint(req: RipeningTimeRequest):
    """
    Estimates time remaining until specified maturity stage at given room temperature.
    Only supported when validated time-series training data is available.
    """
    validate_temperature(req.temperature)
    return predict_single_ripening_time(req.fruit, req.temperature, req.current_stage or "Semi-Ripe")

# =============================================================================
# MILESTONE 4: GAN-BASED FRUIT TWIN SIMULATION ENDPOINT
# =============================================================================
@app.post("/api/generate-twin")
@app.post("/generate-twin")
@app.post("/api/simulate")
async def generate_twin_endpoint(
    file: UploadFile = File(...),
    fruit: str = Form("mango"),
    temperature: float = Form(28.0),
    target_stage: str = Form("Ripe")
):
    """
    Synthesizes plausible visual appearance of the uploaded fruit at a selected target stage
    using Conditional GAN generation with clear AI simulation disclaimer.
    """
    try:
        file_bytes = await file.read()
        pil_img = validate_image_file(file, file_bytes)
        validate_temperature(temperature)

        result = simulate_future_stage(pil_img, temperature, target_stage, fruit)
        result["disclaimer"] = "This image is an AI-generated simulation of future fruit maturation, not a physical photograph."
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GAN simulation failed: {str(e)}")

# =============================================================================
# UNIFIED ANALYSIS ENDPOINT (FOR WEB DASHBOARD)
# =============================================================================
@app.post("/api/analyze")
async def analyze_fruit(
    file: UploadFile = File(...),
    fruit: str = Form("mango"),
    temperature: float = Form(28.0),
    humidity: Optional[float] = Form(None)
):
    """
    Full end-to-end analysis: runs Fruits-360 identification, RawRipe maturity,
    cross-tab verification, plant/leaf checks, and ripening time regression.
    """
    try:
        file_bytes = await file.read()
        pil_img = validate_image_file(file, file_bytes)
        validate_temperature(temperature)

        _, features = process_uploaded_image(file_bytes)
        result = analyze_fruit_image(pil_img, features, temperature, fruit)
        result["environmental_input"]["humidity"] = humidity
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/what-if")
def what_if_simulation(req: WhatIfRequest):
    validate_temperature(req.temperature)
    return calculate_what_if(req.temperature, req.current_stage, req.fruit or "mango")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
