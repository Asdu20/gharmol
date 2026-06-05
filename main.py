from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(title="GharMol - House Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load ML model bundle ──
MODEL_PATH = "house_model.pkl"

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
    model = bundle["model"]
    le_city = bundle["le_city"]
    le_loc = bundle["le_loc"]
    FEATURES = bundle["features"]
    CITIES = bundle["cities"]
    LOCALITIES = bundle["localities"]
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    raise e


class HouseFeatures(BaseModel):
    bedrooms:      int   = Field(..., ge=1, le=5)
    bathrooms:     int   = Field(..., ge=1, le=4)
    sqft:          float = Field(..., ge=200, le=10000)
    city:          str
    locality_tier: str
    parking:       bool = False
    gym:           bool = False
    pool:          bool = False
    security:      bool = False
    lift:          bool = False


class PredictionResponse(BaseModel):
    predicted_price:  float
    price_per_sqft:   float
    price_range_low:  float
    price_range_high: float
    model_used:       str
    breakdown:        dict


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    city_key = features.city.lower().strip()
    loc_key  = features.locality_tier.lower().strip()

    if city_key not in CITIES:
        city_key = "other"
    if loc_key not in LOCALITIES:
        raise HTTPException(status_code=400, detail=f"Invalid locality_tier. Use: {LOCALITIES}")

    city_enc = int(le_city.transform([city_key])[0])
    loc_enc  = int(le_loc.transform([loc_key])[0])

    X = np.array([[
        city_enc, loc_enc,
        features.bedrooms, features.bathrooms, features.sqft,
        int(features.parking), int(features.gym),
        int(features.pool), int(features.security), int(features.lift)
    ]])

    pred_lakh = float(model.predict(X)[0])
    pred_lakh = max(pred_lakh, 1.0)

    price_per_sqft = round((pred_lakh * 100000) / features.sqft, 0)
    low  = round(pred_lakh * 0.88, 2)
    high = round(pred_lakh * 1.12, 2)

    breakdown = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURES, model.feature_importances_)
    }

    return PredictionResponse(
        predicted_price  = round(pred_lakh, 2),
        price_per_sqft   = price_per_sqft,
        price_range_low  = low,
        price_range_high = high,
        model_used       = "GradientBoostingRegressor (R2=0.9768)",
        breakdown        = breakdown
    )


@app.get("/cities")
async def get_cities():
    return {"cities": CITIES, "localities": LOCALITIES}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "supported_cities": len(CITIES),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
