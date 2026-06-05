import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="GharMol - House Price Predictor")

# Load model
with open("house_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]
le_city  = bundle["le_city"]
le_loc   = bundle["le_loc"]
cities   = bundle["cities"]
localities = bundle["localities"]


class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft: int
    city: str
    locality_tier: str
    parking: bool = False
    gym: bool = False
    pool: bool = False
    security: bool = False
    lift: bool = False


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>GharMol API is running!</h1><p>Visit /docs for API docs.</p>"


@app.post("/predict")
async def predict(house: HouseInput):
    city = house.city.lower()
    loc  = house.locality_tier.lower()

    if city not in cities:
        city = "other"
    if loc not in localities:
        loc = "suburban"

    city_enc = le_city.transform([city])[0]
    loc_enc  = le_loc.transform([loc])[0]

    features = np.array([[
        city_enc, loc_enc,
        house.bedrooms, house.bathrooms, house.sqft,
        int(house.parking), int(house.gym), int(house.pool),
        int(house.security), int(house.lift)
    ]])

    predicted = float(model.predict(features)[0])
    price_per_sqft = int((predicted * 1e5) / house.sqft)

    city_factors = {
        'mumbai': 1.8, 'delhi': 1.6, 'bangalore': 1.5, 'hyderabad': 1.3,
        'pune': 1.2, 'chennai': 1.25, 'kolkata': 1.1, 'jaipur': 1.0,
        'lucknow': 0.9, 'other': 1.0
    }
    loc_factors = {'prime': 1.4, 'central': 1.2, 'suburban': 1.0, 'outskirts': 0.8}
    amenity_count = sum([house.parking, house.gym, house.pool, house.security, house.lift])

    return {
        "predicted_price": round(predicted, 1),
        "price_per_sqft": price_per_sqft,
        "price_range_low": round(predicted * 0.88, 1),
        "price_range_high": round(predicted * 1.12, 1),
        "breakdown": {
            "city_factor": city_factors.get(city, 1.0),
            "locality_factor": loc_factors.get(loc, 1.0),
            "amenity_bonus": round(1 + amenity_count * 0.03, 2)
        }
    }


@app.get("/cities")
async def get_cities():
    return {"cities": cities, "localities": localities}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
