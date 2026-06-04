# GharMol — House Price Predictor

FastAPI-based house price prediction app with location + amenities support.

## Features
- City-wise pricing (Mumbai, Delhi, Bangalore, etc.)
- Locality tier (Prime, Central, Suburban, Outskirts)
- Amenities: Parking, Gym, Pool, Security, Lift
- Price range estimate (±12%)
- Detailed breakdown

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the server
```bash
python main.py
```

### Step 3 — Open in browser
```
http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| POST | `/predict` | Get price prediction |
| GET | `/cities` | List of supported cities |
| GET | `/health` | Server health check |

## Predict API Example

**POST /predict**

```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft": 1200,
  "city": "bangalore",
  "locality_tier": "central",
  "parking": true,
  "gym": false,
  "pool": false,
  "security": true,
  "lift": true
}
```

**Response:**
```json
{
  "predicted_price": 98.5,
  "price_per_sqft": 8210,
  "price_range_low": 86.7,
  "price_range_high": 110.3,
  "breakdown": {
    "city_factor": 1.8,
    "locality_factor": 1.3,
    "amenity_bonus": 1.1
  }
}
```

## Deploy on Render

1. Upload code to GitHub
2. Create new Web Service on Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
