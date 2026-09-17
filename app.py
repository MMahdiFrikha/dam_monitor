import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Edge AI Multi-Step Forecaster")

# Load models and preprocessing pipelines globally into memory
scaler = joblib.load("models/scaler.pkl")

xgb_soil = xgb.XGBRegressor()
xgb_soil.load_model("models/xgb_soil.json")

lgb_flow = joblib.load("models/lgb_flow.pkl")

xgb_dist = xgb.XGBRegressor()
xgb_dist.load_model("models/xgb_distance.json")

lgb_dist = joblib.load("models/lgb_distance.pkl")


class TelemetryPayload(BaseModel):
    sequence: list[float] = Field(..., description="Flat telemetry feature array")
    layer2_risk: int = Field(..., ge=0, le=2, description="Risk level: 0=Normal, 1=Warning, 2=Critical")


@app.get("/")
def home():
    return {"status": "Model service running on Hugging Face Spaces"}


@app.post("/predict")
def predict_horizon(payload: TelemetryPayload):
    # Convert input payload to a 2D NumPy array
    raw_data = np.array(payload.sequence, dtype=np.float32).reshape(1, -1)

    # Scale telemetry features before passing to models
    try:
        scaled_data = scaler.transform(raw_data)
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing error: Verify that sequence length matches expected feature count. Details: {str(err)}",
        )

    # Run core baseline forecasting models
    soil_pred = xgb_soil.predict(scaled_data)
    flow_pred = lgb_flow.predict(scaled_data)

    # Dynamic fallback route for distance forecaster during elevated risk states
    if payload.layer2_risk in (1, 2):
        dist_pred = lgb_dist.predict(scaled_data)
        active_model = "LightGBM"
    else:
        dist_pred = xgb_dist.predict(scaled_data)
        active_model = "XGBoost"

    # Flatten predictions safely regardless of 1D or 2D output shape
    dist_flat = np.asarray(dist_pred).ravel()
    flow_flat = np.asarray(flow_pred).ravel()
    soil_flat = np.asarray(soil_pred).ravel()

    # Truncate multi-step forecasts to the top 6 horizon steps (t+1 to t+6)
    return {
        "distance_cm": dist_flat[:6].tolist(),
        "flow_rate": flow_flat[:6].tolist(),
        "soil_raw": soil_flat[:6].tolist(),
        "distance_model_used": active_model,
    }