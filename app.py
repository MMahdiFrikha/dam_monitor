import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Edge AI Multi-Step Forecaster")

# 1. Load models into memory when Space starts
scaler = joblib.load("models/scaler.pkl")

xgb_soil = xgb.XGBRegressor()
xgb_soil.load_model("models/xgb_soil.json")

lgb_flow = joblib.load("models/lgb_flow.pkl")

xgb_dist = xgb.XGBRegressor()
xgb_dist.load_model("models/xgb_distance.json")

lgb_dist = joblib.load("models/lgb_distance.pkl")


class TelemetryPayload(BaseModel):
    sequence: list  # Flat array of 36 timesteps x features
    layer2_risk: int  # 0: Normal, 1: Warning, 2: Critical


@app.get("/")
def home():
    return {"status": "Model service running on Hugging Face Spaces"}


@app.post("/predict")
def predict_horizon(payload: TelemetryPayload):
    # Convert input array to 2D numpy array
    data = np.array(payload.sequence).reshape(1, -1)

    # Always-active models
    soil_pred = xgb_soil.predict(data)
    flow_pred = lgb_flow.predict(data)

    # Dynamic Switch based on Layer 2 Risk state (Switch to LightGBM on Risk 1 or 2)
    if payload.layer2_risk in [1, 2]:
        dist_pred = lgb_dist.predict(data)
        active_model = "LightGBM"
    else:
        dist_pred = xgb_dist.predict(data)
        active_model = "XGBoost"

    # Truncate horizon to top 6 steps (t+1 to t+6)
    return {
        "distance_cm": dist_pred[0, :6].tolist(),
        "flow_rate": flow_pred[0, :6].tolist(),
        "soil_raw": soil_pred[0, :6].tolist(),
        "distance_model_used": active_model,
    }