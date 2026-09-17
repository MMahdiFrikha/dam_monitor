import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Edge AI Multi-Step Forecaster")

# 1. Load scalers into memory using your actual filenames
scaler_lgb = joblib.load("models/scaler_lgb.pkl")
scaler_xgb = joblib.load("models/scaler_xgboost.pkl")

# 2. Load models into memory
xgb_soil = xgb.XGBRegressor()
xgb_soil.load_model("models/xgb_model_soil_raw.json")

lgb_flow = joblib.load("models/lgbm_model_flow_rate_l_min.pkl")

xgb_dist = xgb.XGBRegressor()
xgb_dist.load_model("models/xgb_model_distance_cm.json")

lgb_dist = joblib.load("models/lgbm_model_distance_cm.pkl")


class TelemetryPayload(BaseModel):
    sequence: list[float] = Field(..., description="Flat telemetry feature array")
    layer2_risk: int = Field(..., ge=0, le=2, description="Risk level: 0=Normal, 1=Warning, 2=Critical")


@app.get("/")
def home():
    return {"status": "Model service running"}


@app.post("/predict")
def predict_horizon(payload: TelemetryPayload):
    # Convert input payload to 2D NumPy array
    raw_data = np.array(payload.sequence, dtype=np.float32).reshape(1, -1)

    try:
        # Scale input telemetry features separately using model-specific scalers
        scaled_data_xgb = scaler_xgb.transform(raw_data)
        scaled_data_lgb = scaler_lgb.transform(raw_data)
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing error: Verify sequence length matches expected features. Details: {str(err)}",
        )

    # Always-active baseline predictions (using their respective scaled inputs)
    soil_pred = xgb_soil.predict(scaled_data_xgb)
    flow_pred = lgb_flow.predict(scaled_data_lgb)

    # Dynamic fallback based on Layer 2 Risk state
    if payload.layer2_risk in (1, 2):
        dist_pred = lgb_dist.predict(scaled_data_lgb)
        active_model = "LightGBM"
    else:
        dist_pred = xgb_dist.predict(scaled_data_xgb)
        active_model = "XGBoost"

    # Flatten prediction arrays safely (handles both 1D and 2D model outputs)
    dist_flat = np.asarray(dist_pred).ravel()
    flow_flat = np.asarray(flow_pred).ravel()
    soil_flat = np.asarray(soil_pred).ravel()

    # Truncate multi-step forecasts to top 6 horizon steps (t+1 to t+6)
    return {
        "distance_cm": dist_flat[:6].tolist(),
        "flow_rate": flow_flat[:6].tolist(),
        "soil_raw": soil_flat[:6].tolist(),
        "distance_model_used": active_model,
    }