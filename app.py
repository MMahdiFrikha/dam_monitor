from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Edge AI Multi-Step Forecaster")

# Get directory containing app.py
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" if (BASE_DIR / "models").exists() else BASE_DIR

# 1. Load scalers and models
scaler_lgb = joblib.load(MODELS_DIR / "scaler_lgb.pkl")
scaler_xgb = joblib.load(MODELS_DIR / "scaler_xgboost.pkl")

xgb_soil = xgb.XGBRegressor()
xgb_soil.load_model(str(MODELS_DIR / "xgb_soil.json"))

lgb_flow = joblib.load(MODELS_DIR / "lgb_flow.pkl")

xgb_dist = xgb.XGBRegressor()
xgb_dist.load_model(str(MODELS_DIR / "xgb_distance.json"))

lgb_dist = joblib.load(MODELS_DIR / "lgb_distance.pkl")


class TelemetryPayload(BaseModel):
    sequence: list[float] = Field(..., description="Flat array of 36 timesteps x 7 features (252 total floats)")
    layer2_risk: int = Field(..., ge=0, le=2, description="Risk level: 0=Normal, 1=Warning, 2=Critical")


@app.get("/")
def home():
    return {"status": "Model service running on Render"}


@app.post("/predict")
def predict_horizon(payload: TelemetryPayload):
    raw_array = np.array(payload.sequence, dtype=np.float32)

    # Validate that exactly 252 values were received (36 timesteps * 7 features)
    if raw_array.size != 252:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 252 values (36 timesteps x 7 features), but received {raw_array.size}.",
        )

    try:
        # 1. Reshape flat 252 array to (36 timesteps, 7 features) so scaler can process it
        timesteps_2d = raw_array.reshape(36, 7)

        # 2. Scale all 36 timesteps using 7-feature scalers
        scaled_timesteps_xgb = scaler_xgb.transform(timesteps_2d)
        scaled_timesteps_lgb = scaler_lgb.transform(timesteps_2d)

        # 3. Flatten back into (1, 252) row matrix expected by models
        scaled_data_xgb = scaled_timesteps_xgb.reshape(1, -1)
        scaled_data_lgb = scaled_timesteps_lgb.reshape(1, -1)

    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing error: {str(err)}",
        )

    # Core predictions
    soil_pred = xgb_soil.predict(scaled_data_xgb)
    flow_pred = lgb_flow.predict(scaled_data_lgb)

    # Dynamic fallback route based on risk state
    if payload.layer2_risk in (1, 2):
        dist_pred = lgb_dist.predict(scaled_data_lgb)
        active_model = "LightGBM"
    else:
        dist_pred = xgb_dist.predict(scaled_data_xgb)
        active_model = "XGBoost"

    dist_flat = np.asarray(dist_pred).ravel()
    flow_flat = np.asarray(flow_pred).ravel()
    soil_flat = np.asarray(soil_pred).ravel()

    return {
        "distance_cm": dist_flat[:6].tolist(),
        "flow_rate": flow_flat[:6].tolist(),
        "soil_raw": soil_flat[:6].tolist(),
        "distance_model_used": active_model,
    }