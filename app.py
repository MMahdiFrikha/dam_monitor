from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Edge AI Multi-Step Forecaster")

BASE_DIR = Path(__file__).resolve().parent

def resolve_model_path(filename: str) -> Path:
    """Check models/ directory first, fallback to base directory."""
    path_in_models = BASE_DIR / "models" / filename
    if path_in_models.exists():
        return path_in_models
    path_in_root = BASE_DIR / filename
    if path_in_root.exists():
        return path_in_root
    raise FileNotFoundError(f"Model asset '{filename}' not found in 'models/' or root directory.")

# 1. Load scalers using updated filenames
scaler_lgb = joblib.load(resolve_model_path("scaler_lgb.pkl"))
scaler_xgb = joblib.load(resolve_model_path("scaler_xgboost.pkl"))

# 2. Load models using updated filenames
xgb_soil = xgb.XGBRegressor()
xgb_soil.load_model(str(resolve_model_path("xgb_model_soil_raw.json")))

lgb_flow = joblib.load(resolve_model_path("lgbm_model_flow_rate_l_min.pkl"))

xgb_dist = xgb.XGBRegressor()
xgb_dist.load_model(str(resolve_model_path("xgb_model_distance_cm.json")))

lgb_dist = joblib.load(resolve_model_path("lgbm_model_distance_cm.pkl"))


class TelemetryPayload(BaseModel):
    sequence: list[float] = Field(..., description="Flat array of 36 timesteps x 7 features (252 total floats)")
    layer2_risk: int = Field(..., ge=0, le=2, description="Risk level: 0=Normal, 1=Warning, 2=Critical")


@app.get("/")
def home():
    return {"status": "Model service running on Render"}


@app.post("/predict")
def predict_horizon(payload: TelemetryPayload):
    raw_array = np.array(payload.sequence, dtype=np.float32)

    if raw_array.size != 252:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 252 values (36 timesteps x 7 features), received {raw_array.size}.",
        )

    try:
        # Reshape 252 flat values to (36 timesteps, 7 features) for scaling
        timesteps_2d = raw_array.reshape(36, 7)

        # Apply scalers trained on 7 features across all 36 timesteps
        scaled_timesteps_xgb = scaler_xgb.transform(timesteps_2d)
        scaled_timesteps_lgb = scaler_lgb.transform(timesteps_2d)

        # Flatten back into (1, 252) matrix expected by models
        scaled_data_xgb = scaled_timesteps_xgb.reshape(1, -1)
        scaled_data_lgb = scaled_timesteps_lgb.reshape(1, -1)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(err)}")

    # Core predictions
    soil_pred = xgb_soil.predict(scaled_data_xgb)
    flow_pred = lgb_flow.predict(scaled_data_lgb)

    # Dynamic model selection for distance based on risk state
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