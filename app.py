from pathlib import Path
import os
import json
import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establish and return a connection to Supabase PostgreSQL."""
    if not DATABASE_URL:
        print("[DB] DATABASE_URL is not set!")
        return None
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"[DB] Error connecting to Supabase: {e}")
        return None

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

# 1. Load scalers
scaler_lgb = joblib.load(resolve_model_path("scaler_lgb.pkl"))
scaler_xgb = joblib.load(resolve_model_path("scaler_xgboost.pkl"))

# 2. Load models
xgb_soil = xgb.XGBRegressor()
xgb_soil.load_model(str(resolve_model_path("xgb_model_soil_raw.json")))

lgb_flow = joblib.load(resolve_model_path("lgbm_model_flow_rate_l_min.pkl"))

xgb_dist = xgb.XGBRegressor()
xgb_dist.load_model(str(resolve_model_path("xgb_model_distance_cm.json")))

lgb_dist = joblib.load(resolve_model_path("lgbm_model_distance_cm.pkl"))


class TelemetryPayload(BaseModel):
    sequence: list[float] = Field(..., description="Flat array of 36 timesteps x 7 features (252 total floats)")
    layer2_risk: int = Field(..., ge=0, le=2, description="Risk level: 0=Normal, 1=Warning, 2=Critical")
    station_id: str | None = Field(default="0", description="Station identifier")
    overflow_alert: bool | None = Field(default=False)
    ai_anomaly_alert: bool | None = Field(default=False)


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
        # Reshape 252 flat values to (36 timesteps, 7 features)
        timesteps_2d = raw_array.reshape(36, 7)

        # Apply scalers
        scaled_timesteps_xgb = scaler_xgb.transform(timesteps_2d)
        scaled_timesteps_lgb = scaler_lgb.transform(timesteps_2d)

        # Flatten into (1, 252) expected by the models
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

    # Extract first 6 steps of the forecast horizon
    dist_horizon = np.asarray(dist_pred).ravel()[:6].tolist()
    flow_horizon = np.asarray(flow_pred).ravel()[:6].tolist()
    soil_horizon = np.asarray(soil_pred).ravel()[:6].tolist()

    # Extract the current/latest sensor readings from the last timestep (t=36)
    latest_step = timesteps_2d[-1]
    curr_distance = float(latest_step[0])
    curr_rain_raw = int(latest_step[1])
    curr_flow_rate = float(latest_step[2])
    curr_soil_raw = int(latest_step[3])
    curr_tds_voltage = float(latest_step[4])
    curr_edge_ai_score = float(latest_step[5])
    curr_fog_class = str(int(latest_step[6]))

    # --- Persist to Supabase ---
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO forecasts (
                        station_id,
                        distance,
                        rain_raw,
                        flow_rate,
                        soil_raw,
                        tds_voltage,
                        edge_ai_score,
                        fog_class,
                        overflow_alert,
                        ai_anomaly_alert,
                        synced_to_cloud,
                        prediction_distance,
                        prediction_flow_rate,
                        prediction_soil_raw,
                        distance_model_used
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        payload.station_id,
                        curr_distance,
                        curr_rain_raw,
                        curr_flow_rate,
                        curr_soil_raw,
                        curr_tds_voltage,
                        curr_edge_ai_score,
                        curr_fog_class,
                        payload.overflow_alert,
                        payload.ai_anomaly_alert,
                        True,  # synced_to_cloud
                        json.dumps(dist_horizon),
                        json.dumps(flow_horizon),
                        json.dumps(soil_horizon),
                        active_model,
                    ),
                )
                conn.commit()
        except Exception as db_err:
            print(f"[DB] Insert failed: {db_err}")
            conn.rollback()
        finally:
            conn.close()

    return {
        "prediction_distance": dist_horizon,
        "prediction_flow_rate": flow_horizon,
        "prediction_soil_raw": soil_horizon,
        "distance_model_used": active_model,
    }