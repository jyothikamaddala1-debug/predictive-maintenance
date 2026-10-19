from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Predictive Maintenance API")

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "model" / "gradient_boosting_model.pkl")
encoder = joblib.load(BASE_DIR / "model" / "type_encoder.pkl")

class MachineData(BaseModel):
    machine_type: str = Field(..., description="Machine type: H, L, or M")
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: float = Field(..., description="Rotational speed in RPM")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: float = Field(..., description="Tool wear in minutes")
    @field_validator("machine_type")
    @classmethod
    def validate_machine_type(cls, value):
        if value not in ["H", "L", "M"]:
            raise ValueError("machine_type must be H, L, or M")
        return value
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running!"
    }


@app.post("/predict")
def predict(data: MachineData):

    type_encoded = encoder.transform([[data.machine_type]])

    numeric_features = np.array([[
        data.air_temperature,
        data.process_temperature,
        data.rotational_speed,
        data.torque,
        data.tool_wear
    ]])

    features = np.hstack([type_encoded, numeric_features])

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Machine Failure Predicted"
    else:
        result = "No Machine Failure Predicted"

    return {
        "prediction": int(prediction),
        "result": result
    }