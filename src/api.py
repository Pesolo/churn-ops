"""
FastAPI service for the Telco Churn model.
Demo-grade: loads the registered model fresh on every request (see README
note in project chat log for why -- acceptable for a demo, not for real
production traffic).
"""
import os
from typing import Literal

import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocessing import simplify_service_columns, encode_binary_columns

MODEL_URI = "models:/telco-churn-logreg@champion"

app = FastAPI(title="Telco Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float


def load_champion_model():
    """
    Loads the current @champion model fresh. Raises a clean HTTPException
    (503) on failure instead of letting an MLflow/network error surface
    as an unhandled 500 with a raw stack trace.
    """
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        return mlflow.sklearn.load_model(MODEL_URI)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model service unavailable: could not load {MODEL_URI} ({exc})"
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model = load_champion_model()

    df = pd.DataFrame([customer.model_dump()])
    df = simplify_service_columns(df)
    df = encode_binary_columns(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]

    return PredictionResponse(
        churn_prediction=int(prediction),
        churn_probability=float(probability),
    )


@app.get("/health")
def health():
    return {"status": "ok"}