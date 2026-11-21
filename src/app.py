from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.ML_TRACKING import train
from src.features import extract, FEATURES

app = FastAPI(title="Phishing URL Detection API")


class UrlRequest(BaseModel):
    url: str = Field(min_length=3, max_length=2048, description="A domain or full URL to check",
                     examples=["paypal-login-secure.xyz"])


class Prediction(BaseModel):
    url: str = Field(examples=["paypal-login-secure.xyz"])
    result: Literal["Fake", "Genuine"] = Field(examples=["Fake"])
    fake_probability: float = Field(ge=0, le=1, description="Model confidence that it is fake",
                                        examples=[0.97])


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")  # Swagger UI


@app.post("/predict", summary="Is this domain fake (phishing) or genuine?", response_model=Prediction)
def predict(req: UrlRequest):
    try:
        model = joblib.load("models/model.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found. Call /train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load the model: {e}")

    try:
        features = pd.DataFrame([extract(req.url)])[FEATURES]
        result = model.predict(features)[0]
        proba = dict(zip(model.classes_, model.predict_proba(features)[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"url": req.url, "result": result, "fake_probability": round(float(proba.get("Fake", 0)), 4)}


@app.post("/train", summary="Train the models")
def train_models():
    # trains the models, logs them to MLflow and saves the best one to models/model.pkl
    try:
        return {"runs": train()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Training data not found: {e.filename}")
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Column missing in the dataset: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
