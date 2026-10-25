import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

DB_PATH = Path("mlflow.db").resolve()
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

MODEL_NAME = "logreg"
MLFLOW_URI = f"models:/forage-logreg/latest"
LOCAL_PKL = Path(f"models/{MODEL_NAME}.pkl")
SEUIL = 0.5

model = None
model_source = "none"

try:
    model = mlflow.sklearn.load_model(MLFLOW_URI)
    model_source = "mlflow"
    print(f"Modele charge depuis MLflow : {MLFLOW_URI}")
except Exception as e:
    print(f"MLflow KO ({e}). Tentative chargement local...")

if model is None and LOCAL_PKL.exists():
    try:
        model = joblib.load(LOCAL_PKL)
        model_source = "local"
        print(f"Modele charge depuis fichier local : {LOCAL_PKL}")
    except Exception as e:
        print(f"Local KO ({e})")

if model is None:
    print("Aucun modele charge. Lance d'abord : python -m src.train.train")


app = FastAPI(title="API Prediction Pannes Forages", version="1.0.0")


class ForageInput(BaseModel):
    age_ans: int = Field(..., ge=0, le=100)
    profondeur_m: int = Field(..., gt=0, le=500)
    debit_m3h: float = Field(..., ge=0)
    usagers_jour: int = Field(..., gt=0)
    jours_depuis_maintenance: int = Field(..., ge=0)
    pannes_12m: int = Field(..., ge=0)
    type_pompe: str
    region: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_source": model_source}


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return {"model_name": MODEL_NAME, "source": model_source, "seuil": SEUIL}


@app.post("/predict")
def predict(data: ForageInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    df = pd.DataFrame([data.model_dump()])
    proba = float(model.predict_proba(df)[0][1])
    return {
        "panne_probability": round(proba, 4),
        "risque": "eleve" if proba >= SEUIL else "faible",
        "seuil": SEUIL,
        "model_source": model_source,
    }