import os
from typing import List
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_RUN = os.getenv("MODEL_RUN_NAME", "tabular_baseline_v1")
MODEL_PATH = f"artifacts/{MODEL_RUN}/model.pkl"

app = FastAPI(title="Applied ML Platform API", version="0.1.0")


class PredictRequest(BaseModel):
    rows: List[List[float]]


@app.get("/health")
def health():
    return {"status": "ok", "model_run": MODEL_RUN, "model_path": MODEL_PATH}


@app.post("/predict")
def predict(req: PredictRequest):
    model = joblib.load(MODEL_PATH)
    X = np.array(req.rows, dtype=float)
    pred = model.predict(X).tolist()
    out = {"run_name": MODEL_RUN, "predictions": pred}
    if hasattr(model, "predict_proba"):
        out["probabilities"] = model.predict_proba(X)[:, 1].tolist()
    return out
