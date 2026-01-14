from curses import meta
import os
from typing import List, Optional
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path

from fastapi import HTTPException

MODEL_RUN = os.getenv("MODEL_RUN_NAME", "tabular_baseline_v1")
MODEL_PATH = f"artifacts/{MODEL_RUN}/model.pkl"
META_PATH = Path(f"artifacts/{MODEL_RUN}/metadata.json")


app = FastAPI(title="Applied ML Platform API", version="0.1.0")

model = None  # global cache
import json
from pathlib import Path

META_PATH = Path(f"artifacts/{MODEL_RUN}/metadata.json")


class PredictRequest(BaseModel):
    rows: List[List[float]]


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else None
    return {"status": "ok", "model_run": MODEL_RUN, "metadata": meta}


@app.get("/model-info")
def model_info():
    if not META_PATH.exists():
        return {"error": "metadata not found", "run_name": MODEL_RUN}
    return json.loads(META_PATH.read_text())


@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.rows, dtype=float)
    expected = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] != expected:
        return {
            "error": f"Expected {expected} features per row, go {X.shape[1]}",
            "run_name": MODEL_RUN,
        }

    if X.shape[1] != expected:
        raise HTTPException(
            status_code=422, detail=f"Expected {expected} features, got {X.shape[1]}"
        )

    pred = model.predict(X).tolist()
    out = {"run_name": MODEL_RUN, "predictions": pred}
    if hasattr(model, "predict_proba"):
        out["probabilities"] = model.predict_proba(X)[:, 1].tolist()
    return out
