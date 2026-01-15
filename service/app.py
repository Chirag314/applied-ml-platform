import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Config ---
MODEL_RUN = os.getenv("MODEL_RUN_NAME", "latest")


app = FastAPI(title="Applied ML Platform API", version="0.1.0")

# Load model once (production pattern)
model = None

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # set this in env for protection


class PredictRequest(BaseModel):
    rows: List[List[float]]


def resolve_run_name(run_name: str) -> str:
    if run_name != "latest":
        return run_name
    latest_path = Path("artifacts") / "latest.txt"
    if not latest_path.exists():
        raise RuntimeError(
            "latest.txt not found. Create artifacts/latest.txt with a run name."
        )
    return latest_path.read_text(encoding="utf-8").strip()


def require_admin(token: str | None):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def startup_load_model():
    global model, ACTIVE_RUN, ARTIFACT_DIR, MODEL_PATH, META_PATH

    ACTIVE_RUN = resolve_run_name(MODEL_RUN)

    ARTIFACT_DIR = Path("artifacts") / ACTIVE_RUN
    MODEL_PATH = ARTIFACT_DIR / "model.pkl"
    META_PATH = ARTIFACT_DIR / "metadata.json"

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)


import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Applied ML Platform API", version="0.1.0")

# Config (do NOT resolve at import time)
MODEL_RUN_NAME = os.getenv("MODEL_RUN_NAME", "latest")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Runtime state (set during startup / reload)
ACTIVE_RUN: Optional[str] = None
MODEL_PATH: Optional[Path] = None
META_PATH: Optional[Path] = None
model = None


def resolve_run_name(run_name: str) -> str:
    """Resolve 'latest' -> run name via artifacts/latest.txt."""
    if run_name != "latest":
        return run_name

    latest_path = Path("artifacts") / "latest.txt"
    if not latest_path.exists():
        raise RuntimeError(
            "artifacts/latest.txt not found. Create it and put a run name inside."
        )
    return latest_path.read_text(encoding="utf-8").strip()


def require_admin(token: Optional[str]) -> None:
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_active_model(run_name: str) -> None:
    """Load model + paths into global runtime state."""
    global ACTIVE_RUN, ARTIFACT_DIR, MODEL_PATH, META_PATH, model

    ACTIVE_RUN = resolve_run_name(run_name)
    ARTIFACT_DIR = Path("artifacts") / ACTIVE_RUN
    MODEL_PATH = ARTIFACT_DIR / "model.pkl"
    META_PATH = ARTIFACT_DIR / "metadata.json"

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    rows: List[List[float]]


class SetLatestRequest(BaseModel):
    run_name: str


@app.on_event("startup")
def startup_load_model():
    # Resolve + load once when the server starts
    load_active_model(MODEL_RUN_NAME)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_run_name_env": MODEL_RUN_NAME,
        "active_run": ACTIVE_RUN,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "metadata_path": str(META_PATH) if META_PATH else None,
    }


@app.get("/model-info")
def model_info():
    if META_PATH is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Metadata not found: {META_PATH}")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = np.array(req.rows, dtype=float)

    if X.ndim != 2:
        raise HTTPException(
            status_code=422, detail="rows must be 2D: List[List[float]]"
        )

    expected = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] != expected:
        raise HTTPException(
            status_code=422, detail=f"Expected {expected} features, got {X.shape[1]}"
        )

    preds = model.predict(X).tolist()
    out = {"active_run": ACTIVE_RUN, "predictions": preds}

    if hasattr(model, "predict_proba"):
        out["probabilities"] = model.predict_proba(X)[:, 1].tolist()

    return out


@app.post("/admin/set-latest")
def admin_set_latest(req: SetLatestRequest, token: Optional[str] = None):
    require_admin(token)

    run_dir = Path("artifacts") / req.run_name
    if not (run_dir / "model.pkl").exists():
        raise HTTPException(
            status_code=404, detail=f"model.pkl not found for run: {req.run_name}"
        )

    (Path("artifacts") / "latest.txt").write_text(req.run_name, encoding="utf-8")
    return {"status": "ok", "latest": req.run_name}


@app.post("/admin/reload-model")
def admin_reload_model(token: Optional[str] = None):
    require_admin(token)
    # Always reload from whatever latest.txt points to (or from env if not latest)
    load_active_model("latest" if MODEL_RUN_NAME == "latest" else MODEL_RUN_NAME)
    return {"status": "ok", "active_run": ACTIVE_RUN, "model_path": str(MODEL_PATH)}


@app.get("/health")
def health():
    return {"status": "ok", "model_run": ACTIVE_RUN, "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    # Guaranteed fast return
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Metadata not found: {META_PATH}")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = np.array(req.rows, dtype=float)

    # Input shape validation
    expected = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.ndim != 2:
        raise HTTPException(
            status_code=422, detail="rows must be 2D: List[List[float]]"
        )
    if X.shape[1] != expected:
        raise HTTPException(
            status_code=422, detail=f"Expected {expected} features, got {X.shape[1]}"
        )

    preds = model.predict(X).tolist()
    out = {"run_name": MODEL_RUN, "predictions": preds}

    if hasattr(model, "predict_proba"):
        out["probabilities"] = model.predict_proba(X)[:, 1].tolist()

    return out


from pydantic import BaseModel


class SetLatestRequest(BaseModel):
    run_name: str


@app.post("/admin/set-latest")
def admin_set_latest(req: SetLatestRequest, token: str | None = None):
    require_admin(token)
    # validate artifact exists
    run_dir = Path("artifacts") / req.run_name
    if not (run_dir / "model.pkl").exists():
        raise HTTPException(
            status_code=404, detail=f"model.pkl not found for run: {req.run_name}"
        )
    (Path("artifacts") / "latest.txt").write_text(req.run_name, encoding="utf-8")
    return {"status": "ok", "latest": req.run_name}


@app.post("/admin/reload-model")
def admin_reload_model(token: str | None = None):
    require_admin(token)
    global model, ARTIFACT_DIR, MODEL_PATH, META_PATH

    # re-resolve active run
    active = resolve_run_name("latest")
    new_dir = Path("artifacts") / active
    new_model_path = new_dir / "model.pkl"
    new_meta_path = new_dir / "metadata.json"

    if not new_model_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Model not found: {new_model_path}"
        )

    # load new model
    model = joblib.load(new_model_path)

    # update pointers
    ARTIFACT_DIR = new_dir
    MODEL_PATH = new_model_path
    META_PATH = new_meta_path

    return {"status": "ok", "active_run": active, "model_path": str(MODEL_PATH)}
