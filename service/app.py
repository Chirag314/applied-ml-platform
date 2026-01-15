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


# Time series endpoint
"""

class TSPredictRequest(BaseModel):
    history: List[
        float
    ]  # most recent first or oldest first? We'll define oldest->newest
    lags: Optional[List[int]] = None
    rolling_windows: Optional[List[int]] = None


@app.post("/predict-timeseries")
def predict_timeseries(req: TSPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Load feature columns from metadata to keep train/serve consistent
    if META_PATH is None or not META_PATH.exists():
        raise HTTPException(
            status_code=500, detail="metadata.json missing for feature schema"
        )

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("modality") != "timeseries":
        raise HTTPException(
            status_code=400,
            detail=f"Loaded model is not timeseries (modality={meta.get('modality')})",
        )

    feature_cols = meta["feature_columns"]

    hist = np.asarray(req.history, dtype=float)
    if hist.size < 30:
        raise HTTPException(
            status_code=422, detail="history too short; provide at least ~30 points"
        )

    # Rebuild only the last row's features (v1: lag + rolling based on last values)
    # We assume history is oldest -> newest.
    import pandas as pd

    s = pd.Series(hist)
    feats = {}

    # infer lags/windows from metadata config if not provided
    cfg = meta.get("config", {})
    default_lags = cfg.get("features", {}).get("lags", [1, 2, 3, 6, 12, 24])
    default_windows = cfg.get("features", {}).get("rolling_windows", [6, 12, 24])

    lags = req.lags or default_lags
    windows = req.rolling_windows or default_windows

    n = len(s)
    for lag in lags:
        lag = int(lag)
        if n - lag - 1 < 0:
            raise HTTPException(
                status_code=422, detail=f"history too short for lag {lag}"
            )
        feats[f"lag_{lag}"] = float(s.iloc[-1 - lag])

    for w in windows:
        w = int(w)
        if n - 1 - w < 0:
            raise HTTPException(
                status_code=422, detail=f"history too short for rolling window {w}"
            )
        window_vals = s.iloc[-1 - w : -1]
        feats[f"roll_mean_{w}"] = float(window_vals.mean())
        feats[f"roll_std_{w}"] = (
            float(window_vals.std(ddof=1)) if len(window_vals) > 1 else 0.0
        )

    # time features are not available in this simplified request unless we pass timestamps
    # v1: if your model was trained with time features, we set them to 0
    for c in feature_cols:
        if c.startswith(("hour_", "dow_")) and c not in feats:
            feats[c] = 0.0

    X_row = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
    yhat = float(model.predict(X_row)[0])
    return {"active_run": ACTIVE_RUN, "yhat_next": yhat}
"""


class TSPredictRequest(BaseModel):
    history: List[float]  # oldest -> newest
    horizon: int = 1
    lags: Optional[List[int]] = None
    rolling_windows: Optional[List[int]] = None


@app.post("/predict-timeseries")
def predict_timeseries(req: TSPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if META_PATH is None or not META_PATH.exists():
        raise HTTPException(status_code=500, detail="metadata.json missing")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("modality") != "timeseries":
        raise HTTPException(
            status_code=400,
            detail=f"Loaded model is not timeseries (modality={meta.get('modality')})",
        )

    feature_cols = meta["feature_columns"]
    cfg = meta.get("config", {})
    default_lags = cfg.get("features", {}).get("lags", [1, 2, 3, 6, 12, 24])
    default_windows = cfg.get("features", {}).get("rolling_windows", [6, 12, 24])

    lags = [int(x) for x in (req.lags or default_lags)]
    windows = [int(x) for x in (req.rolling_windows or default_windows)]
    horizon = int(req.horizon)

    if horizon < 1 or horizon > 500:
        raise HTTPException(status_code=422, detail="horizon must be between 1 and 500")

    hist = np.asarray(req.history, dtype=float)
    if hist.ndim != 1:
        raise HTTPException(
            status_code=422, detail="history must be a 1D list of floats"
        )

    min_needed = max(max(lags), max(windows)) + 5
    if hist.size < min_needed:
        raise HTTPException(
            status_code=422,
            detail=f"history too short; need at least {min_needed} points",
        )

    import pandas as pd

    s = pd.Series(hist.tolist())  # mutable history for rollout

    preds = []
    for _ in range(horizon):
        feats = {}

        for lag in lags:
            feats[f"lag_{lag}"] = float(s.iloc[-1 - lag])

        for w in windows:
            window_vals = s.iloc[-1 - w : -1]
            feats[f"roll_mean_{w}"] = float(window_vals.mean())
            feats[f"roll_std_{w}"] = (
                float(window_vals.std(ddof=1)) if len(window_vals) > 1 else 0.0
            )

        # v1: no timestamps in request, so time features become 0
        for c in feature_cols:
            if c.startswith(("hour_", "dow_")) and c not in feats:
                feats[c] = 0.0

        X_row = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
        yhat = float(model.predict(X_row)[0])
        preds.append(yhat)

        # recursive rollout
        s.loc[len(s)] = yhat

    return {"active_run": ACTIVE_RUN, "horizon": horizon, "yhat": preds}
