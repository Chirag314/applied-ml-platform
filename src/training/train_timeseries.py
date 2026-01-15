import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
import joblib

from src.utils.config import load_yaml
from src.utils.reproducibility import set_global_seed
from src.models.registry import ModelRegistry
from src.data.timeseries import make_sine_series
from src.features.timeseries_features import build_ts_features
from src.validation.timeseries_cv import walk_forward_folds
from src.evaluation.metrics import mae, rmse, mape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    run_name = (
        cfg.get("run_name") or f"ts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    save_dir = cfg.get("training", {}).get("save_dir", "artifacts")

    # --- Data (v1 synthetic) ---
    dcfg = cfg["data"]
    df = make_sine_series(
        n_points=int(dcfg.get("n_points", 2000)),
        freq=str(dcfg.get("freq", "H")),
        noise_std=float(dcfg.get("noise_std", 0.2)),
        seed=seed,
    )

    # --- Features ---
    fcfg = cfg["features"]
    lags = list(map(int, fcfg.get("lags", [1, 2, 3, 24])))
    windows = list(map(int, fcfg.get("rolling_windows", [6, 12, 24])))
    add_time = bool(fcfg.get("add_time_features", True))

    feats = build_ts_features(
        df, target_col="y", lags=lags, windows=windows, add_time=add_time
    )

    X = feats.drop(columns=["y"])
    y = feats["y"]

    # --- Walk-forward CV ---
    vcfg = cfg["validation"]
    horizon = int(vcfg.get("horizon", 24))
    n_folds = int(vcfg.get("n_folds", 5))
    min_train_size = int(vcfg.get("min_train_size", 500))
    step_size = int(vcfg.get("step_size", 100))

    folds = walk_forward_folds(
        X.index,
        min_train_size=min_train_size,
        horizon=horizon,
        step_size=step_size,
        n_folds=n_folds,
    )
    if len(folds) == 0:
        raise RuntimeError(
            "No folds created. Increase n_points or reduce min_train_size/horizon."
        )

    alpha = float(cfg.get("model", {}).get("alpha", 1.0))
    fold_metrics = []
    last_model = None

    for i, fold in enumerate(folds, 1):
        X_tr, y_tr = X.loc[fold.train_idx], y.loc[fold.train_idx]
        X_va, y_va = X.loc[fold.val_idx], y.loc[fold.val_idx]

        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)
        fm = {
            "fold": i,
            "mae": mae(y_va, pred),
            "rmse": rmse(y_va, pred),
            "mape": mape(y_va, pred),
            "train_end": str(fold.train_idx[-1]),
            "val_start": str(fold.val_idx[0]),
            "val_end": str(fold.val_idx[-1]),
        }
        fold_metrics.append(fm)
        last_model = model

    # Train final model on all data available (simple v1 policy)
    final_model = Ridge(alpha=alpha, random_state=seed)
    final_model.fit(X, y)

    # Save artifact + metadata
    registry = ModelRegistry(save_dir)
    artifact = registry.get(run_name)
    joblib.dump(final_model, artifact.model_path)

    avg = {
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mape": float(np.mean([m["mape"] for m in fold_metrics])),
    }

    # --- Backtest artifact (v1: last fold window) ---
    last_fold = folds[-1]
    X_tr, y_tr = X.loc[last_fold.train_idx], y.loc[last_fold.train_idx]
    X_va, y_va = X.loc[last_fold.val_idx], y.loc[last_fold.val_idx]

    bt_model = Ridge(alpha=alpha, random_state=seed)
    bt_model.fit(X_tr, y_tr)
    bt_pred = bt_model.predict(X_va)

    backtest_df = pd.DataFrame(
        {"y_true": y_va.values, "y_pred": bt_pred}, index=y_va.index
    )

    # Save CSV
    run_dir = artifact.model_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    backtest_csv = run_dir / "backtest.csv"

    backtest_df.to_csv(backtest_csv)

    # Save plot
    backtest_png = run_dir / "backtest.png"
    plt.figure()
    plt.plot(backtest_df.index, backtest_df["y_true"], label="y_true")
    plt.plot(backtest_df.index, backtest_df["y_pred"], label="y_pred")
    plt.title("Backtest (last fold)")
    plt.xlabel("time")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(backtest_png, dpi=150)
    plt.close()

    registry.save_metadata(
        run_name,
        {
            "run_name": run_name,
            "modality": "timeseries",
            "model": "Ridge",
            "config": cfg,
            "cv": {"folds": fold_metrics, "avg": avg},
            "model_path": str(artifact.model_path),
            "feature_columns": list(X.columns),
            "target_col": "y",
            "artifacts": {
                "backtest_csv": str(backtest_csv),
                "backtest_png": str(backtest_png),
            },
        },
    )

    print(
        f"[OK] run={run_name} avg_mae={avg['mae']:.4f} avg_rmse={avg['rmse']:.4f} avg_mape={avg['mape']:.4f}"
    )
    print(f"[OK] saved: {artifact.model_path}")


if __name__ == "__main__":
    main()
