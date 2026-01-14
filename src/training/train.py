import argparse
from datetime import datetime
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.utils.config import load_yaml
from src.utils.reproducibility import set_global_seed
from src.models.registry import ModelRegistry
from src.data.loaders import load_breast_cancer_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    run_name = (
        cfg.get("run_name") or f"tabular_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    save_dir = cfg.get("training", {}).get("save_dir", "artifacts")

    ds = load_breast_cancer_dataset(
        test_size=float(cfg.get("data", {}).get("test_size", 0.2)),
        seed=seed,
        stratify=bool(cfg.get("data", {}).get("stratify", True)),
    )

    model = LogisticRegression(max_iter=200)
    model.fit(ds.X_train, ds.y_train)

    preds = model.predict(ds.X_val)
    acc = float(accuracy_score(ds.y_val, preds))

    auc = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(ds.X_val)[:, 1]
        auc = float(roc_auc_score(ds.y_val, proba))

    registry = ModelRegistry(save_dir)
    artifact = registry.get(run_name)
    joblib.dump(model, artifact.model_path)

    registry.save_metadata(
        run_name,
        {
            "run_name": run_name,
            "modality": "tabular",
            "model": "LogisticRegression",
            "metrics": {"accuracy": acc, "roc_auc": auc},
            "model_path": str(artifact.model_path),
        },
    )

    auc_str = f"{auc:.4f}" if auc is not None else "NA"
    print(f"[OK] run={run_name} accuracy={acc:.4f} roc_auc={auc_str}")

    print(f"[OK] saved: {artifact.model_path}")


if __name__ == "__main__":
    main()
