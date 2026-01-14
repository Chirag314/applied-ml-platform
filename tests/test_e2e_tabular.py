import json
import subprocess
import sys
from pathlib import Path


def test_train_creates_artifacts():
    repo = Path(__file__).resolve().parents[1]
    cfg = repo / "configs" / "example_tabular.yaml"

    # run training
    subprocess.check_call(
        [sys.executable, "-m", "src.training.train", "--config", str(cfg)], cwd=repo
    )

    # check artifacts
    model_path = repo / "artifacts" / "tabular_baseline_v1" / "model.pkl"
    meta_path = repo / "artifacts" / "tabular_baseline_v1" / "metadata.json"

    assert model_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert "metrics" in meta
