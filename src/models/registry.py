from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

ARTIFACTS_DIR = Path("artifacts")


@dataclass
class ModelArtifact:
    model_path: Path
    metadata_path: Path


class ModelRegistry:
    def __init__(self, root: str = "artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, run_name: str) -> ModelArtifact:
        run_dir = self.root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return ModelArtifact(
            model_path=run_dir / "model.pkl",
            metadata_path=run_dir / "metadata.json",
        )

    def save_metadata(self, run_name: str, metadata: Dict[str, Any]) -> None:
        artifact = self.get(run_name)
        artifact.metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


@dataclass
class RunInfo:
    run_name: str
    task: str
    metrics: Dict[str, Any]
    artifact_dir: str
    is_active: bool


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_active_run_name() -> Optional[str]:
    latest = ARTIFACTS_DIR / "latest.txt"
    if not latest.exists():
        return None
    return latest.read_text(encoding="utf-8").strip().strip('"').strip()


def list_runs() -> List[RunInfo]:
    active = get_active_run_name()

    runs: List[RunInfo] = []
    if not ARTIFACTS_DIR.exists():
        return runs

    for p in ARTIFACTS_DIR.iterdir():
        if not p.is_dir():
            continue

        meta_path = p / "metadata.json"
        meta = _safe_read_json(meta_path) if meta_path.exists() else {}

        task = str(meta.get("task", meta.get("problem_type", "unknown")))
        metrics = dict(meta.get("metrics", {}))

        runs.append(
            RunInfo(
                run_name=p.name,
                task=task,
                metrics=metrics,
                artifact_dir=str(p),
                is_active=(p.name == active),
            )
        )

    # newest first (best-effort)
    runs.sort(key=lambda r: Path(r.artifact_dir).stat().st_mtime, reverse=True)
    return runs


def promote_run(run_name: str) -> None:
    run_dir = ARTIFACTS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_name}")

    model_path = run_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found for run: {run_name}")

    (ARTIFACTS_DIR / "latest.txt").write_text(run_name, encoding="utf-8")
