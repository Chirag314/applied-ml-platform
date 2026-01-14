from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict


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
