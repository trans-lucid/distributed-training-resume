from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DATABASE_URL = os.environ.get("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/training")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def find_runtime_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "fixtures" / "public").exists():
            return parent
    template_candidate = path.parents[2] / "candidate"
    if (template_candidate / "fixtures" / "public").exists():
        return template_candidate
    return path.parents[1]


ROOT = find_runtime_root(Path(__file__).resolve())


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    seed: int
    steps: int
    crash_step: int
    learning_rate: float
    batch_size: int
    checkpoint_every: int
    bucket: str = "checkpoints"

    @property
    def config_hash(self) -> str:
        payload = {
            "seed": self.seed,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "checkpoint_every": self.checkpoint_every,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_config(path: str | Path = "fixtures/public/config.yaml") -> TrainConfig:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    data: dict[str, Any] = yaml.safe_load(resolved.read_text())
    return TrainConfig(
        run_id=data["run_id"],
        seed=int(data["seed"]),
        steps=int(data["steps"]),
        crash_step=int(data["crash_step"]),
        learning_rate=float(data["learning_rate"]),
        batch_size=int(data["batch_size"]),
        checkpoint_every=int(data["checkpoint_every"]),
        bucket=data.get("bucket", "checkpoints"),
    )
