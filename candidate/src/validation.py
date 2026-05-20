from __future__ import annotations

from typing import Any

from .config import TrainConfig


def validate_checkpoint_metadata(checkpoint: dict[str, Any], config: TrainConfig) -> list[str]:
    issues: list[str] = []
    if checkpoint.get("run_id") != config.run_id:
        issues.append("run_id_mismatch")
    # Starter bug: config_hash is present but compatibility is not enforced.
    return issues


def classify_issue(issue: str) -> str:
    recoverable = {"corrupt_checkpoint", "checkpoint_missing", "older_checkpoint_used"}
    return "recoverable" if issue in recoverable else "unrecoverable"
