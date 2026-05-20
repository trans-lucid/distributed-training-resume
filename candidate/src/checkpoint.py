from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import boto3
import torch
from botocore.client import Config

from .config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, TrainConfig


class LocalCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, payload: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def get_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()


class S3CheckpointStore:
    def __init__(self, bucket: str = "checkpoints", endpoint_url: str = S3_ENDPOINT_URL) -> None:
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            config=Config(signature_version="s3v4"),
        )
        wait_for_s3(self.client)

    def ensure_bucket(self) -> None:
        existing = [bucket["Name"] for bucket in self.client.list_buckets().get("Buckets", [])]
        if self.bucket not in existing:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, key: str, payload: bytes) -> None:
        self.ensure_bucket()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False


def wait_for_s3(client: Any, attempts: int = 50) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            client.list_buckets()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"minio not ready: {last_error}")


def checkpoint_key(run_id: str, step: int) -> str:
    return f"{run_id}/step-{step:06d}.pt"


def serialize_checkpoint(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize_checkpoint(payload: bytes) -> dict[str, Any]:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def build_checkpoint(
    *,
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    loss: float,
) -> dict[str, Any]:
    # Starter bug: model weights alone are not enough for deterministic resume.
    return {
        "schema_version": "training-checkpoint/v1",
        "run_id": config.run_id,
        "step": step,
        "config_hash": config.config_hash,
        "model_state": model.state_dict(),
        "loss": loss,
    }


def save_checkpoint(store: LocalCheckpointStore | S3CheckpointStore, key: str, checkpoint: dict[str, Any]) -> None:
    store.put_bytes(key, serialize_checkpoint(checkpoint))


def load_checkpoint(store: LocalCheckpointStore | S3CheckpointStore, key: str) -> dict[str, Any]:
    return deserialize_checkpoint(store.get_bytes(key))


def setup_storage() -> None:
    S3CheckpointStore().ensure_bucket()
    print("checkpoint bucket ready")


def main() -> None:
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "setup-storage"
    if command == "setup-storage":
        setup_storage()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
