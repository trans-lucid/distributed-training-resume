from __future__ import annotations

from typing import Any

from .checkpoint import LocalCheckpointStore, S3CheckpointStore, checkpoint_key, build_checkpoint, load_checkpoint, save_checkpoint
from .config import TrainConfig, load_config
from .run_store import InMemoryRunStore
from .train import TrainingState, init_state, state_fingerprint, train_until, uninterrupted_fingerprint
from .validation import validate_checkpoint_metadata


def save_training_checkpoint(
    *,
    config: TrainConfig,
    state: TrainingState,
    checkpoint_store: LocalCheckpointStore | S3CheckpointStore,
    run_store: InMemoryRunStore,
) -> str:
    key = checkpoint_key(config.run_id, state.step)
    checkpoint = build_checkpoint(
        config=config,
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        step=state.step,
        loss=state.losses[-1] if state.losses else 0.0,
    )
    save_checkpoint(checkpoint_store, key, checkpoint)
    run_store.record_checkpoint(config.run_id, key, state.step, config.config_hash)
    return key


def restore_from_checkpoint(config: TrainConfig, checkpoint: dict[str, Any]) -> TrainingState:
    issues = validate_checkpoint_metadata(checkpoint, config)
    if issues:
        raise ValueError(f"incompatible checkpoint: {issues}")
    state = init_state(config)
    state.model.load_state_dict(checkpoint["model_state"])
    # Starter bug: optimizer, scheduler, and RNG are silently reset.
    state.step = int(checkpoint["step"])
    state.losses = [float(checkpoint.get("loss", 0.0))]
    return state


def latest_checkpoint(run_store: InMemoryRunStore, checkpoint_store: LocalCheckpointStore | S3CheckpointStore, config: TrainConfig) -> tuple[str, dict[str, Any]]:
    checkpoints = run_store.list_checkpoints(config.run_id)
    if not checkpoints:
        raise FileNotFoundError("no checkpoints found")
    latest = checkpoints[0]
    # Starter bug: corrupt latest checkpoint raises instead of falling back.
    return latest["checkpoint_key"], load_checkpoint(checkpoint_store, latest["checkpoint_key"])


def run_split_resume(
    *,
    config: TrainConfig | None = None,
    checkpoint_store: LocalCheckpointStore | S3CheckpointStore,
    run_store: InMemoryRunStore,
) -> dict[str, Any]:
    config = config or load_config()
    run_store.upsert_run(config, status="running")
    first = init_state(config)
    train_until(first, config, config.crash_step)
    saved_key = save_training_checkpoint(config=config, state=first, checkpoint_store=checkpoint_store, run_store=run_store)
    _key, checkpoint = latest_checkpoint(run_store, checkpoint_store, config)
    resumed = restore_from_checkpoint(config, checkpoint)
    train_until(resumed, config, config.steps)
    final = state_fingerprint(resumed)
    baseline = uninterrupted_fingerprint(config)
    final["saved_key"] = saved_key
    final["matches_baseline"] = final["weights"] == baseline["weights"] and final["losses"][-3:] == baseline["losses"][-3:]
    return {"final": final, "baseline": baseline}
