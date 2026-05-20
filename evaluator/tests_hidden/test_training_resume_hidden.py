from __future__ import annotations

from dataclasses import replace

import pytest

from src.checkpoint import LocalCheckpointStore, checkpoint_key, load_checkpoint
from src.config import load_config
from src.report import build_report
from src.resume import latest_checkpoint, restore_from_checkpoint, run_split_resume, save_training_checkpoint
from src.run_store import InMemoryRunStore
from src.train import init_state, train_until
from src.validation import classify_issue


def test_changed_batch_size_rejects_incompatible_checkpoint(tmp_path):
    config = load_config()
    store = LocalCheckpointStore(tmp_path)
    run_store = InMemoryRunStore()
    run_store.upsert_run(config)
    state = init_state(config)
    train_until(state, config, 3)
    key = save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    checkpoint = load_checkpoint(store, key)
    changed = replace(config, batch_size=config.batch_size + 1)
    with pytest.raises(ValueError):
        restore_from_checkpoint(changed, checkpoint)


def test_corrupt_latest_checkpoint_falls_back_to_valid_prior(tmp_path):
    config = load_config()
    store = LocalCheckpointStore(tmp_path)
    run_store = InMemoryRunStore()
    run_store.upsert_run(config)
    state = init_state(config)
    train_until(state, config, 2)
    good_key = save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    bad_key = checkpoint_key(config.run_id, 4)
    store.put_bytes(bad_key, b"corrupt")
    run_store.record_checkpoint(config.run_id, bad_key, 4, config.config_hash)

    key, checkpoint = latest_checkpoint(run_store, store, config)

    assert key == good_key
    assert checkpoint["step"] == 2


def test_rng_and_optimizer_state_preserve_loss_curve(tmp_path):
    config = load_config()
    result = run_split_resume(config=config, checkpoint_store=LocalCheckpointStore(tmp_path), run_store=InMemoryRunStore())
    assert result["final"]["matches_baseline"]
    assert result["final"]["losses"][-3:] == result["baseline"]["losses"][-3:]


def test_repeated_resume_command_is_idempotent(tmp_path):
    config = load_config()
    first = run_split_resume(config=config, checkpoint_store=LocalCheckpointStore(tmp_path), run_store=InMemoryRunStore())
    second = run_split_resume(config=config, checkpoint_store=LocalCheckpointStore(tmp_path), run_store=InMemoryRunStore())
    assert first["final"]["weights"] == second["final"]["weights"]
    assert first["final"]["saved_key"] == second["final"]["saved_key"]


def test_report_classifies_recoverable_and_unrecoverable_issues(tmp_path):
    config = load_config()
    result = run_split_resume(config=config, checkpoint_store=LocalCheckpointStore(tmp_path), run_store=InMemoryRunStore())
    report = build_report(result, ["corrupt_checkpoint", "config_hash_mismatch"])
    classifications = {item["code"]: item["classification"] for item in report["issues"]}
    assert classifications["corrupt_checkpoint"] == "recoverable"
    assert classifications["config_hash_mismatch"] == "unrecoverable"
    assert report["schema_version"] == "training-resume-report/v1"
