from __future__ import annotations

import pytest

from load_target import load


checkpoint_module = load("src.checkpoint")
config_module = load("src.config")
resume_module = load("src.resume")
run_store_module = load("src.run_store")
train_module = load("src.train")


def test_checkpoint_file_is_created_and_model_weights_load(tmp_path):
    config = config_module.load_config()
    store = checkpoint_module.LocalCheckpointStore(tmp_path)
    run_store = run_store_module.InMemoryRunStore()
    run_store.upsert_run(config)
    state = train_module.init_state(config)
    train_module.train_until(state, config, 2)

    key = resume_module.save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    loaded = checkpoint_module.load_checkpoint(store, key)

    assert store.exists(key)
    assert "model_state" in loaded
    assert loaded["step"] == 2


def test_checkpoint_contains_optimizer_scheduler_state(tmp_path):
    config = config_module.load_config()
    store = checkpoint_module.LocalCheckpointStore(tmp_path)
    run_store = run_store_module.InMemoryRunStore()
    run_store.upsert_run(config)
    state = train_module.init_state(config)
    train_module.train_until(state, config, 3)

    key = resume_module.save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    loaded = checkpoint_module.load_checkpoint(store, key)

    assert "optimizer_state" in loaded, "optimizer_state_missing: checkpoint must include optimizer state"
    assert "scheduler_state" in loaded, "optimizer_state_missing: checkpoint must include scheduler state"


def test_checkpoint_contains_rng_state(tmp_path):
    config = config_module.load_config()
    store = checkpoint_module.LocalCheckpointStore(tmp_path)
    run_store = run_store_module.InMemoryRunStore()
    run_store.upsert_run(config)
    state = train_module.init_state(config)
    train_module.train_until(state, config, 3)

    key = resume_module.save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    loaded = checkpoint_module.load_checkpoint(store, key)

    assert "rng_state" in loaded, "rng_state_missing: checkpoint must include RNG state"


def test_split_resume_matches_uninterrupted_baseline(tmp_path):
    config = config_module.load_config()
    result = resume_module.run_split_resume(
        config=config,
        checkpoint_store=checkpoint_module.LocalCheckpointStore(tmp_path),
        run_store=run_store_module.InMemoryRunStore(),
    )

    assert result["final"]["matches_baseline"], "resume_diverged: resumed run must match uninterrupted baseline"


def test_corrupt_latest_checkpoint_falls_back_to_prior_valid(tmp_path):
    config = config_module.load_config()
    store = checkpoint_module.LocalCheckpointStore(tmp_path)
    run_store = run_store_module.InMemoryRunStore()
    run_store.upsert_run(config)
    state = train_module.init_state(config)
    train_module.train_until(state, config, 2)
    good_key = resume_module.save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    bad_key = checkpoint_module.checkpoint_key(config.run_id, 3)
    store.put_bytes(bad_key, b"not a torch checkpoint")
    run_store.record_checkpoint(config.run_id, bad_key, 3, config.config_hash, valid=True)

    try:
        key, checkpoint = resume_module.latest_checkpoint(run_store, store, config)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"corrupt_checkpoint_unhandled: should fall back to prior valid checkpoint, got {exc}")

    assert key == good_key, "corrupt_checkpoint_unhandled: corrupt latest checkpoint should not be selected"
    assert checkpoint["step"] == 2
