from __future__ import annotations

from src.checkpoint import LocalCheckpointStore, load_checkpoint
from src.config import load_config
from src.resume import run_split_resume, save_training_checkpoint
from src.run_store import InMemoryRunStore
from src.train import init_state, train_until


def test_reference_solution_checkpoint_and_resume_are_complete(tmp_path):
    config = load_config()
    store = LocalCheckpointStore(tmp_path)
    run_store = InMemoryRunStore()
    run_store.upsert_run(config)
    state = init_state(config)
    train_until(state, config, config.crash_step)
    key = save_training_checkpoint(config=config, state=state, checkpoint_store=store, run_store=run_store)
    checkpoint = load_checkpoint(store, key)

    assert "optimizer_state" in checkpoint
    assert "scheduler_state" in checkpoint
    assert "rng_state" in checkpoint

    result = run_split_resume(config=config, checkpoint_store=store, run_store=InMemoryRunStore())
    assert result["final"]["matches_baseline"]
