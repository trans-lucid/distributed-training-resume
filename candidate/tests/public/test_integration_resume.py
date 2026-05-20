from __future__ import annotations

from load_target import load


checkpoint_module = load("src.checkpoint")
config_module = load("src.config")
resume_module = load("src.resume")
run_store_module = load("src.run_store")


def test_public_docker_training_resume_path_uses_minio_and_postgres():
    config = config_module.load_config()
    run_store = run_store_module.PostgresRunStore()
    try:
        run_store.reset()
        checkpoint_store = checkpoint_module.S3CheckpointStore(config.bucket)
        result = resume_module.run_split_resume(config=config, checkpoint_store=checkpoint_store, run_store=run_store)
        checkpoints = run_store.list_checkpoints(config.run_id)

        assert checkpoints, "optimizer_state_missing: integration should record checkpoint metadata in Postgres"
        assert checkpoint_store.exists(result["final"]["saved_key"]), "rng_state_missing: integration should write checkpoint bytes to MinIO"
        assert result["final"]["matches_baseline"], "resume_diverged: Docker-backed resumed run must match uninterrupted baseline"
    finally:
        run_store.close()
