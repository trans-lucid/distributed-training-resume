CREATE TABLE IF NOT EXISTS training_runs (
  run_id TEXT PRIMARY KEY,
  config_hash TEXT NOT NULL,
  seed INTEGER NOT NULL,
  total_steps INTEGER NOT NULL,
  batch_size INTEGER NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS checkpoints (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES training_runs(run_id),
  checkpoint_key TEXT NOT NULL,
  step INTEGER NOT NULL,
  config_hash TEXT NOT NULL,
  valid BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, checkpoint_key)
);

CREATE TABLE IF NOT EXISTS validation_history (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  check_name TEXT NOT NULL,
  status TEXT NOT NULL,
  details JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step ON checkpoints(run_id, step DESC);
