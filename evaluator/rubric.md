# Evaluator Rubric

Total: 100 points

- Checkpoint completeness: 25
- Optimizer, scheduler, and RNG resume correctness: 25
- Run metadata validation: 15
- Corrupt checkpoint fallback: 15
- Training recovery report quality: 10
- Code quality and simulator discipline: 10

Major deductions:

- Saving only model weights.
- Resetting optimizer, scheduler, or RNG on resume.
- Resuming incompatible run configurations silently.
- Crashing on corrupt latest checkpoint when a prior valid checkpoint exists.
- Bypassing MinIO or Postgres in the integration path.
