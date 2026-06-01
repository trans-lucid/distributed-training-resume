# Distributed Training Resume

This is an internal Translucid challenge template, not a generated candidate repo.

The template generates a production-shaped ML infra challenge around checkpointing, deterministic resume after crash/preemption, run metadata validation, corrupt checkpoint handling, and training recovery reporting.

The generated candidate repo intentionally contains flawed starter code. Candidates must repair checkpoint save/load, optimizer/RNG/scheduler continuity, run config validation, corrupt checkpoint fallback, and report quality.

## Local Simulator

Validation uses local services only:

- MinIO for S3-compatible checkpoint storage.
- Postgres for run metadata, checkpoint records, and validation history.
- CPU PyTorch for a tiny deterministic training loop.

No GPU, cloud credentials, external object storage, or customer data are required.

## Time Budget

- Expected candidate coding time: 75-100 minutes for senior ML infra/research engineering candidates.
- Staff variant: up to 120 minutes with stricter corrupt checkpoint and metadata cases.
- Setup time after cached images: under 10 minutes on a normal laptop.
- Docker image pull cost: Postgres and MinIO.

## Validation

```bash
make validate-solution
make validate-candidate-main-expected-failure
make render
make check-render
make check-published-repo
make scan-safety
make validate-personalization
make validate-rendered-smoke
make validate-docker-integration
make validate
```

Expected starter failure markers:

- `optimizer_state_missing`
- `rng_state_missing`
- `resume_diverged`
- `corrupt_checkpoint_unhandled`

## For Challenge Creation Agents

Do not infer how to use this template from README prose.

Read `translucid-template.json`.

Normal use:

```bash
make render
make check-render
make check-published-repo
make scan-safety
make validate-personalization
make validate-solution
make validate-candidate-main-expected-failure
make validate-docker-integration
```

Use:

- `generated/main` as candidate-facing main branch
- `generated/solution` as private solution/evaluator branch

Do not manually copy `candidate/` to root.
Do not manually restructure `solution/`.
Do not edit hidden tests or evaluator imports unless a validation command fails and the exact blocker is recorded.
