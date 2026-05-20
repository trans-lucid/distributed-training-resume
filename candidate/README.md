# Distributed Training Resume

You are inheriting a fragile training driver that loses progress or resumes incorrectly after crash/preemption.

Repair checkpoint save/load, optimizer/RNG/scheduler recovery, run metadata validation, corrupt checkpoint handling, and recovery reporting.

## Local Services

```txt
MinIO      S3-compatible checkpoint storage
Postgres   run metadata, checkpoint rows, validation history
PyTorch    CPU-only deterministic tiny model
```

No GPU, cloud credentials, or customer data are needed.

## Commands

```bash
make dev
make seed
make test
make test-integration
make run
make clean
```

Private tests use harder corrupt checkpoint, config mismatch, RNG continuity, repeated resume, and report classification cases. Do not hardcode fixture outputs or bypass MinIO/Postgres.
