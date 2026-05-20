# Source Dossier: Distributed Training Resume

This template uses public sources only as architecture references. The candidate code, fixtures, tests, hidden evaluator, and rubrics are original Translucid-owned material.

## Sources Studied

- PyTorch checkpointing docs: complete checkpoint contents for resumable training.
- PyTorch torchrun fault-tolerance tutorial: snapshot/restart vocabulary for preemptable training jobs.
- Hugging Face Accelerate checkpointing docs: save/load state for model, optimizer, RNG, and scalers.
- PyTorch Lightning checkpoint docs: complete internal state concept including epoch, global step, optimizers, and schedulers.
- MinIO local object storage docs: S3-compatible checkpoint storage simulation.

## Allowed Reuse

- Architecture concepts such as model state, optimizer state, scheduler state, RNG state, config hash, checkpoint cadence, and preemption recovery.
- Generic checkpoint metadata vocabulary.
- Local emulator patterns for S3-compatible storage and run metadata.

## Forbidden

- Copying source code from PyTorch, Accelerate, Lightning, MinIO, or customer repositories.
- Copying real training data, model weights, production configs, or customer checkpoints.
- Requiring GPUs, live cloud object storage, or credentials.
- Turning a connected startup repo into the generated challenge repo unless source-slice mode is explicitly approved.
