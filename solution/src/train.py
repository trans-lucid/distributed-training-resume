from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ROOT, TrainConfig, load_config


@dataclass
class TrainingState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int
    losses: list[float]


def load_dataset(path: str | Path = "fixtures/public/tiny_dataset.jsonl") -> tuple[torch.Tensor, torch.Tensor]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    rows = [json.loads(line) for line in resolved.read_text().splitlines() if line.strip()]
    xs = torch.tensor([[float(row["x"])] for row in rows], dtype=torch.float32)
    ys = torch.tensor([[float(row["y"])] for row in rows], dtype=torch.float32)
    return xs, ys


def init_state(config: TrainConfig) -> TrainingState:
    torch.manual_seed(config.seed)
    model = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    return TrainingState(model=model, optimizer=optimizer, scheduler=scheduler, step=0, losses=[])


def train_until(state: TrainingState, config: TrainConfig, target_step: int) -> TrainingState:
    xs, ys = load_dataset()
    for _ in range(state.step, target_step):
        indices = torch.randint(0, len(xs), (config.batch_size,))
        prediction = state.model(xs[indices])
        loss = torch.nn.functional.mse_loss(prediction, ys[indices])
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()
        state.scheduler.step()
        state.step += 1
        state.losses.append(float(loss.detach().cpu()))
    return state


def state_fingerprint(state: TrainingState) -> dict[str, Any]:
    tensors = [param.detach().cpu().flatten() for param in state.model.parameters()]
    weights = torch.cat(tensors)
    return {
        "step": state.step,
        "weights": [round(float(value), 10) for value in weights],
        "losses": [round(value, 10) for value in state.losses],
        "lr": round(float(state.optimizer.param_groups[0]["lr"]), 10),
    }


def uninterrupted_fingerprint(config: TrainConfig | None = None) -> dict[str, Any]:
    config = config or load_config()
    state = init_state(config)
    train_until(state, config, config.steps)
    return state_fingerprint(state)


if __name__ == "__main__":
    print(json.dumps(uninterrupted_fingerprint(), indent=2))
