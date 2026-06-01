#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


COUNT_BY_ENTITY = {
    "low": 8,
    "medium": 18,
    "high": 48,
}


def load_profile(path: str | None) -> dict:
    if not path:
        return {}
    profile_path = Path(path)
    if not profile_path.exists():
        return {}
    return json.loads(profile_path.read_text())


def profile_seed(profile: dict, fallback: int) -> int:
    try:
        return int(profile.get("generator_seed") or fallback)
    except (TypeError, ValueError):
        return fallback


def scenario_profile(profile: dict) -> dict:
    value = profile.get("scenario_profile")
    return value if isinstance(value, dict) else {}


def profile_count(profile: dict, fallback: int) -> int:
    entity_count = str(scenario_profile(profile).get("entity_count") or "").lower()
    if entity_count in COUNT_BY_ENTITY:
        return COUNT_BY_ENTITY[entity_count]
    difficulty = str(profile.get("difficulty") or profile.get("difficulty_profile") or "").lower()
    if difficulty == "junior":
        return COUNT_BY_ENTITY["low"]
    if difficulty == "staff":
        return COUNT_BY_ENTITY["high"]
    return fallback


def dataset(count: int = 10) -> list[dict[str, float]]:
    return [{"x": -2.0 + i * 0.5, "y": 2 * (-2.0 + i * 0.5) + 1} for i in range(count)]


def failure_modes(profile: dict) -> list[str]:
    mode = str(scenario_profile(profile).get("failure_modes") or "multi_step")
    if mode == "basic":
        return ["optimizer_state_missing", "single_preemption"]
    if mode == "ambiguous":
        return [
            "optimizer_state_missing",
            "rng_state_missing",
            "scheduler_state_missing",
            "corrupt_latest_checkpoint",
            "config_hash_mismatch",
            "repeated_resume",
            "partial_metadata_write",
        ]
    return [
        "optimizer_state_missing",
        "rng_state_missing",
        "resume_diverged",
        "corrupt_latest_checkpoint",
    ]


def profile_payload(profile: dict, seed: int, count: int) -> dict:
    rng = random.Random(seed)
    difficulty = str(profile.get("difficulty") or profile.get("difficulty_profile") or "senior").lower()
    if difficulty not in {"junior", "senior", "staff"}:
        difficulty = "senior"
    modes = failure_modes(profile)
    steps = {"junior": 6, "senior": 12, "staff": 24}[difficulty]
    checkpoint_every = {"junior": 3, "senior": 4, "staff": 4}[difficulty]
    crash_points = {
        "junior": [3],
        "senior": [4, 8],
        "staff": [4, 11, 17],
    }[difficulty]
    dataset_rows = dataset(count)
    return {
        "schema_version": "training-resume-fixture/v1",
        "difficulty": difficulty,
        "generator_seed": seed,
        "config": {
            "run_id": f"generated-{difficulty}-{seed}",
            "seed": seed,
            "steps": steps,
            "batch_size": 2 + (seed % 3),
            "learning_rate": round(0.04 + rng.random() * 0.06, 4),
            "checkpoint_every": checkpoint_every,
            "bucket": "checkpoints",
        },
        "dataset_summary": {
            "row_count": len(dataset_rows),
            "first": dataset_rows[0],
            "last": dataset_rows[-1],
        },
        "crash_plan": {
            "crash_points": crash_points,
            "preemption_reasons": ["spot_interruption"] if difficulty == "junior" else ["spot_interruption", "worker_eviction"],
            "requires_idempotent_resume": difficulty in {"senior", "staff"},
        },
        "checkpoint_cases": [
            {
                "case_id": f"case-{index:02d}",
                "failure_mode": mode,
                "expected_recoverable": mode not in {"config_hash_mismatch"},
                "requires_full_state": mode in {"optimizer_state_missing", "rng_state_missing", "scheduler_state_missing", "resume_diverged"},
            }
            for index, mode in enumerate(modes)
        ],
        "reporting_depth": scenario_profile(profile).get("reporting_depth", "operator"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out", default="candidate/fixtures/public/tiny_dataset.jsonl")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    seed = profile_seed(profile, args.seed)
    count = profile_count(profile, args.count)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not profile:
        out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset(count)))
        print(f"wrote {count} dataset rows to {out}")
        return

    out.write_text(json.dumps(profile_payload(profile, seed, count), indent=2, sort_keys=True) + "\n")
    print(f"wrote generated training profile to {out}")


if __name__ == "__main__":
    main()
