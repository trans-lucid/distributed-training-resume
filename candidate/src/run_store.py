from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL, ROOT, TrainConfig


class InMemoryRunStore:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.checkpoints: list[dict[str, Any]] = []
        self.validations: list[dict[str, Any]] = []

    def upsert_run(self, config: TrainConfig, status: str = "running") -> None:
        self.runs[config.run_id] = {
            "run_id": config.run_id,
            "config_hash": config.config_hash,
            "seed": config.seed,
            "total_steps": config.steps,
            "batch_size": config.batch_size,
            "status": status,
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return dict(self.runs[run_id]) if run_id in self.runs else None

    def record_checkpoint(self, run_id: str, checkpoint_key: str, step: int, config_hash: str, valid: bool = True) -> None:
        if not any(item["run_id"] == run_id and item["checkpoint_key"] == checkpoint_key for item in self.checkpoints):
            self.checkpoints.append(
                {
                    "run_id": run_id,
                    "checkpoint_key": checkpoint_key,
                    "step": step,
                    "config_hash": config_hash,
                    "valid": valid,
                }
            )

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        return sorted([dict(item) for item in self.checkpoints if item["run_id"] == run_id], key=lambda item: item["step"], reverse=True)

    def record_validation(self, run_id: str, check_name: str, status: str, details: dict[str, Any]) -> None:
        self.validations.append({"run_id": run_id, "check_name": check_name, "status": status, "details": details})

    def list_validations(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self.validations if item["run_id"] == run_id]


class PostgresRunStore:
    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self.connection = wait_for_postgres(database_url)

    def close(self) -> None:
        self.connection.close()

    def reset(self) -> None:
        with self.connection.cursor() as cur:
            cur.execute("TRUNCATE validation_history, checkpoints, training_runs RESTART IDENTITY CASCADE")
        self.connection.commit()

    def upsert_run(self, config: TrainConfig, status: str = "running") -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (run_id, config_hash, seed, total_steps, batch_size, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id)
                DO UPDATE SET status = EXCLUDED.status, updated_at = now()
                """,
                (config.run_id, config.config_hash, config.seed, config.steps, config.batch_size, status),
            )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM training_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def record_checkpoint(self, run_id: str, checkpoint_key: str, step: int, config_hash: str, valid: bool = True) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO checkpoints (run_id, checkpoint_key, step, config_hash, valid)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (run_id, checkpoint_key)
                DO UPDATE SET step = EXCLUDED.step, valid = EXCLUDED.valid
                """,
                (run_id, checkpoint_key, step, config_hash, valid),
            )
        self.connection.commit()

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM checkpoints WHERE run_id = %s ORDER BY step DESC, id DESC", (run_id,))
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    def record_validation(self, run_id: str, check_name: str, status: str, details: dict[str, Any]) -> None:
        with self.connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO validation_history (run_id, check_name, status, details)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (run_id, check_name, status, json.dumps(details)),
            )
        self.connection.commit()

    def list_validations(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM validation_history WHERE run_id = %s ORDER BY id", (run_id,))
            rows = cur.fetchall()
        return [dict(row) for row in rows]


def wait_for_postgres(database_url: str = DATABASE_URL, attempts: int = 50) -> psycopg.Connection:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            connection = psycopg.connect(database_url)
            connection.execute("SELECT 1")
            return connection
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"postgres not ready: {last_error}")


def migrate(database_url: str = DATABASE_URL) -> None:
    connection = wait_for_postgres(database_url)
    try:
        connection.execute((ROOT / "migrations" / "001_init.sql").read_text())
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "migrate"
    if command == "migrate":
        migrate()
        print("postgres migration complete")
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
