#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def dataset() -> list[dict[str, float]]:
    return [{"x": -2.0 + i * 0.5, "y": 2 * (-2.0 + i * 0.5) + 1} for i in range(10)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="candidate/fixtures/public/tiny_dataset.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in dataset()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
