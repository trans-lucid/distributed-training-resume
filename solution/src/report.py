from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .checkpoint import S3CheckpointStore
from .config import load_config
from .resume import run_split_resume
from .run_store import PostgresRunStore
from .validation import classify_issue


def build_report(result: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "training-resume-report/v1",
        "matches_baseline": result["final"].get("matches_baseline", False),
        "issues": [{"code": issue, "classification": classify_issue(issue)} for issue in issues],
        "final": result["final"],
        "baseline": result["baseline"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/resume_report.json")
    args = parser.parse_args()
    config = load_config()
    run_store = PostgresRunStore()
    try:
        run_store.reset()
        result = run_split_resume(config=config, checkpoint_store=S3CheckpointStore(config.bucket), run_store=run_store)
        issues = [] if result["final"]["matches_baseline"] else ["resume_diverged"]
        report = build_report(result, issues)
    finally:
        run_store.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    (out.parent / "summary.md").write_text(
        f"# Training Resume Summary\n\nMatches baseline: {report['matches_baseline']}\nIssues: {len(report['issues'])}\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
