"""CLI: ``uv run python -m news_eval``."""

from __future__ import annotations

import argparse
from pathlib import Path

from news_eval.readers import DataError, load_data
from news_eval.reporting import build_report, write_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved pipeline and judge outputs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=Path("report.json"))
    parser.add_argument("--conclusion", type=Path, default=Path("conclusion.md"))
    args = parser.parse_args()
    try:
        data, integrity = load_data(args.data_dir)
        report = build_report(data, integrity)
        write_reports(report, args.report, args.conclusion)
    except (DataError, ValueError, OSError) as error:
        parser.error(str(error))

    versions = report["pipeline_evaluation"]["versions"]
    print(f"integrity: {'passed' if integrity['passed'] else 'failed'}")
    for version in ("current", "updated"):
        passed = versions[version]["strict_evaluation_passed"]
        print(f"{version} strict evaluation: {'passed' if passed else 'failed'}")
    can_replace = report["pipeline_evaluation"]["comparison"]["can_replace"]
    print(f"updated can replace current: {'yes' if can_replace else 'no'}")
    print(f"wrote {args.report} and {args.conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
