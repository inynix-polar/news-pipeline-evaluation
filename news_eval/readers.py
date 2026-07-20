from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from news_eval.constants import EXPECTED_ROW_COUNTS, JUDGE_THRESHOLDS, VERSIONS


FILES = {
    "goldens": "goldens.jsonl",
    "pipeline_outputs": "pipeline_outputs.jsonl",
    "judge_outputs": "judge_outputs.jsonl",
}
KEY_FIELDS = {
    "goldens": ("id",),
    "pipeline_outputs": ("case_id", "version"),
    "judge_outputs": ("case_id", "version", "criterion"),
}


class DataError(ValueError):
    def __init__(self, message: str, integrity: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.integrity = integrity


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _non_finite(value: str) -> None:
    raise ValueError(f"non-finite number {value!r}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load strict UTF-8 JSONL: one non-empty JSON object per line."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DataError(f"cannot read {source}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise DataError(f"{source}: blank line {number}")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_json_object,
                parse_constant=_non_finite,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DataError(f"{source}: invalid JSON at line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise DataError(f"{source}: line {number} must contain a JSON object")
        rows.append(row)
    return rows


def _keys(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> tuple[list[tuple[str, ...]], list[str]]:
    keys: list[tuple[str, ...]] = []
    errors: list[str] = []
    for number, row in enumerate(rows, 1):
        values = tuple(row.get(field) for field in fields)
        if any(not isinstance(value, str) or not value for value in values):
            errors.append(f"row {number}: {fields!r} must be non-empty strings")
        else:
            keys.append(values)  # type: ignore[arg-type]
    return keys, errors


def _key_dicts(
    keys: set[tuple[str, ...]] | list[tuple[str, ...]], fields: tuple[str, ...]
) -> list[dict[str, str]]:
    return [dict(zip(fields, key, strict=True)) for key in sorted(keys)]


def _file_report(
    name: str,
    rows: list[dict[str, Any]],
    path: Path,
    *,
    expected_keys: set[tuple[str, ...]] | None = None,
    golden_ids: set[str] | None = None,
) -> dict[str, Any]:
    fields = KEY_FIELDS[name]
    keys, key_errors = _keys(rows, fields)
    counts = Counter(keys)
    duplicates = {key for key, count in counts.items() if count > 1}
    actual_keys = set(keys)
    missing = expected_keys - actual_keys if expected_keys is not None else set()
    unexpected = actual_keys - expected_keys if expected_keys is not None else set()
    orphans = sorted(
        {
            row.get("case_id")
            for row in rows
            if golden_ids is not None
            and isinstance(row.get("case_id"), str)
            and row["case_id"] not in golden_ids
        }
    )

    row_count = {
        "expected": EXPECTED_ROW_COUNTS[name],
        "actual": len(rows),
        "passed": len(rows) == EXPECTED_ROW_COUNTS[name],
    }
    unique = {
        "fields": list(fields),
        "passed": not duplicates and not key_errors,
        "duplicates": _key_dicts(duplicates, fields),
        "errors": key_errors,
    }
    combinations = {
        "applicable": expected_keys is not None,
        "passed": not missing and not unexpected,
        "missing": _key_dicts(missing, fields),
        "unexpected": _key_dicts(unexpected, fields),
    }
    if expected_keys is not None:
        combinations["expected_count"] = len(expected_keys)
    orphan_check = {"passed": not orphans, "values": orphans}
    return {
        "passed": all(
            check["passed"]
            for check in (row_count, unique, combinations, orphan_check)
        ),
        "row_count": row_count,
        "unique_key": unique,
        "complete_combinations": combinations,
        "orphan_case_ids": orphan_check,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def load_data(data_dir: str | Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    directory = Path(data_dir)
    paths = {name: directory / file_name for name, file_name in FILES.items()}
    data = {name: load_jsonl(path) for name, path in paths.items()}
    golden_ids = {
        row["id"]
        for row in data["goldens"]
        if isinstance(row.get("id"), str) and row["id"]
    }
    pipeline_keys = {
        (case_id, version) for case_id in golden_ids for version in VERSIONS
    }
    judge_keys = {
        (case_id, version, criterion)
        for case_id in golden_ids
        for version in VERSIONS
        for criterion in JUDGE_THRESHOLDS
    }
    reports = {
        "goldens.jsonl": _file_report(
            "goldens", data["goldens"], paths["goldens"]
        ),
        "pipeline_outputs.jsonl": _file_report(
            "pipeline_outputs",
            data["pipeline_outputs"],
            paths["pipeline_outputs"],
            expected_keys=pipeline_keys,
            golden_ids=golden_ids,
        ),
        "judge_outputs.jsonl": _file_report(
            "judge_outputs",
            data["judge_outputs"],
            paths["judge_outputs"],
            expected_keys=judge_keys,
            golden_ids=golden_ids,
        ),
    }
    integrity = {
        "passed": all(item["passed"] for item in reports.values()),
        "files": reports,
    }
    if not integrity["passed"]:
        failed = ", ".join(name for name, item in reports.items() if not item["passed"])
        raise DataError(f"dataset integrity failed: {failed}", integrity)
    return data, integrity
