"""Strict offline evaluation of pipeline and saved judge outputs."""
from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from news_eval.constants import INVALID_ROLE, JUDGE_THRESHOLDS, PIPELINE_THRESHOLDS
from news_eval.constants import ROLES, VALID_SCORES, VERSIONS
from news_eval.matching import match_entities

class EvaluationError(ValueError):
    """Input records do not satisfy the evaluation contract."""

def _valid_score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value in VALID_SCORES
    )

def _decision(entity: Mapping[str, Any]) -> bool | None:
    a_value = entity.get("a_event")
    c_value = entity.get("c_persistence")
    if not (_valid_score(a_value) and _valid_score(c_value)):
        return None
    return (a_value + c_value) / 2 >= 0.8

def _metric(
    numerator: int,
    denominator: int,
    zero_value: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    value = numerator / denominator if denominator else zero_value
    result: dict[str, Any] = {
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
    }
    if threshold is not None:
        result["threshold"] = threshold
        result["passed"] = value >= threshold
    return result

def _validated_goldens(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    goldens: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row_number, golden in enumerate(records, 1):
        if not isinstance(golden, Mapping):
            raise EvaluationError(f"golden row {row_number} must be an object")
        case_id = golden.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationError(f"golden row {row_number}: id must be a string")
        if case_id in seen:
            raise EvaluationError(f"duplicate golden id {case_id!r}")
        seen.add(case_id)
        output = golden.get("expected_output")
        entities = output.get("entities") if isinstance(output, Mapping) else None
        if not isinstance(entities, list) or any(
            not isinstance(entity, Mapping) for entity in entities
        ):
            raise EvaluationError(
                f"golden {case_id!r}: expected_output.entities must be an array of objects"
            )
        for index, entity in enumerate(entities):
            prefix = f"golden {case_id!r} entity {index}"
            if not isinstance(entity.get("name"), str) or not entity["name"]:
                raise EvaluationError(f"{prefix}: name must be a non-empty string")
            aliases = entity.get("aliases")
            if not isinstance(aliases, list) or any(
                not isinstance(alias, str) for alias in aliases
            ):
                raise EvaluationError(f"{prefix}: aliases must be an array of strings")
            if entity.get("role_in_event") not in ROLES:
                raise EvaluationError(f"{prefix}: invalid role_in_event")
            for field in ("a_event", "b_entity", "c_persistence"):
                if not _valid_score(entity.get(field)):
                    raise EvaluationError(f"{prefix}: invalid {field}")
            if type(entity.get("expected_pass")) is not bool:
                raise EvaluationError(f"{prefix}: expected_pass must be a boolean")
            if _decision(entity) != entity["expected_pass"]:
                raise EvaluationError(f"{prefix}: expected_pass contradicts A/C")
        goldens.append(golden)
    return goldens

def _indexed_pipeline(
    rows: Sequence[Mapping[str, Any]], case_ids: set[str]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise EvaluationError(f"pipeline row {row_number} must be an object")
        case_id = row.get("case_id")
        version = row.get("version")
        if not isinstance(case_id, str) or case_id not in case_ids:
            raise EvaluationError(f"pipeline row {row_number}: unknown case_id")
        if version not in VERSIONS:
            raise EvaluationError(f"pipeline row {row_number}: invalid version")
        if row.get("status") not in {"success", "error"}:
            raise EvaluationError(f"pipeline row {row_number}: invalid status")
        key = (case_id, version)
        if key in indexed:
            raise EvaluationError(f"duplicate pipeline key {key!r}")
        indexed[key] = row
    expected = {(case_id, version) for case_id in case_ids for version in VERSIONS}
    if set(indexed) != expected:
        raise EvaluationError(
            f"pipeline key coverage mismatch: missing={sorted(expected - set(indexed))!r}"
        )
    return indexed

def _actual_entities(
    row: Mapping[str, Any], case_id: str, version: str
) -> tuple[list[Mapping[str, Any]], dict[str, Any] | None]:
    reasons: list[str] = []
    if row["status"] == "error":
        reasons.append("status_error")
    output = row.get("actual_output")
    if not isinstance(output, Mapping):
        reasons.append("actual_output_not_object")
    else:
        entities = output.get("entities")
        if not isinstance(entities, list):
            reasons.append("entities_not_array")
        elif any(not isinstance(entity, Mapping) for entity in entities):
            reasons.append("entities_contain_non_objects")
    if reasons:
        return [], {
            "case_id": case_id,
            "version": version,
            "status": row["status"],
            "error": row.get("error"),
            "reasons": reasons,
        }
    return output["entities"], None

def _invalid_actual_issues(entity: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(entity.get("name"), str):
        issues.append({"field": "name", "actual": entity.get("name"), "reason": "invalid"})
    if entity.get("role_in_event") not in ROLES:
        issues.append(
            {"field": "role_in_event", "actual": entity.get("role_in_event"), "reason": "invalid"}
        )
    for field in ("a_event", "b_entity", "c_persistence"):
        if not _valid_score(entity.get(field)):
            issues.append({"field": field, "actual": entity.get(field), "reason": "invalid"})
    return issues

def _empty_counts() -> dict[str, int]:
    return {
        key: 0
        for key in (
            "E_TP", "E_FN", "E_FP", "S_TP", "S_FN", "S_FP",
            "role_correct", "a_correct", "c_correct", "decision_correct",
            "expected_entity_count", "actual_entity_count",
        )
    }

def _evaluate_version(
    goldens: Sequence[Mapping[str, Any]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    version: str,
) -> dict[str, Any]:
    counts = _empty_counts()
    matrix = {
        role: {column: 0 for column in (*ROLES, INVALID_ROLE)} for role in ROLES
    }
    technical: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {
        "missing": [], "extra": [], "incorrect": []
    }
    for golden in goldens:
        case_id = golden["id"]
        expected = golden["expected_output"]["entities"]
        actual, technical_error = _actual_entities(rows[(case_id, version)], case_id, version)
        if technical_error:
            technical.append(technical_error)
        counts["expected_entity_count"] += len(expected)
        counts["actual_entity_count"] += len(actual)
        pairs, missing, extra = match_entities(expected, actual)
        for expected_index, actual_index in pairs:
            wanted = expected[expected_index]
            found = actual[actual_index]
            counts["E_TP"] += 1
            expected_pass = wanted["expected_pass"]
            actual_pass = _decision(found)
            actual_role = found.get("role_in_event")
            matrix[wanted["role_in_event"]][
                actual_role if actual_role in ROLES else INVALID_ROLE
            ] += 1
            issues: list[dict[str, Any]] = []
            checks = (
                ("role_in_event", "role_correct", actual_role in ROLES),
                ("a_event", "a_correct", _valid_score(found.get("a_event"))),
                ("c_persistence", "c_correct", _valid_score(found.get("c_persistence"))),
            )
            for field, counter, valid in checks:
                actual_value = found.get(field)
                if valid and actual_value == wanted[field]:
                    counts[counter] += 1
                else:
                    issues.append(
                        {
                            "field": field,
                            "expected": wanted[field],
                            "actual": actual_value,
                            "reason": "mismatch" if valid else "invalid",
                        }
                    )
            b_value = found.get("b_entity")
            if not _valid_score(b_value) or b_value != wanted["b_entity"]:
                issues.append(
                    {
                        "field": "b_entity",
                        "expected": wanted["b_entity"],
                        "actual": b_value,
                        "reason": "mismatch_no_metric_effect" if _valid_score(b_value) else "invalid_no_metric_effect",
                    }
                )
            if actual_pass is not None and actual_pass == expected_pass:
                counts["decision_correct"] += 1
            else:
                issues.append(
                    {
                        "field": "decision",
                        "expected": expected_pass,
                        "actual": actual_pass,
                        "reason": "invalid_a_or_c" if actual_pass is None else "mismatch",
                    }
                )
            if expected_pass:
                if actual_pass is True:
                    counts["S_TP"] += 1
                else:
                    counts["S_FN"] += 1
            elif actual_pass is True:
                counts["S_FP"] += 1
            if issues:
                details["incorrect"].append(
                    {
                        "case_id": case_id,
                        "expected_index": expected_index,
                        "actual_index": actual_index,
                        "expected_name": wanted["name"],
                        "actual_name": found.get("name"),
                        "issues": issues,
                    }
                )
        for expected_index in missing:
            wanted = expected[expected_index]
            counts["E_FN"] += 1
            if wanted["expected_pass"]:
                counts["S_FN"] += 1
            else:
                counts["decision_correct"] += 1
            details["missing"].append(
                {
                    "case_id": case_id,
                    "expected_index": expected_index,
                    "expected_name": wanted["name"],
                    "expected_pass": wanted["expected_pass"],
                    "technical_error": technical_error is not None,
                }
            )
        for actual_index in extra:
            found = actual[actual_index]
            actual_pass = _decision(found)
            counts["E_FP"] += 1
            if actual_pass is True:
                counts["S_FP"] += 1
            details["extra"].append(
                {
                    "case_id": case_id,
                    "actual_index": actual_index,
                    "actual_name": found.get("name"),
                    "actual_pass": actual_pass,
                    "issues": _invalid_actual_issues(found),
                }
            )
    specs = {
        "extraction_precision": (counts["E_TP"], counts["E_TP"] + counts["E_FP"], 0.0, None),
        "extraction_recall": (counts["E_TP"], counts["E_TP"] + counts["E_FN"], 0.0, None),
        "extraction_f1": (2 * counts["E_TP"], 2 * counts["E_TP"] + counts["E_FP"] + counts["E_FN"], 0.0, None),
        "role_accuracy": (counts["role_correct"], counts["E_TP"], 1.0, PIPELINE_THRESHOLDS["role_accuracy"]),
        "a_accuracy": (counts["a_correct"], counts["E_TP"], 1.0, PIPELINE_THRESHOLDS["a_accuracy"]),
        "c_accuracy": (counts["c_correct"], counts["E_TP"], 1.0, PIPELINE_THRESHOLDS["c_accuracy"]),
        "decision_accuracy": (counts["decision_correct"], counts["expected_entity_count"], 1.0, PIPELINE_THRESHOLDS["decision_accuracy"]),
        "significance_precision": (counts["S_TP"], counts["S_TP"] + counts["S_FP"], 0.0, None),
        "significance_recall": (counts["S_TP"], counts["S_TP"] + counts["S_FN"], 0.0, None),
        "significance_f1": (2 * counts["S_TP"], 2 * counts["S_TP"] + counts["S_FP"] + counts["S_FN"], 0.0, None),
        "false_positive_decision_rate": (counts["S_FP"], counts["S_TP"] + counts["S_FP"], 0.0, None),
    }
    metrics = {name: _metric(*spec) for name, spec in specs.items()}
    strict = all(metrics[name]["passed"] for name in PIPELINE_THRESHOLDS)
    return {
        "counters": counts,
        "metrics": metrics,
        "strict_evaluation_passed": strict,
        "role_confusion_matrix": matrix,
        "technical_errors": {
            "count": len(technical),
            "case_ids": [item["case_id"] for item in technical],
            "details": technical,
        },
        "entity_details": details,
    }

def _compare(current: Mapping[str, Any], updated: Mapping[str, Any]) -> dict[str, Any]:
    counter_deltas = {
        key: updated["counters"][key] - value
        for key, value in current["counters"].items()
    }
    lower_better = {"false_positive_decision_rate"}
    metrics: dict[str, Any] = {}
    for name, current_metric in current["metrics"].items():
        current_value = current_metric["value"]
        updated_value = updated["metrics"][name]["value"]
        delta = updated_value - current_value
        if delta == 0:
            change = "unchanged"
        elif (delta < 0) == (name in lower_better):
            change = "improved"
        else:
            change = "worsened"
        metrics[name] = {
            "current": current_value,
            "updated": updated_value,
            "delta": delta,
            "change": change,
        }
    current_technical = set(current["technical_errors"]["case_ids"])
    updated_technical = set(updated["technical_errors"]["case_ids"])
    introduced = sorted(updated_technical - current_technical)
    technical = {
        "current_count": len(current_technical),
        "updated_count": len(updated_technical),
        "delta_count": len(updated_technical) - len(current_technical),
        "resolved_case_ids": sorted(current_technical - updated_technical),
        "introduced_case_ids": introduced,
        "shared_case_ids": sorted(current_technical & updated_technical),
    }
    failed = [
        name for name in PIPELINE_THRESHOLDS if not updated["metrics"][name]["passed"]
    ]
    can_replace = updated["strict_evaluation_passed"] and not introduced
    reason_codes = []
    if failed:
        reason_codes.append("updated_failed_fixed_thresholds")
    if introduced:
        reason_codes.append("updated_introduced_technical_errors")
    if can_replace:
        reason_codes.append("updated_passed_all_fixed_thresholds")
    return {
        "counter_deltas": counter_deltas,
        "metrics": metrics,
        "technical_errors": technical,
        "can_replace": can_replace,
        "reason_codes": reason_codes,
        "rule": (
            "updated passes all four fixed thresholds and introduces no new "
            "technical errors relative to current"
        ),
    }

def evaluate_pipelines(
    goldens: Sequence[Mapping[str, Any]],
    pipeline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate both saved pipeline versions using micro-aggregated counters."""
    validated_goldens = _validated_goldens(goldens)
    rows = _indexed_pipeline(pipeline_rows, {golden["id"] for golden in validated_goldens})
    versions = {
        version: _evaluate_version(validated_goldens, rows, version)
        for version in VERSIONS
    }
    return {
        "versions": versions,
        "comparison": _compare(versions["current"], versions["updated"]),
    }

_JUDGE_FIELDS = {
    "case_id", "version", "criterion", "judge_score", "judge_reason",
    "reference_pass", "judge_model_version", "judge_prompt_version", "rubric_version",
}
_OUTCOMES = (
    "correct_acceptances", "correct_rejections",
    "false_acceptances", "false_rejections",
)

def _validated_judge(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or set(row) != _JUDGE_FIELDS:
            raise EvaluationError(f"judge row {row_number}: invalid field set")
        for field in (
            "case_id", "version", "criterion", "judge_reason",
            "judge_model_version", "judge_prompt_version", "rubric_version",
        ):
            if not isinstance(row[field], str) or not row[field].strip():
                raise EvaluationError(f"judge row {row_number}: invalid {field}")
        if row["version"] not in VERSIONS:
            raise EvaluationError(f"judge row {row_number}: unsupported version")
        if row["criterion"] not in JUDGE_THRESHOLDS:
            raise EvaluationError(f"judge row {row_number}: unsupported criterion")
        score = row["judge_score"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= score <= 1
        ):
            raise EvaluationError(f"judge row {row_number}: invalid judge_score")
        if type(row["reference_pass"]) is not bool:
            raise EvaluationError(f"judge row {row_number}: reference_pass must be boolean")
        key = (row["case_id"], row["version"], row["criterion"])
        if key in seen:
            raise EvaluationError(f"duplicate judge key {key!r}")
        seen.add(key)
        validated.append({**row, "judge_score": float(score)})
    rank = {version: index for index, version in enumerate(VERSIONS)}
    criterion_rank = {name: index for index, name in enumerate(JUDGE_THRESHOLDS)}
    validated.sort(
        key=lambda row: (rank[row["version"]], criterion_rank[row["criterion"]], row["case_id"])
    )
    return validated

def _judge_summary(counts: Mapping[str, int], total: int) -> dict[str, Any]:
    agreed = counts["correct_acceptances"] + counts["correct_rejections"]
    return {
        "total": total,
        **{outcome: counts[outcome] for outcome in _OUTCOMES},
        "agreement": {
            "numerator": agreed,
            "denominator": total,
            "value": agreed / total if total else 0.0,
        },
    }

def evaluate_judge(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare thresholded judge scores with human ``reference_pass`` labels."""
    groups = {
        version: {
            criterion: {
                "threshold": threshold,
                "counts": {outcome: 0 for outcome in _OUTCOMES},
                "total": 0,
                "disagreements": [],
            }
            for criterion, threshold in JUDGE_THRESHOLDS.items()
        }
        for version in VERSIONS
    }
    overall_counts = {outcome: 0 for outcome in _OUTCOMES}
    total = 0
    for row in _validated_judge(rows):
        threshold = JUDGE_THRESHOLDS[row["criterion"]]
        judge_pass = row["judge_score"] >= threshold
        reference_pass = row["reference_pass"]
        if judge_pass and reference_pass:
            outcome = "correct_acceptances"
        elif not judge_pass and not reference_pass:
            outcome = "correct_rejections"
        elif judge_pass:
            outcome = "false_acceptances"
        else:
            outcome = "false_rejections"
        group = groups[row["version"]][row["criterion"]]
        group["counts"][outcome] += 1
        group["total"] += 1
        overall_counts[outcome] += 1
        total += 1
        if judge_pass != reference_pass:
            group["disagreements"].append(
                {
                    "case_id": row["case_id"],
                    "version": row["version"],
                    "criterion": row["criterion"],
                    "judge_score": row["judge_score"],
                    "judge_pass": judge_pass,
                    "reference_pass": reference_pass,
                    "outcome": outcome,
                }
            )
    by_version: dict[str, Any] = {}
    for version in VERSIONS:
        by_version[version] = {}
        for criterion in JUDGE_THRESHOLDS:
            group = groups[version][criterion]
            by_version[version][criterion] = {
                "threshold": group["threshold"],
                **_judge_summary(group["counts"], group["total"]),
                "disagreements": group["disagreements"],
            }
    return {
        "by_version": by_version,
        "overall": _judge_summary(overall_counts, total),
    }

__all__ = ["EvaluationError", "evaluate_judge", "evaluate_pipelines"]
