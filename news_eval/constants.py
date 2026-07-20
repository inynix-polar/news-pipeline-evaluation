"""Domain constants defined by the assignment."""

from __future__ import annotations

VERSIONS: tuple[str, ...] = ("current", "updated")

ROLES: tuple[str, ...] = (
    "constituting",
    "status_change",
    "achievement",
    "agent",
    "umbrella_routine",
    "context",
)
INVALID_ROLE = "<invalid>"

VALID_SCORES: frozenset[float] = frozenset({0.2, 0.6, 1.0})

PIPELINE_THRESHOLDS: dict[str, float] = {
    "role_accuracy": 0.85,
    "a_accuracy": 0.84,
    "c_accuracy": 0.78,
    "decision_accuracy": 0.82,
}

JUDGE_THRESHOLDS: dict[str, float] = {
    "entity_extraction_completeness": 0.78,
    "significance_score_adequacy": 0.82,
}

EXPECTED_ROW_COUNTS: dict[str, int] = {
    "goldens": 24,
    "pipeline_outputs": 48,
    "judge_outputs": 96,
}
