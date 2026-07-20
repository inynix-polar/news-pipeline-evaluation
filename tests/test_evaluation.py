from __future__ import annotations

import json
from pathlib import Path
import unittest

from news_eval.evaluation import EvaluationError, evaluate_judge, evaluate_pipelines


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines()
    ]


def entity(
    name: str,
    role: str,
    a: object,
    b: object,
    c: object,
    expected_pass: bool | None = None,
) -> dict:
    result = {
        "name": name,
        "role_in_event": role,
        "a_event": a,
        "b_entity": b,
        "c_persistence": c,
    }
    if expected_pass is not None:
        result.update({"aliases": [], "expected_pass": expected_pass})
    return result


def golden(case_id: str, entities: list[dict]) -> dict:
    return {"id": case_id, "expected_output": {"entities": entities}}


def row(
    case_id: str,
    version: str,
    entities: object,
    status: str = "success",
) -> dict:
    return {
        "case_id": case_id,
        "version": version,
        "status": status,
        "actual_output": None if entities is None else {"entities": entities},
        "error": "boom" if status == "error" else None,
    }


def judge_row(
    case_id: str,
    version: str,
    criterion: str,
    score: object,
    reference: object,
) -> dict:
    return {
        "case_id": case_id,
        "version": version,
        "criterion": criterion,
        "judge_score": score,
        "judge_reason": "reason",
        "reference_pass": reference,
        "judge_model_version": "judge-1",
        "judge_prompt_version": "prompt-1",
        "rubric_version": "rubric-1",
    }


class ProvidedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = evaluate_pipelines(
            load("goldens.jsonl"), load("pipeline_outputs.jsonl")
        )
        cls.judge = evaluate_judge(load("judge_outputs.jsonl"))

    def test_exact_pipeline_results(self) -> None:
        expected = {
            "current": {
                "extraction": (25, 13, 22),
                "significance": (12, 7, 4),
                "correct": (22, 22, 22, 31),
                "technical": ["news-015"],
                "details": (13, 22),
            },
            "updated": {
                "extraction": (22, 16, 24),
                "significance": (11, 8, 3),
                "correct": (20, 20, 18, 30),
                "technical": ["news-018"],
                "details": (16, 24),
            },
        }
        for version, wanted in expected.items():
            with self.subTest(version=version):
                result = self.pipeline["versions"][version]
                counts = result["counters"]
                self.assertEqual(
                    (counts["E_TP"], counts["E_FN"], counts["E_FP"]),
                    wanted["extraction"],
                )
                self.assertEqual(
                    (counts["S_TP"], counts["S_FN"], counts["S_FP"]),
                    wanted["significance"],
                )
                self.assertEqual(
                    (
                        counts["role_correct"], counts["a_correct"],
                        counts["c_correct"], counts["decision_correct"],
                    ),
                    wanted["correct"],
                )
                self.assertEqual(result["technical_errors"]["case_ids"], wanted["technical"])
                self.assertEqual(
                    (
                        len(result["entity_details"]["missing"]),
                        len(result["entity_details"]["extra"]),
                    ),
                    wanted["details"],
                )
                self.assertEqual(
                    sum(sum(role.values()) for role in result["role_confusion_matrix"].values()),
                    counts["E_TP"],
                )
                self.assertFalse(result["strict_evaluation_passed"])

        current = self.pipeline["versions"]["current"]["metrics"]
        updated = self.pipeline["versions"]["updated"]["metrics"]
        self.assertEqual(current["extraction_f1"]["numerator"], 50)
        self.assertEqual(current["extraction_f1"]["denominator"], 85)
        self.assertAlmostEqual(current["decision_accuracy"]["value"], 31 / 38)
        self.assertFalse(current["decision_accuracy"]["passed"])
        self.assertAlmostEqual(updated["significance_f1"]["value"], 22 / 33)

    def test_exact_comparison(self) -> None:
        comparison = self.pipeline["comparison"]
        self.assertEqual(
            (comparison["counter_deltas"]["E_TP"], comparison["counter_deltas"]["E_FN"], comparison["counter_deltas"]["E_FP"]),
            (-3, 3, 2),
        )
        self.assertEqual(comparison["metrics"]["extraction_f1"]["change"], "worsened")
        self.assertEqual(comparison["metrics"]["role_accuracy"]["change"], "improved")
        self.assertEqual(
            comparison["technical_errors"]["resolved_case_ids"], ["news-015"]
        )
        self.assertEqual(
            comparison["technical_errors"]["introduced_case_ids"], ["news-018"]
        )
        self.assertFalse(comparison["can_replace"])
        self.assertEqual(
            comparison["reason_codes"],
            ["updated_failed_fixed_thresholds", "updated_introduced_technical_errors"],
        )

    def test_exact_judge_results(self) -> None:
        expected = {
            ("current", "entity_extraction_completeness"): (19, 2, 2, 1, 21),
            ("current", "significance_score_adequacy"): (16, 6, 1, 1, 22),
            ("updated", "entity_extraction_completeness"): (18, 3, 2, 1, 21),
            ("updated", "significance_score_adequacy"): (11, 7, 2, 4, 18),
        }
        for (version, criterion), wanted in expected.items():
            group = self.judge["by_version"][version][criterion]
            self.assertEqual(
                tuple(group[name] for name in (
                    "correct_acceptances", "correct_rejections",
                    "false_acceptances", "false_rejections",
                )),
                wanted[:4],
            )
            self.assertEqual(group["agreement"]["numerator"], wanted[4])
            self.assertEqual(group["agreement"]["denominator"], 24)
            self.assertEqual(len(group["disagreements"]), wanted[2] + wanted[3])
        self.assertEqual(self.judge["overall"]["agreement"]["numerator"], 82)
        self.assertEqual(self.judge["overall"]["agreement"]["denominator"], 96)


class EdgeRuleTests(unittest.TestCase):
    def test_technical_errors_missing_decisions_and_zero_denominators(self) -> None:
        goldens = [
            golden("positive", [entity("P", "agent", 1.0, 1.0, 1.0, True)]),
            golden("negative", [entity("N", "context", 0.2, 0.2, 0.2, False)]),
            golden("empty", []),
        ]
        rows = [
            row("positive", "current", None, "error"),
            row("negative", "current", []),
            row("empty", "current", []),
            row("positive", "updated", "malformed"),
            row("negative", "updated", []),
            row("empty", "updated", []),
        ]
        result = evaluate_pipelines(goldens, rows)
        for version in ("current", "updated"):
            evaluated = result["versions"][version]
            self.assertEqual(evaluated["technical_errors"]["count"], 1)
            self.assertEqual(evaluated["counters"]["E_FN"], 2)
            self.assertEqual(evaluated["counters"]["S_FN"], 1)
            self.assertEqual(evaluated["counters"]["decision_correct"], 1)
            for metric in ("role_accuracy", "a_accuracy", "c_accuracy"):
                self.assertEqual(evaluated["metrics"][metric]["value"], 1.0)

    def test_invalid_fields_have_separate_metric_effects(self) -> None:
        goldens = [golden("case", [
            entity("P", "agent", 1.0, 1.0, 1.0, True),
            entity("N", "context", 0.2, 0.2, 0.2, False),
        ])]
        actual = [
            entity("P", "bad-role", 1.0, 0.7, 0.4),
            entity("N", "context", 0.2, 0.7, 0.2),
            entity("extra+", "agent", 1.0, 1.0, 1.0),
            entity("extra-", "context", 0.2, 0.2, 0.2),
        ]
        rows = [row("case", version, actual) for version in ("current", "updated")]
        result = evaluate_pipelines(goldens, rows)["versions"]["current"]
        counts = result["counters"]
        self.assertEqual((counts["E_TP"], counts["E_FP"]), (2, 2))
        self.assertEqual((counts["S_TP"], counts["S_FN"], counts["S_FP"]), (0, 1, 1))
        self.assertEqual((counts["role_correct"], counts["a_correct"], counts["c_correct"]), (1, 2, 1))
        self.assertEqual(counts["decision_correct"], 1)
        self.assertEqual(result["role_confusion_matrix"]["agent"]["<invalid>"], 1)
        self.assertEqual(len(result["entity_details"]["incorrect"]), 2)

    def test_huge_numbers_are_field_errors_not_runtime_errors(self) -> None:
        huge = 10**400
        goldens = [
            golden("case", [entity("P", "agent", 1.0, 1.0, 1.0, True)])
        ]
        rows = [
            row("case", version, [entity("P", "agent", huge, 1.0, 1.0)])
            for version in ("current", "updated")
        ]

        result = evaluate_pipelines(goldens, rows)["versions"]["current"]

        self.assertEqual(result["counters"]["a_correct"], 0)
        self.assertEqual(result["counters"]["decision_correct"], 0)
        self.assertEqual(result["counters"]["S_FN"], 1)
        with self.assertRaisesRegex(EvaluationError, "invalid judge_score"):
            evaluate_judge([
                judge_row(
                    "case",
                    "current",
                    "entity_extraction_completeness",
                    huge,
                    True,
                )
            ])

    def test_strict_golden_validation(self) -> None:
        invalid = entity("N", "context", 0.2, 0.2, 0.2, False)
        invalid["expected_pass"] = "false"
        with self.assertRaisesRegex(EvaluationError, "expected_pass must be a boolean"):
            evaluate_pipelines(
                [golden("case", [invalid])],
                [row("case", version, []) for version in ("current", "updated")],
            )

    def test_new_technical_error_blocks_replacement(self) -> None:
        result = evaluate_pipelines(
            [golden("empty", [])],
            [row("empty", "current", []), row("empty", "updated", None, "error")],
        )
        self.assertTrue(result["versions"]["updated"]["strict_evaluation_passed"])
        self.assertFalse(result["comparison"]["can_replace"])
        self.assertEqual(
            result["comparison"]["reason_codes"],
            ["updated_introduced_technical_errors"],
        )

    def test_judge_boundaries_determinism_and_validation(self) -> None:
        records = [
            judge_row("a", "current", "entity_extraction_completeness", 0.78, True),
            judge_row("b", "current", "entity_extraction_completeness", 0.2, False),
            judge_row("a", "updated", "significance_score_adequacy", 0.82, False),
            judge_row("b", "updated", "significance_score_adequacy", 0.81, True),
        ]
        result = evaluate_judge(records)
        self.assertEqual(result, evaluate_judge(reversed(records)))
        current = result["by_version"]["current"]["entity_extraction_completeness"]
        updated = result["by_version"]["updated"]["significance_score_adequacy"]
        self.assertEqual((current["correct_acceptances"], current["correct_rejections"]), (1, 1))
        self.assertEqual((updated["false_acceptances"], updated["false_rejections"]), (1, 1))

        invalid = dict(records[0], reference_pass=1)
        with self.assertRaisesRegex(EvaluationError, "reference_pass must be boolean"):
            evaluate_judge([invalid])
        with self.assertRaisesRegex(EvaluationError, "duplicate judge key"):
            evaluate_judge([records[0], dict(records[0])])


if __name__ == "__main__":
    unittest.main()
