"""End-to-end control totals for the provided evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from news_eval.evaluation import evaluate_judge, evaluate_pipelines
from news_eval.readers import load_data
from news_eval.reporting import build_report, render_conclusion


ROOT = Path(__file__).resolve().parents[1]


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.integrity = load_data(ROOT / "data")
        cls.pipeline = evaluate_pipelines(
            cls.data["goldens"], cls.data["pipeline_outputs"]
        )
        cls.judge = evaluate_judge(cls.data["judge_outputs"])
        cls.report = build_report(cls.data, cls.integrity)

    def test_pipeline_control_totals_matrices_and_technical_errors(self) -> None:
        expected = {
            "current": {
                "counters": {
                    "E_TP": 25,
                    "E_FN": 13,
                    "E_FP": 22,
                    "S_TP": 12,
                    "S_FN": 7,
                    "S_FP": 4,
                    "role_correct": 22,
                    "a_correct": 22,
                    "c_correct": 22,
                    "decision_correct": 31,
                    "expected_entity_count": 38,
                    "actual_entity_count": 47,
                },
                "metric_fractions": {
                    "extraction_precision": (25, 47),
                    "extraction_recall": (25, 38),
                    "extraction_f1": (50, 85),
                    "role_accuracy": (22, 25),
                    "a_accuracy": (22, 25),
                    "c_accuracy": (22, 25),
                    "decision_accuracy": (31, 38),
                    "significance_precision": (12, 16),
                    "significance_recall": (12, 19),
                    "significance_f1": (24, 35),
                    "false_positive_decision_rate": (4, 16),
                },
                "technical_ids": ["news-015"],
            },
            "updated": {
                "counters": {
                    "E_TP": 22,
                    "E_FN": 16,
                    "E_FP": 24,
                    "S_TP": 11,
                    "S_FN": 8,
                    "S_FP": 3,
                    "role_correct": 20,
                    "a_correct": 20,
                    "c_correct": 18,
                    "decision_correct": 30,
                    "expected_entity_count": 38,
                    "actual_entity_count": 46,
                },
                "metric_fractions": {
                    "extraction_precision": (22, 46),
                    "extraction_recall": (22, 38),
                    "extraction_f1": (44, 84),
                    "role_accuracy": (20, 22),
                    "a_accuracy": (20, 22),
                    "c_accuracy": (18, 22),
                    "decision_accuracy": (30, 38),
                    "significance_precision": (11, 14),
                    "significance_recall": (11, 19),
                    "significance_f1": (22, 33),
                    "false_positive_decision_rate": (3, 14),
                },
                "technical_ids": ["news-018"],
            },
        }

        self.assertTrue(self.integrity["passed"])
        self.assertEqual(self.report["pipeline_evaluation"], self.pipeline)
        for version, control in expected.items():
            result = self.pipeline["versions"][version]
            self.assertEqual(result["counters"], control["counters"])
            self.assertEqual(
                result["technical_errors"]["case_ids"], control["technical_ids"]
            )
            matrix_total = sum(
                count
                for row in result["role_confusion_matrix"].values()
                for count in row.values()
            )
            self.assertEqual(matrix_total, result["counters"]["E_TP"])
            for metric_name, (numerator, denominator) in control[
                "metric_fractions"
            ].items():
                metric = result["metrics"][metric_name]
                self.assertEqual(
                    (metric["numerator"], metric["denominator"]),
                    (numerator, denominator),
                )
                self.assertAlmostEqual(metric["value"], numerator / denominator)

        comparison = self.pipeline["comparison"]
        self.assertFalse(comparison["can_replace"])
        self.assertEqual(
            comparison["technical_errors"]["resolved_case_ids"], ["news-015"]
        )
        self.assertEqual(
            comparison["technical_errors"]["introduced_case_ids"], ["news-018"]
        )

    def test_judge_control_totals_and_overall(self) -> None:
        controls = {
            ("current", "entity_extraction_completeness"): (19, 2, 2, 1, 21),
            ("current", "significance_score_adequacy"): (16, 6, 1, 1, 22),
            ("updated", "entity_extraction_completeness"): (18, 3, 2, 1, 21),
            ("updated", "significance_score_adequacy"): (11, 7, 2, 4, 18),
        }
        self.assertEqual(self.report["judge_evaluation"], self.judge)
        for (version, criterion), expected in controls.items():
            group = self.judge["by_version"][version][criterion]
            actual = (
                group["correct_acceptances"],
                group["correct_rejections"],
                group["false_acceptances"],
                group["false_rejections"],
                group["agreement"]["numerator"],
            )
            self.assertEqual(actual, expected)
            self.assertEqual(group["agreement"]["denominator"], 24)
            self.assertEqual(
                len(group["disagreements"]), expected[2] + expected[3]
            )

        overall = self.judge["overall"]
        self.assertEqual(
            (
                overall["correct_acceptances"],
                overall["correct_rejections"],
                overall["false_acceptances"],
                overall["false_rejections"],
            ),
            (64, 18, 7, 7),
        )
        self.assertEqual(
            overall["agreement"],
            {"numerator": 82, "denominator": 96, "value": 82 / 96},
        )

    def test_report_provenance_and_decision_first_conclusion(self) -> None:
        self.assertEqual(self.report["report_schema_version"], "1.1")
        self.assertEqual(self.report["dataset"]["slices"], [])
        self.assertEqual(
            self.report["provenance"],
            {
                "dataset_versions": ["1.0"],
                "judge_model_versions": ["judge-1"],
                "judge_prompt_versions": ["prompt-1"],
                "rubric_versions": ["rubric-1"],
            },
        )

        conclusion = render_conclusion(self.report)
        self.assertTrue(conclusion.startswith("# Итог оценки\n\n## Вердикт"))
        for statement in (
            "**Не заменять `current` на `updated` в текущем виде.**",
            "дефицит правильных решений: **2**",
            "**82/96 = 85.42%**",
            "**18/24 = 75.00%**",
            "`metadata.slices` не заполнены",
        ):
            self.assertIn(statement, conclusion)

    def test_committed_artifacts_are_byte_for_byte_reproducible(self) -> None:
        expected_report = json.dumps(self.report, ensure_ascii=False, indent=2) + "\n"
        expected_conclusion = render_conclusion(self.report)
        self.assertEqual(
            (ROOT / "report.json").read_text(encoding="utf-8"), expected_report
        )
        self.assertEqual(
            (ROOT / "conclusion.md").read_text(encoding="utf-8"),
            expected_conclusion,
        )


if __name__ == "__main__":
    unittest.main()
