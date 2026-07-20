"""Create the machine-readable report and the short human conclusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from news_eval.constants import JUDGE_THRESHOLDS, PIPELINE_THRESHOLDS, ROLES, VALID_SCORES
from news_eval.evaluation import evaluate_judge, evaluate_pipelines


METRIC_NAMES = {
    "extraction_precision": "Точность извлечения",
    "extraction_recall": "Полнота извлечения",
    "extraction_f1": "F1 извлечения",
    "role_accuracy": "Правильность роли",
    "a_accuracy": "Правильность A",
    "c_accuracy": "Правильность C",
    "decision_accuracy": "Правильность решения",
    "significance_precision": "Точность значимости",
    "significance_recall": "Полнота значимости",
    "significance_f1": "F1 значимости",
    "false_positive_decision_rate": "Доля ложноположительных решений",
}

CRITERION_NAMES = {
    "entity_extraction_completeness": "полнота извлечения",
    "significance_score_adequacy": "адекватность значимости",
}


def build_report(data: dict[str, list[dict]], integrity: dict) -> dict[str, Any]:
    """Calculate every required result from the three loaded datasets."""

    goldens = data["goldens"]
    pipeline_evaluation = evaluate_pipelines(goldens, data["pipeline_outputs"])
    judge_evaluation = evaluate_judge(data["judge_outputs"])
    expected = [
        entity
        for golden in goldens
        for entity in golden["expected_output"]["entities"]
    ]
    return {
        "report_schema_version": "1.0",
        "integrity": integrity,
        "dataset": {
            "case_count": len(goldens),
            "expected_entity_count": len(expected),
            "expected_positive_count": sum(e["expected_pass"] is True for e in expected),
            "expected_negative_count": sum(e["expected_pass"] is False for e in expected),
            "empty_expected_case_ids": [
                g["id"] for g in goldens if not g["expected_output"]["entities"]
            ],
        },
        "configuration": {
            "aggregation": "micro counters over all cases in each version",
            "matching": "greedy one-to-one; golden name and explicit aliases only",
            "entity_decision_rule": "(a_event + c_persistence) / 2 >= 0.8",
            "roles": list(ROLES),
            "valid_abc_values": sorted(VALID_SCORES),
            "pipeline_thresholds": PIPELINE_THRESHOLDS,
            "judge_thresholds": JUDGE_THRESHOLDS,
        },
        "pipeline_evaluation": pipeline_evaluation,
        "judge_evaluation": judge_evaluation,
    }


def _fraction(metric: dict) -> str:
    return f"{metric['numerator']}/{metric['denominator']} = {metric['value']:.6f}"


def render_conclusion(report: dict[str, Any]) -> str:
    """Render conclusion.md only from calculated values in report.json."""

    pipeline = report["pipeline_evaluation"]
    versions = pipeline["versions"]
    current, updated = versions["current"], versions["updated"]
    comparison = pipeline["comparison"]
    judge = report["judge_evaluation"]

    lines = [
        "# Заключение",
        "",
        "## Обязательные пороги",
        "",
        "| Показатель | Порог | current | updated |",
        "|---|---:|---:|---:|",
    ]
    for name, threshold in PIPELINE_THRESHOLDS.items():
        left, right = current["metrics"][name], updated["metrics"][name]
        lines.append(
            f"| {METRIC_NAMES[name]} | ≥ {threshold:.2f} | {_fraction(left)} — "
            f"{'пройден' if left['passed'] else 'не пройден'} | {_fraction(right)} — "
            f"{'пройден' if right['passed'] else 'не пройден'} |"
        )

    lines += [""]
    for version, result in versions.items():
        status = "пройдена" if result["strict_evaluation_passed"] else "не пройдена"
        failed = [
            METRIC_NAMES[name]
            for name in PIPELINE_THRESHOLDS
            if not result["metrics"][name]["passed"]
        ]
        suffix = f"; не пройдены: {', '.join(failed)}" if failed else ""
        lines.append(f"Строгая оценка `{version}`: **{status}**{suffix}.")

    current_decision = current["metrics"]["decision_accuracy"]
    if not current_decision["passed"] and round(current_decision["value"], 2) >= current_decision["threshold"]:
        lines += [
            "",
            "`current` не проходит порог после сравнения по неокруглённому значению "
            f"`{current_decision['value']:.12f}`; отображаемое `0.82` не означает прохождение.",
        ]

    changes = comparison["metrics"]
    improved = [name for name, item in changes.items() if item["change"] == "improved"]
    worsened = [name for name, item in changes.items() if item["change"] == "worsened"]
    lines += ["", "## Изменения updated относительно current", "", "Улучшились:", ""]
    lines += [
        f"- {METRIC_NAMES[name]}: `{changes[name]['current']:.6f}` → "
        f"`{changes[name]['updated']:.6f}` ({changes[name]['delta'] * 100:+.2f} п.п.)."
        for name in improved
    ] or ["- Нет."]
    lines += ["", "Ухудшились:", ""]
    lines += [
        f"- {METRIC_NAMES[name]}: `{changes[name]['current']:.6f}` → "
        f"`{changes[name]['updated']:.6f}` ({changes[name]['delta'] * 100:+.2f} п.п.)."
        for name in worsened
    ] or ["- Нет."]

    if (
        current["counters"]["E_TP"] != updated["counters"]["E_TP"]
        and {"role_accuracy", "a_accuracy"} & set(improved)
    ):
        lines += [
            "",
            "Доли роли или A выросли при другом числе "
            "сопоставленных сущностей: "
            f"`E_TP` изменился с `{current['counters']['E_TP']}` до "
            f"`{updated['counters']['E_TP']}`.",
        ]
    lines += ["", "## Технические ошибки", ""]
    for version, result in versions.items():
        details = result["technical_errors"]["details"]
        if not details:
            lines.append(f"- `{version}`: нет.")
        for detail in details:
            error = f"; {detail['error']}" if detail.get("error") else ""
            lines.append(
                f"- `{version}/{detail['case_id']}`: "
                f"{', '.join(detail['reasons'])}{error}."
            )
    technical = comparison["technical_errors"]
    if technical["resolved_case_ids"]:
        lines.append(
            "- Исправлены: "
            + ", ".join(f"`{case}`" for case in technical["resolved_case_ids"])
            + "."
        )
    if technical["introduced_case_ids"]:
        lines.append(
            "- Появились: "
            + ", ".join(f"`{case}`" for case in technical["introduced_case_ids"])
            + "."
        )

    lines += [
        "",
        "## Проверка модели-судьи",
        "",
        "| Версия | Критерий | Верные принятия | Верные отклонения | Ошибочные принятия | Ошибочные отклонения | Совпадение |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for version in ("current", "updated"):
        for criterion in JUDGE_THRESHOLDS:
            group = judge["by_version"][version][criterion]
            agreement = group["agreement"]
            lines.append(
                f"| {version} | {CRITERION_NAMES[criterion]} | "
                f"{group['correct_acceptances']} | {group['correct_rejections']} | "
                f"{group['false_acceptances']} | {group['false_rejections']} | "
                f"{agreement['numerator']}/{agreement['denominator']} = "
                f"{agreement['value']:.2%} |"
            )
    lines += [
        "",
        "Порог качества судьи в задании не определён; agreement описывается, но "
        "не заменяет строгую оценку pipeline по golden-разметке.",
        "",
        "## Рекомендация",
        "",
    ]
    if comparison["can_replace"]:
        lines.append(
            "**Да, updated может заменить current:** обязательные пороги пройдены "
            "и новых технических ошибок нет."
        )
    else:
        failed = [
            METRIC_NAMES[name]
            for name in PIPELINE_THRESHOLDS
            if not updated["metrics"][name]["passed"]
        ]
        reasons = []
        if failed:
            reasons.append("не пройдены пороги: " + ", ".join(failed))
        if worsened:
            reasons.append("ухудшились: " + ", ".join(METRIC_NAMES[n] for n in worsened))
        if technical["introduced_case_ids"]:
            count = len(technical["introduced_case_ids"])
            reasons.append(f"появились новые технические ошибки ({count})")
        text = "; ".join(reasons)
        lines.append(
            "**Нет, updated не может заменить current.** "
            + text[:1].upper()
            + text[1:]
            + "."
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], report_path: Path, conclusion_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    conclusion_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    conclusion_path.write_text(render_conclusion(report), encoding="utf-8")
