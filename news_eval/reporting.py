"""Create the machine-readable report and the human-readable conclusion."""

from __future__ import annotations

from collections import Counter
import json
from math import ceil
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

TECHNICAL_REASON_NAMES = {
    "status_error": "status=error",
    "actual_output_not_object": "actual_output не является объектом",
    "entities_not_array": "entities не является массивом",
    "entities_contain_non_objects": "entities содержит не-объекты",
}


def build_report(data: dict[str, list[dict]], integrity: dict) -> dict[str, Any]:
    """Calculate every required result from the three loaded datasets."""

    goldens = data["goldens"]
    judge_rows = data["judge_outputs"]
    expected = [
        entity
        for golden in goldens
        for entity in golden["expected_output"]["entities"]
    ]
    return {
        "report_schema_version": "1.1",
        "integrity": integrity,
        "dataset": {
            "case_count": len(goldens),
            "expected_entity_count": len(expected),
            "expected_positive_count": sum(e["expected_pass"] is True for e in expected),
            "expected_negative_count": sum(e["expected_pass"] is False for e in expected),
            "empty_expected_case_ids": [
                g["id"] for g in goldens if not g["expected_output"]["entities"]
            ],
            "slices": sorted(
                {
                    slice_name
                    for golden in goldens
                    for slice_name in golden["metadata"]["slices"]
                }
            ),
        },
        "provenance": {
            "dataset_versions": sorted(
                {golden["metadata"]["dataset_version"] for golden in goldens}
            ),
            "judge_model_versions": sorted(
                {row["judge_model_version"] for row in judge_rows}
            ),
            "judge_prompt_versions": sorted(
                {row["judge_prompt_version"] for row in judge_rows}
            ),
            "rubric_versions": sorted({row["rubric_version"] for row in judge_rows}),
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
        "pipeline_evaluation": evaluate_pipelines(
            goldens, data["pipeline_outputs"]
        ),
        "judge_evaluation": evaluate_judge(judge_rows),
    }


def _fraction(metric: dict) -> str:
    return f"{metric['numerator']}/{metric['denominator']} = {metric['value']:.2%}"


def _threshold_cell(metric: dict) -> str:
    return f"{_fraction(metric)} {'✅' if metric['passed'] else '❌'}"


def _technical_reason(reason: str) -> str:
    return TECHNICAL_REASON_NAMES.get(reason, reason)


def _verdict_lines(report: dict[str, Any]) -> list[str]:
    pipeline = report["pipeline_evaluation"]
    current = pipeline["versions"]["current"]
    updated = pipeline["versions"]["updated"]
    comparison = pipeline["comparison"]
    updated_decision = updated["metrics"]["decision_accuracy"]
    required = ceil(updated_decision["threshold"] * updated_decision["denominator"])
    shortfall = max(0, required - updated_decision["numerator"])
    introduced = comparison["technical_errors"]["introduced_case_ids"]

    lines = ["## Вердикт", ""]
    if comparison["can_replace"]:
        return lines + [
            "**`updated` может заменить `current`.**",
            "",
            "- Все четыре обязательных порога пройдены.",
            "- Новых технических ошибок относительно `current` нет.",
        ]

    lines += [
        "**Не заменять `current` на `updated` в текущем виде.**",
        "",
        "Почему:",
        "",
    ]
    if not updated_decision["passed"]:
        lines.append(
            f"- `updated` не проходит порог правильности решений: "
            f"**{_fraction(updated_decision)}** при требовании "
            f"**≥ {updated_decision['threshold']:.0%}**; дефицит правильных решений: "
            f"**{shortfall}**."
        )
    other_failed = [
        name
        for name in PIPELINE_THRESHOLDS
        if name != "decision_accuracy" and not updated["metrics"][name]["passed"]
    ]
    if other_failed:
        lines.append(
            "- Также не пройдены: "
            + ", ".join(METRIC_NAMES[name] for name in other_failed)
            + "."
        )
    extraction_f1 = comparison["metrics"]["extraction_f1"]
    if extraction_f1["change"] == "worsened":
        lines.append(
            f"- F1 извлечения снизился: **{extraction_f1['current']:.2%} → "
            f"{extraction_f1['updated']:.2%}** ({extraction_f1['delta'] * 100:+.2f} п.п.)."
        )
    if introduced:
        lines.append(
            "- Появилась новая техническая ошибка: "
            + ", ".join(f"`{case}`" for case in introduced)
            + "."
        )
    if not current["strict_evaluation_passed"]:
        current_decision = current["metrics"]["decision_accuracy"]
        lines.append(
            f"- `current` тоже не проходит все пороги "
            f"({_fraction(current_decision)}). Отказ от `updated` не означает, "
            "что `current` достаточно хорош."
        )
    return lines


def _gate_lines(report: dict[str, Any]) -> list[str]:
    versions = report["pipeline_evaluation"]["versions"]
    current, updated = versions["current"], versions["updated"]
    comparison = report["pipeline_evaluation"]["comparison"]
    introduced = comparison["technical_errors"]["introduced_case_ids"]
    current_decision = current["metrics"]["decision_accuracy"]
    updated_decision = updated["metrics"]["decision_accuracy"]
    threshold = updated_decision["threshold"]
    required = ceil(threshold * updated_decision["denominator"])

    lines = [
        "## Условия замены",
        "",
        "| Проверка | Требование | current | updated |",
        "|---|---:|---:|---:|",
    ]
    for name, metric_threshold in PIPELINE_THRESHOLDS.items():
        lines.append(
            f"| {METRIC_NAMES[name]} | ≥ {metric_threshold:.0%} | "
            f"{_threshold_cell(current['metrics'][name])} | "
            f"{_threshold_cell(updated['metrics'][name])} |"
        )
    introduced_text = ", ".join(f"`{case}`" for case in introduced) if introduced else "нет"
    lines += [
        f"| Новые технические ошибки | нет | базовая версия | "
        f"{introduced_text}{' — новая' if introduced else ''} "
        f"{'❌' if introduced else '✅'} |",
        "",
        f"При {updated_decision['denominator']} эталонных сущностях порог {threshold:.0%} "
        f"означает минимум **{required} правильных решения**. "
        f"`current`: {current_decision['numerator']}/{current_decision['denominator']} "
        f"(до минимума — {max(0, required - current_decision['numerator'])}); "
        f"`updated`: {updated_decision['numerator']}/{updated_decision['denominator']} "
        f"(до минимума — {max(0, required - updated_decision['numerator'])}).",
        "",
        "Прохождение определяется по точным значениям до округления; "
        "отображаемые проценты на решение не влияют.",
    ]
    return lines


def _comparison_lines(report: dict[str, Any]) -> list[str]:
    pipeline = report["pipeline_evaluation"]
    current = pipeline["versions"]["current"]
    updated = pipeline["versions"]["updated"]
    changes = pipeline["comparison"]["metrics"]
    improved = {name for name, item in changes.items() if item["change"] == "improved"}

    lines = [
        "## Что изменилось в `updated`",
        "",
        "| Показатель | current | updated | Δ | Вывод |",
        "|---|---:|---:|---:|---|",
    ]
    labels = {
        "improved": "лучше",
        "worsened": "**хуже**",
        "unchanged": "без изменений",
    }
    for name, change in changes.items():
        lines.append(
            f"| {METRIC_NAMES[name]} | {change['current']:.2%} | "
            f"{change['updated']:.2%} | {change['delta'] * 100:+.2f} п.п. | "
            f"{labels[change['change']]} |"
        )
    if (
        current["counters"]["E_TP"] != updated["counters"]["E_TP"]
        and {"role_accuracy", "a_accuracy"} & improved
    ):
        lines += [
            "",
            "Примечание: рост долей роли и A наблюдается на меньшем числе "
            f"сопоставленных сущностей: `{current['counters']['E_TP']}` → "
            f"`{updated['counters']['E_TP']}`. Это не означает общего улучшения извлечения.",
        ]
    return lines


def _technical_lines(report: dict[str, Any]) -> list[str]:
    pipeline = report["pipeline_evaluation"]
    versions = pipeline["versions"]
    current, updated = versions["current"], versions["updated"]
    technical = pipeline["comparison"]["technical_errors"]
    introduced = technical["introduced_case_ids"]
    updated_decision = updated["metrics"]["decision_accuracy"]
    required = ceil(updated_decision["threshold"] * updated_decision["denominator"])
    shortfall = max(0, required - updated_decision["numerator"])

    lines = ["## Технические ошибки", ""]
    for version, result in versions.items():
        details = result["technical_errors"]["details"]
        if not details:
            lines.append(f"- `{version}`: нет.")
        for detail in details:
            states = []
            if detail["case_id"] in technical["resolved_case_ids"]:
                states.append("исправлена в updated")
            if detail["case_id"] in introduced:
                states.append("новая в updated")
            suffix = f"; {', '.join(states)}" if states else ""
            lines.append(
                f"- `{version}/{detail['case_id']}`: "
                f"{'; '.join(_technical_reason(reason) for reason in detail['reasons'])}"
                f"{suffix}."
            )
    if technical["current_count"] == technical["updated_count"] and (
        technical["resolved_case_ids"] or introduced
    ):
        lines += [
            "",
            f"Общее число ошибок не изменилось: "
            f"`{technical['current_count']}` → `{technical['updated_count']}`. Это не "
            "нейтральный результат: старая ошибка исчезла, но появилась новая.",
        ]
    else:
        lines += [
            "",
            f"Общее число ошибок: `{technical['current_count']}` → "
            f"`{technical['updated_count']}`.",
        ]

    for case_id in introduced:
        missing = [
            item
            for item in updated["entity_details"]["missing"]
            if item["case_id"] == case_id and item["technical_error"]
        ]
        positive_missing = sum(item["expected_pass"] is True for item in missing)
        current_case_has_issues = any(
            item["case_id"] == case_id
            for category in current["entity_details"].values()
            for item in category
        ) or case_id in current["technical_errors"]["case_ids"]
        lines += [
            "",
            f"Техническая ошибка в `{case_id}` превратила {len(missing)} ожидаемые "
            "сущности в пропуски; "
            f"среди них {positive_missing} значимые (`expected_pass=true`).",
        ]
        if not current_case_has_issues and positive_missing == shortfall:
            counterfactual = updated_decision["numerator"] + positive_missing
            lines.append(
                f"Если восстановить **именно корректный результат уровня `current`** "
                f"на этом кейсе, `updated` получит "
                f"{counterfactual}/{updated_decision['denominator']} = "
                f"{counterfactual / updated_decision['denominator']:.2%} по решениям. "
                "Простое устранение структурной ошибки без восстановления "
                "содержимого этого не гарантирует."
            )
    return lines


def _judge_lines(report: dict[str, Any]) -> list[str]:
    judge = report["judge_evaluation"]
    overall = judge["overall"]
    lines = [
        "## Насколько можно доверять судье",
        "",
        f"В целом судья совпал с `reference_pass` в "
        f"**{overall['agreement']['numerator']}/{overall['agreement']['denominator']} = "
        f"{overall['agreement']['value']:.2%}** случаев: "
        f"{overall['false_acceptances']} ошибочных принятий и "
        f"{overall['false_rejections']} ошибочных отклонений.",
        "",
        "| Версия | Критерий | Совпадение с человеком | Ошибочно принял | Ошибочно отклонил |",
        "|---|---|---:|---:|---:|",
    ]
    groups = []
    disagreement_case_ids = []
    for version in ("current", "updated"):
        for criterion in JUDGE_THRESHOLDS:
            group = judge["by_version"][version][criterion]
            groups.append((version, criterion, group))
            disagreement_case_ids.extend(
                disagreement["case_id"] for disagreement in group["disagreements"]
            )
            agreement = group["agreement"]
            lines.append(
                f"| {version} | {CRITERION_NAMES[criterion]} | "
                f"{agreement['numerator']}/{agreement['denominator']} = "
                f"{agreement['value']:.2%} | {group['false_acceptances']} | "
                f"{group['false_rejections']} |"
            )

    weakest_version, weakest_criterion, weakest = min(
        groups, key=lambda item: item[2]["agreement"]["value"]
    )
    weakest_agreement = weakest["agreement"]
    lines += [
        "",
        f"Самая слабая группа — `{weakest_version}` / "
        f"{CRITERION_NAMES[weakest_criterion]}: "
        f"**{weakest_agreement['numerator']}/{weakest_agreement['denominator']} = "
        f"{weakest_agreement['value']:.2%}**. В ней {weakest['false_acceptances']} "
        f"ошибочных принятия и {weakest['false_rejections']} ошибочных отклонения.",
    ]
    repeated = [
        (case_id, count)
        for case_id, count in sorted(Counter(disagreement_case_ids).items())
        if count > 1
    ]
    if repeated:
        lines += [
            "",
            "Число disagreement-записей в скобках: "
            + ", ".join(f"`{case_id}` ({count})" for case_id, count in repeated)
            + ". Это кандидаты на ручной аудит, "
            "но по этим данным "
            "нельзя однозначно отделить проблему judge, rubric или reference.",
        ]
    lines += [
        "",
        "Порог приемлемого качества судьи в задании не задан, поэтому его нельзя "
        "формально объявить `passed` или `failed`. Этих данных недостаточно, чтобы "
        "использовать судью как единственный release gate; pipeline по-прежнему "
        "оценивается по golden-разметке.",
    ]
    return lines


def _limitations_lines(report: dict[str, Any]) -> list[str]:
    dataset = report["dataset"]
    provenance = report["provenance"]
    first_group = report["judge_evaluation"]["by_version"]["current"][
        next(iter(JUDGE_THRESHOLDS))
    ]
    lines = [
        "## Ограничения вывода",
        "",
        f"- Выборка состоит из {dataset['case_count']} фиксированных кейсов и "
        f"{dataset['expected_entity_count']} эталонных сущностей; это описательный "
        "результат, а не доказательство продовой обобщаемости.",
        f"- Каждая группа проверки судьи содержит {first_group['total']} наблюдения.",
        "- `reference_pass` трактуется как предоставленная эталонная метка; число "
        "аннотаторов и их согласованность не известны.",
        "- Проверен один snapshot: "
        f"dataset `{', '.join(provenance['dataset_versions'])}`, "
        f"judge `{', '.join(provenance['judge_model_versions'])}`, "
        f"prompt `{', '.join(provenance['judge_prompt_versions'])}`, "
        f"rubric `{', '.join(provenance['rubric_versions'])}`.",
    ]
    if not dataset["slices"]:
        lines.append(
            "- `metadata.slices` не заполнены, поэтому slice-анализ на этом наборе невозможен."
        )
    lines.append(
        "- Оцениваются только сохранённые outputs; pipeline и модель-судья заново не запускаются."
    )
    return lines


def _next_steps_lines(report: dict[str, Any]) -> list[str]:
    pipeline = report["pipeline_evaluation"]
    updated = pipeline["versions"]["updated"]
    comparison = pipeline["comparison"]
    if comparison["can_replace"]:
        steps = ["Зафиксировать прохождение release-условий и переходить к замене."]
    else:
        decision = updated["metrics"]["decision_accuracy"]
        required = ceil(decision["threshold"] * decision["denominator"])
        introduced = comparison["technical_errors"]["introduced_case_ids"]
        steps = ["Не заменять `current` на `updated` в текущем виде."]
        if introduced:
            steps.append(
                "Сначала исправить новую техническую регрессию: "
                + ", ".join(f"`{case}`" for case in introduced)
                + "."
            )
        steps += [
            "Разобрать регрессию извлечения по `entity_details`: "
            f"совпавших сущностей стало {pipeline['versions']['current']['counters']['E_TP']} → "
            f"{updated['counters']['E_TP']}, пропущенных — "
            f"{pipeline['versions']['current']['counters']['E_FN']} → {updated['counters']['E_FN']}, "
            f"лишних — {pipeline['versions']['current']['counters']['E_FP']} → "
            f"{updated['counters']['E_FP']}.",
            "Повторно запустить офлайн-eval и убедиться, что `updated` "
            f"набрал не менее {required}/{decision['denominator']} правильных решений, "
            "прошёл остальные пороги и не добавил технических ошибок.",
            "Перед использованием судьи как release gate вручную разобрать повторные "
            "расхождения и слабую группу `updated` / адекватность значимости.",
        ]
    return ["## Что делать дальше", ""] + [
        f"{index}. {step}" for index, step in enumerate(steps, 1)
    ]


def render_conclusion(report: dict[str, Any]) -> str:
    """Render conclusion.md from calculated values, with the decision first."""

    lines = ["# Итог оценки"]
    for section in (
        _verdict_lines(report),
        _gate_lines(report),
        _comparison_lines(report),
        _technical_lines(report),
        _judge_lines(report),
        _limitations_lines(report),
        _next_steps_lines(report),
    ):
        lines += [""] + section
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], report_path: Path, conclusion_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    conclusion_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    conclusion_path.write_text(render_conclusion(report), encoding="utf-8")
