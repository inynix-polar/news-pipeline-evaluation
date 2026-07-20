from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_QUOTES = str.maketrans("", "", '"\'`«»„“”‘’')
_EDGES = " \t\r\n\f\v.,;:!?"


def normalize_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.casefold().replace("ё", "е").translate(_QUOTES)
    return " ".join(value.split()).strip(_EDGES)


def _names(entity: object) -> set[str]:
    if not isinstance(entity, Mapping):
        return set()
    values: list[object] = [entity.get("name")]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        values.extend(aliases)
    return {name for value in values if (name := normalize_name(value))}


def match_entities(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    actual_names = [
        normalize_name(entity.get("name")) if isinstance(entity, Mapping) else None
        for entity in actual
    ]
    unused = set(range(len(actual)))
    pairs: list[tuple[int, int]] = []
    missing: list[int] = []
    for expected_index, entity in enumerate(expected):
        allowed = _names(entity)
        actual_index = next(
            (
                index
                for index, name in enumerate(actual_names)
                if index in unused and name and name in allowed
            ),
            None,
        )
        if actual_index is None:
            missing.append(expected_index)
        else:
            pairs.append((expected_index, actual_index))
            unused.remove(actual_index)
    return pairs, missing, sorted(unused)
