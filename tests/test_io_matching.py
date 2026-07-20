from __future__ import annotations

from pathlib import Path
import json
import shutil
from tempfile import TemporaryDirectory
import unittest

from news_eval.matching import match_entities, normalize_name
from news_eval.readers import DataError, load_data, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class MatchingTests(unittest.TestCase):
    def test_normalization_is_exactly_the_specified_normalization(self) -> None:
        self.assertEqual(normalize_name('  ... «Ёлка»\t"ТЕСТ" !?  '), "елка тест")
        self.assertEqual(normalize_name('"\'`«»„“”‘’Ёж'), "еж")
        self.assertEqual(normalize_name("А-Б"), "а-б")
        self.assertIsNone(normalize_name(None))

    def test_matching_is_greedy_one_to_one_and_leaves_duplicates_extra(self) -> None:
        expected = [
            {"name": "Alpha", "aliases": ["Shared"]},
            {"name": "Shared", "aliases": []},
            {"name": "Missing", "aliases": []},
        ]
        actual = [{"name": "shared"}, {"name": "Alpha"}, {"name": "Alpha"}]

        pairs, missing, extra = match_entities(expected, actual)

        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual(missing, [1, 2])
        self.assertEqual(extra, [1, 2])

    def test_matching_uses_only_canonical_names_and_explicit_aliases(self) -> None:
        expected = [{"name": "ООО Ромашка", "aliases": ["«Ромашка Групп»"]}]
        self.assertEqual(
            match_entities(expected, [{"name": "ромашка групп"}]),
            ([(0, 0)], [], []),
        )
        self.assertEqual(
            match_entities(expected, [{"name": "Ромашка"}]),
            ([], [0], [0]),
        )


class ReaderTests(unittest.TestCase):
    def test_provided_data_passes_integrity_and_has_hashes(self) -> None:
        data, integrity = load_data(DATA)

        self.assertEqual(
            {name: len(rows) for name, rows in data.items()},
            {"goldens": 24, "pipeline_outputs": 48, "judge_outputs": 96},
        )
        self.assertTrue(integrity["passed"])
        files = integrity["files"]
        self.assertEqual(
            {name: report["sha256"] for name, report in files.items()},
            {
                "goldens.jsonl": "bd243c0ff0fbfd9eb2ae7caf273b6c857bc4ef371a79a1235c6817faa5cbdcd3",
                "pipeline_outputs.jsonl": "f61170ff1a272b42e3f74bf2b5b736f7c1cef4d5a743b85d82e9e35fd7f72792",
                "judge_outputs.jsonl": "175779509680d8a980247445f33637a62142cf6888d5ee82793b9dfcda3fa76c",
            },
        )
        self.assertEqual(
            files["pipeline_outputs.jsonl"]["complete_combinations"]["expected_count"],
            48,
        )
        self.assertEqual(
            files["judge_outputs.jsonl"]["complete_combinations"]["expected_count"],
            96,
        )

    def test_jsonl_rejects_non_objects_blank_lines_and_non_json_numbers(self) -> None:
        bad_inputs = ["[]\n", "{}\n\n", '{"x":NaN}\n', '{"x":1,"x":2}\n']
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            for content in bad_inputs:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(DataError):
                        load_jsonl(path)

    def test_integrity_error_contains_duplicate_and_missing_combinations(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shutil.copytree(DATA, directory / "data")
            copied = directory / "data"
            rows = load_jsonl(copied / "pipeline_outputs.jsonl")
            rows[-1] = rows[0]
            (copied / "pipeline_outputs.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataError, "pipeline_outputs.jsonl") as caught:
                load_data(copied)

            report = caught.exception.integrity["files"]["pipeline_outputs.jsonl"]
            self.assertTrue(report["row_count"]["passed"])
            self.assertFalse(report["unique_key"]["passed"])
            self.assertEqual(len(report["complete_combinations"]["missing"]), 1)

    def test_integrity_error_reports_orphan_case_id(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shutil.copytree(DATA, directory / "data")
            copied = directory / "data"
            rows = load_jsonl(copied / "judge_outputs.jsonl")
            rows[0] = {**rows[0], "case_id": "orphan"}
            (copied / "judge_outputs.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaises(DataError) as caught:
                load_data(copied)

            report = caught.exception.integrity["files"]["judge_outputs.jsonl"]
            self.assertEqual(report["orphan_case_ids"]["values"], ["orphan"])
            self.assertTrue(report["complete_combinations"]["missing"])
            self.assertTrue(report["complete_combinations"]["unexpected"])


if __name__ == "__main__":
    unittest.main()
