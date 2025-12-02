import unittest
from click.testing import CliRunner
import click
import json
from pathlib import Path
from . import TREC_25_DATA
from trec_auto_judge import main
from tempfile import TemporaryDirectory

EXAMPLE_LEADERBOARD = str((TREC_25_DATA / "spot-check-dataset" / "trec-leaberboard").absolute())

def evaluate_command(measure, truth=EXAMPLE_LEADERBOARD, inp=EXAMPLE_LEADERBOARD):
    cmd = ["evaluate", "--truth-leaderboard", truth, "--input", inp, "--truth-metric", measure]
    return run_cmd_on_main(cmd)
    

def run_cmd_on_main(cmd):
    runner = CliRunner()
    ret = runner.invoke(main, cmd)
    return ret, ' '.join(ret.stdout.split())


class TestEvaluationInterface(unittest.TestCase):
    def test_trec25_spot_check_runs_measure_01(self):
        expected_lines = [
            "trec-leaberboard Measure-02 -1.0",
            "trec-leaberboard Measure-01 1.0"
        ]
        result, stdout = evaluate_command("Measure-01")

        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        for l in expected_lines:
            self.assertIn(l, stdout)

    def test_trec25_spot_check_runs_measure_02(self):
        expected_lines = [
            "trec-leaberboard Measure-02 1.0",
            "trec-leaberboard Measure-01 -1.0"
        ]
        result, stdout = evaluate_command("Measure-02")

        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        for l in expected_lines:
            self.assertIn(l, stdout)

    def test_trec25_spot_check_runs_without_truth(self):
        expected_lines = [
            "trec-leaberboard Measure-02 2.0",
            "trec-leaberboard Measure-01 2.0"
        ]
        cmd = ["evaluate", "--input", EXAMPLE_LEADERBOARD]
        result, stdout = run_cmd_on_main(cmd)

        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        for l in expected_lines:
            self.assertIn(l, stdout)

    def test_trec25_spot_check_runs_measure_01_but_fails_on_output(self):
            target_file = Path("/tmp/results.jsonl-does")

            self.assertFalse(target_file.is_file())

            cmd = ["evaluate", "--truth-leaderboard", EXAMPLE_LEADERBOARD, "--input", EXAMPLE_LEADERBOARD, "--truth-metric", "Measure-01", "--output", target_file]
            result, stdout = run_cmd_on_main(cmd)

            self.assertIsNotNone(result.exception)
            self.assertEqual(result.exit_code, 1)
            self.assertFalse(target_file.is_file())
            self.assertIn("trec-leaberboard Measure-02 -1.0", stdout)

    def test_trec25_spot_check_runs_measure_01_and_produces_output(self):
        expected_lines = [
            "trec-leaberboard Measure-02 -1.0",
            "trec-leaberboard Measure-01 1.0"
        ]

        with TemporaryDirectory() as tmp_dir:
            target_file = (Path(tmp_dir) / "results.jsonl").absolute()

            self.assertFalse(target_file.is_file())

            cmd = ["evaluate", "--truth-leaderboard", EXAMPLE_LEADERBOARD, "--input", EXAMPLE_LEADERBOARD, "--truth-metric", "Measure-01", "--output", target_file]
            result, stdout = run_cmd_on_main(cmd)

            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)

            for l in expected_lines:
                self.assertIn(l, stdout)

            self.assertTrue(target_file.is_file())
            self.assertIn('"Metric":"Measure-01","kendall":1.0', target_file.read_text())


    def test_trec25_spot_check_runs_measure_01_and_produces_aggregated_protetextoutput(self):
        with TemporaryDirectory() as tmp_dir:
            target_file = (Path(tmp_dir) / "results.prototext").absolute()

            self.assertFalse(target_file.is_file())

            cmd = ["evaluate", "--input", EXAMPLE_LEADERBOARD, "--aggregate", "--output", target_file]
            result, stdout = run_cmd_on_main(cmd)

            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("stdev-value 2 2.0", stdout)

            self.assertTrue(target_file.is_file())
            self.assertIn('measure{\n  key: "Judges"\n  value: "2.0"\n}\n', target_file.read_text())