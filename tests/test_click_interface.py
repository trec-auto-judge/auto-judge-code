import unittest
from click.testing import CliRunner
import click
import json
from pathlib import Path
from typing import List
from trec_auto_judge import option_rag_responses, Report
from . import TREC_25_DATA


@click.command()
@option_rag_responses()
def return_responses(rag_responses: List[Report]):
    json_serializable = []
    for report in rag_responses:
        json_serializable.append({
           'team_id': report.metadata.team_id,
           'run_id': report.metadata.run_id,
           'topic_id': report.metadata.topic_id,
           'narrative_id': report.metadata.topic_id,
           "path": str(report.path)
        })
    print(json.dumps(json_serializable))
    return 0

class TestClickInterface(unittest.TestCase):
    def test_trec25_spot_check_runs(self):
        expected_paths = [
            'my_best_run_01.jsonl',
            'my_best_run_02.jsonl',
            'my_best_run_02_with_citations.jsonl'
        ]

        runner = CliRunner()
        result = runner.invoke(return_responses, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs"])

        print(result.stdout)
        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        actual = json.loads(result.stdout.split("\n")[-2])

        self.assertEqual(3, len(actual))
        self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])


    def test_path_that_does_not_exist(self):
        runner = CliRunner()
        result = runner.invoke(return_responses, ["--rag-responses", "/does-not-exist"])

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Invalid value for '--rag-responses': The directory /does-not-exist does not exist", result.output)