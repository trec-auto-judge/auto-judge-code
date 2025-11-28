import unittest
from click.testing import CliRunner
import click
import json
from pathlib import Path
from trec_auto_judge.click import option_rag_responses
from . import TREC_25_DATA


@click.command()
@option_rag_responses()
def return_responses(rag_responses):
    print(json.dumps(rag_responses))
    return 0

class TestClickInterface(unittest.TestCase):
    def test_trec25_spot_check_runs(self):
        expected_metadata = [
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_01', 'topic_id': '101', 'narrative_id': '101'},
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '101', 'narrative_id': '101'},
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02_citations', 'topic_id': '101', 'narrative_id': '101'}
        ]
        expected_paths = [
            'my_best_run_01',
            'my_best_run_02',
            'my_best_run_02_with_citations'
        ]

        runner = CliRunner()
        result = runner.invoke(return_responses, ["--rag-responses", TREC_25_DATA / "spot-check-dataset"])

        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        actual = json.loads(result.stdout)

        self.assertEqual(3, len(actual))
        self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
        self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    def test_path_that_does_not_exist(self):
        runner = CliRunner()
        result = runner.invoke(return_responses, ["--rag-responses", "/does-not-exist"])

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Invalid value for '--rag-responses': The directory /does-not-exist does not exist", result.stderr)