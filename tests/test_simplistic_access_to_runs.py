import unittest
from trec_auto_judge.io import load_runs_failsave
from pathlib import Path


class TestSimplisticAccessToRuns(unittest.TestCase):
    def test_non_existing_repo_yields_no_runs(self):
        expected = []
        actual = load_runs_failsave(Path("/this-directory/does-not-exist"))
        self.assertEqual(expected, actual)

    def test_trec25_spot_check_runs(self):
        expected_metadata = [
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '101'},
            {'team_id': 'my_fantastic_team', 'run_id': 'my_best_run_02', 'topic_id': '101'}
        ]
        expected_paths = [
            'my_best_run_02',
            'my_best_run_02_with_citations'
        ]
        actual = load_runs_failsave(Path(__file__).parent.parent / "trec25" / "spot-check-dataset")
        self.assertEqual(2, len(actual))
        self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
        self.assertEqual(expected_paths, [i["path"].name for i in actual])

