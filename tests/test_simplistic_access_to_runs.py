import unittest
from trec_auto_judge.io import load_runs_failsave
from pathlib import Path
from . import TREC_25_DATA
from tempfile import TemporaryDirectory
from glob import glob
import json

class TestSimplisticAccessToRuns(unittest.TestCase):
    def test_non_existing_repo_yields_no_runs(self):
        expected = []
        actual = load_runs_failsave(Path("/this-directory/does-not-exist"))
        self.assertEqual(expected, actual)

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
        actual = load_runs_failsave(TREC_25_DATA / "spot-check-dataset")
        self.assertEqual(3, len(actual))
        self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
        self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    def test_trec25_spot_check_runs_with_narrative(self):
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

        with TemporaryDirectory() as d:
            for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
                f = Path(f)
                txt = []
                for l in f.read_text().split("\n"):
                    if not l:
                        continue
                    l = json.loads(l)
                    l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]
                    del l["metadata"]["topic_id"]
                    txt.append(json.dumps(l))
                (Path(d) / f.name).write_text("\n".join(txt))

            actual = load_runs_failsave(Path(d))
            self.assertEqual(3, len(actual))
            self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
            self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])

    def test_trec25_spot_check_runs_with_narrative_and_topic_id(self):
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

        with TemporaryDirectory() as d:
            for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
                f = Path(f)
                txt = []
                for l in f.read_text().split("\n"):
                    if not l:
                        continue
                    l = json.loads(l)
                    l["metadata"]["narrative_id"] = l["metadata"]["topic_id"]
                    txt.append(json.dumps(l))
                (Path(d) / f.name).write_text("\n".join(txt))

            actual = load_runs_failsave(Path(d))
            self.assertEqual(3, len(actual))
            self.assertEqual(expected_metadata, [i["metadata"] for i in actual])
            self.assertEqual(expected_paths, [Path(i["path"]).name for i in actual])


    def test_trec25_spot_check_runs_with_inconsistent_narrative_and_topic_id(self):
        with TemporaryDirectory() as d:
            for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
                f = Path(f)
                txt = []
                for l in f.read_text().split("\n"):
                    if not l:
                        continue
                    l = json.loads(l)
                    l["metadata"]["narrative_id"] = l["metadata"]["topic_id"] + "1"
                    txt.append(json.dumps(l))
                (Path(d) / f.name).write_text("\n".join(txt))

            with self.assertRaises(ValueError):
                load_runs_failsave(Path(d))

    def test_trec25_spot_check_runs_with_duplicate_narrative_and_topic_id(self):
        with TemporaryDirectory() as d:
            for f in glob(str(TREC_25_DATA / "spot-check-dataset" / "runs") + "/*" ):
                f = Path(f)
                (Path(d) / f.name).write_text(f.read_text())
                (Path(d) / (f.name + "-duplicate")).write_text(f.read_text())

            with self.assertRaises(ValueError):
                load_runs_failsave(Path(d))