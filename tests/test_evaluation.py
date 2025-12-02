import unittest
from tempfile import TemporaryDirectory
from trec_auto_judge.evaluation import TrecLeaderboardEvaluation
from pathlib import Path

EXAMPLE_01 = """
run_01 ORACLE query-01 12
run_02 ORACLE query-01 12
run_03 ORACLE query-01 12
run_04 ORACLE query-01 12
run_01 ORACLE all 1
run_02 ORACLE all 2
run_03 ORACLE all 3
run_04 ORACLE all 4
"""

EXAMPLE_02 = """
run_01 M1 query-01 12
run_02 M1 query-01 12
run_03 M1 query-01 12
run_04 M1 query-01 12
run_01 M1 all 1
run_02 M1 all 3
run_03 M1 all 2
run_04 M1 all 4
run_01 M2 query-01 12
run_02 M2 query-01 12
run_03 M2 query-01 12
run_04 M2 query-01 12
run_01 M2 all 4
run_02 M2 all 3
run_03 M2 all 2
run_04 M2 all 1
"""

class TestEvaluation(unittest.TestCase):
    def test_correlation_is_perfect_for_identical_leaderboards(self):
        with TemporaryDirectory() as d:
            expected = {'ORACLE': {'kendall': 1.0, 'pearson': 1.0, 'spearman': 1.0, 'tauap_b': 1.0}}
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_01)
            te = TrecLeaderboardEvaluation(leaderboard, "ORACLE")
            actual = te.evaluate(leaderboard)
            self.assertEqual(expected, actual)

    def test_correlation_is_perfect_for_identical_leaderboards_on_multiple_measures_01(self):
        with TemporaryDirectory() as d:
            expected = {
                'M1': {'kendall': -0.666666666, 'pearson': -0.8, 'spearman': -0.8, 'tauap_b': -0.666666666},
                'M2': {'kendall': 1.0, 'pearson': 1.0, 'spearman': 1.0, 'tauap_b': 1.0},
            }
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)
            te = TrecLeaderboardEvaluation(leaderboard, "M2")
            actual = te.evaluate(leaderboard)
            self.assertIn("M1", actual)
            self.assertIn("M2", actual)

            self.assertEqual(expected["M2"], actual["M2"])
            for m in actual["M2"].keys():
                self.assertAlmostEqual(expected["M1"][m], actual["M1"][m], 5, m)

    def test_correlation_is_perfect_for_identical_leaderboards_on_multiple_measures_02(self):
        with TemporaryDirectory() as d:
            expected = {
                'M1': {'kendall': 1.0, 'pearson': 1.0, 'spearman': 1.0, 'tauap_b': 1.0},
                'M2': {'kendall': -0.666666666, 'pearson': -0.8, 'spearman': -0.8, 'tauap_b': -0.666666666},
            }
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)
            te = TrecLeaderboardEvaluation(leaderboard, "M1")
            actual = te.evaluate(leaderboard)
            self.assertIn("M1", actual)
            self.assertIn("M2", actual)

            self.assertEqual(expected["M1"], actual["M1"])
            for m in actual["M2"].keys():
                self.assertAlmostEqual(expected["M2"][m], actual["M2"][m], 5, m)

    def test_correlation_on_two_leaderboards_01(self):
        with TemporaryDirectory() as d:
            expected = {
                'M1': {'kendall': 0.666666666, 'pearson': 0.8, 'spearman': 0.8, 'tauap_b': 0.666666666},
                'M2': {'kendall': -1.0, 'pearson': -1.0, 'spearman': -1.0, 'tauap_b': -1.0},
            }
            l1 = Path(d) / "leaderboard-1"
            l1.write_text(EXAMPLE_01)
            l2 = Path(d) / "leaderboard-2"
            l2.write_text(EXAMPLE_02)

            te = TrecLeaderboardEvaluation(l1, "ORACLE")
            actual = te.evaluate(l2)
            self.assertIn("M1", actual)
            self.assertIn("M2", actual)

            self.assertEqual(expected["M2"], actual["M2"])
            for m in actual["M2"].keys():
                self.assertAlmostEqual(expected["M1"][m], actual["M1"][m], 5, m)

    def test_correlation_on_two_leaderboards_02(self):
        with TemporaryDirectory() as d:
            expected = {
                'ORACLE': {'kendall': -1.0, 'pearson': -1.0, 'spearman': -1.0, 'tauap_b': -1.0},
            }
            l1 = Path(d) / "leaderboard-1"
            l1.write_text(EXAMPLE_01)
            l2 = Path(d) / "leaderboard-2"
            l2.write_text(EXAMPLE_02)

            te = TrecLeaderboardEvaluation(l2, "M2")
            actual = te.evaluate(l1)

            self.assertEqual(expected, actual)

    def test_evaluation_throws_exception_for_non_existing_measure(self):
        with TemporaryDirectory() as d:
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)

            with self.assertRaises(ValueError):
                TrecLeaderboardEvaluation(leaderboard, "measure-does-not-exist")

    def test_evaluation_ground_truth_does_not_exist(self):
        expected = {
            'M1': {'mean-value': 7.25, "stdev-value": 5.14781507},
            'M2': {'mean-value': 7.25, "stdev-value": 5.14781507},
        }
        with TemporaryDirectory() as d:
            leaderboard = Path(d) / "leaderboard"
            leaderboard.write_text(EXAMPLE_02)

            te = TrecLeaderboardEvaluation(None, None)
            actual = te.evaluate(leaderboard)

            for m in expected.keys():
                for k in expected[m].keys():
                    self.assertAlmostEqual(expected[m][k], actual[m][k], 5, f"{m} {k}")