import unittest

# from trec_auto_judge import Leaderboard, LeaderboardEntry
from trec_auto_judge import *
from tempfile import TemporaryDirectory
from pathlib import Path
from approvaltests import verify_file

class TestLeaderboard(unittest.TestCase):
    def test_valid_leaderboard_float(self):
        
        MY_SPEC = LeaderboardSpec(measures=(
            MeasureSpec("measure-01", aggregate=mean_of_floats, cast=float),
        ))
        b = LeaderboardBuilder(MY_SPEC)

        b.add(run_id="run-01",topic_id="topic-01",values={"measure-01": 2.0})
        b.add(run_id="run-01",topic_id="topic-02",values={"measure-01": 0.0})
    
        l = b.build()
        # will throw exception if mistake found
        LeaderboardVerification(l, on_missing="error").complete_measures()
        
        with TemporaryDirectory() as tmp_dir:
            target_file = Path(tmp_dir) / "leaderboard"
            l.write(target_file)
            verify_file(target_file)

    def test_valid_leaderboard_mean_bools_all_true(self):
        
        MY_SPEC = LeaderboardSpec(measures=(
            MeasureSpec("measure-01", aggregate=mean_of_bools, cast=lambda x:x),
        ))
        b = LeaderboardBuilder(MY_SPEC)

        b.add(run_id="run-01",topic_id="topic-01",values={"measure-01": True})
        b.add(run_id="run-01",topic_id="topic-02",values={"measure-01": True})
    
        l = b.build()
        
        with TemporaryDirectory() as tmp_dir:
            target_file = Path(tmp_dir) / "leaderboard"
            l.write(target_file)
            verify_file(target_file)

    # def test_valid_leaderboard_mean_bools_all_true(self):
    #     measures = {
    #         "measure-01": mean_of_floats
    #     }

    #     entries = [
    #         LeaderboardEntry("run-01", "topic-01", {"measure-01": 3.0}),
    #         LeaderboardEntry("run-01", "topic-02", {"measure-01": 4.0}),
    #     ]

    #     l = Leaderboard.from_entries_with_all(measures=measures, entries=entries)
    #     with TemporaryDirectory() as tmp_dir:
    #         target_file = Path(tmp_dir) / "leaderboard"
    #         l.write(target_file)
    #         verify_file(target_file)


class TestLeaderboardBuildOnMissing(unittest.TestCase):
    """Tests for build() with expected_topic_ids and on_missing parameter."""

    def setUp(self):
        """Create a spec with default value for testing."""
        self.spec = LeaderboardSpec(measures=(
            MeasureSpec("SCORE", aggregate=mean_of_floats, cast=float, default=0.0),
        ))
        self.expected_topics = ["t1", "t2", "t3"]

    def test_build_without_expected_topics_no_filling(self):
        """Without expected_topic_ids, missing topics are not filled."""
        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # t3 missing for run-A

        lb = b.build()  # No expected_topic_ids

        # Aggregate should be mean of [0.9, 0.9] = 0.9 (not 0.6 with default)
        all_entry = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_entry.values["SCORE"], 0.9)

    def test_build_on_missing_default_fills_silently(self):
        """on_missing='default' silently fills missing entries with defaults."""
        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # t3 missing for run-A

        lb = b.build(expected_topic_ids=self.expected_topics, on_missing="default")

        # Aggregate should be mean of [0.9, 0.9, 0.0] = 0.6
        all_entry = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_entry.values["SCORE"], 0.6)

    def test_build_on_missing_warn_fills_and_warns(self):
        """on_missing='warn' fills defaults and prints warning."""
        import io
        import sys

        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # t3 missing

        # Capture stderr
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured

        try:
            lb = b.build(expected_topic_ids=self.expected_topics, on_missing="warn")
        finally:
            sys.stderr = old_stderr

        # Check warning was printed
        warning_output = captured.getvalue()
        self.assertIn("Warning", warning_output)
        self.assertIn("run-A", warning_output)
        self.assertIn("t3", warning_output)

        # Check defaults were filled (mean of [0.9, 0.9, 0.0] = 0.6)
        all_entry = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_entry.values["SCORE"], 0.6)

    def test_build_on_missing_error_raises(self):
        """on_missing='error' raises ValueError when topics are missing."""
        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # t3 missing

        with self.assertRaises(ValueError) as ctx:
            b.build(expected_topic_ids=self.expected_topics, on_missing="error")

        self.assertIn("run-A", str(ctx.exception))
        self.assertIn("t3", str(ctx.exception))

    def test_build_complete_data_no_filling(self):
        """When all topics present, no filling occurs regardless of on_missing."""
        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.8)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t3", SCORE=0.7)

        lb = b.build(expected_topic_ids=self.expected_topics, on_missing="error")

        # No error raised, aggregate is mean of [0.8, 0.9, 0.7] = 0.8
        all_entry = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_entry.values["SCORE"], 0.8)

    def test_build_multiple_runs_missing_different_topics(self):
        """Multiple runs with different missing topics."""
        b = LeaderboardBuilder(self.spec)
        # run-A missing t3
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # run-B missing t2
        b.add(run_id="run-B", topic_id="t1", SCORE=0.6)
        b.add(run_id="run-B", topic_id="t3", SCORE=0.6)

        lb = b.build(expected_topic_ids=self.expected_topics, on_missing="default")

        # run-A: mean of [0.9, 0.9, 0.0] = 0.6
        all_A = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_A.values["SCORE"], 0.6)

        # run-B: mean of [0.6, 0.0, 0.6] = 0.4
        all_B = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-B"][0]
        self.assertAlmostEqual(all_B.values["SCORE"], 0.4)

    def test_build_fix_aggregate_no_entries_created(self):
        """fix_aggregate mode fills defaults for 'all' row but doesn't create per-topic entries."""
        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # t3 missing

        lb = b.build(expected_topic_ids=self.expected_topics, on_missing="fix_aggregate")

        # Aggregate should include default (mean of [0.9, 0.9, 0.0] = 0.6)
        all_entry = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_entry.values["SCORE"], 0.6)

        # But NO per-topic entry created for t3
        t3_entries = [e for e in lb.entries if e.topic_id == "t3"]
        self.assertEqual(len(t3_entries), 0)

    def test_build_default_creates_entries(self):
        """'default' mode creates actual per-topic entries for missing topics."""
        b = LeaderboardBuilder(self.spec)
        b.add(run_id="run-A", topic_id="t1", SCORE=0.9)
        b.add(run_id="run-A", topic_id="t2", SCORE=0.9)
        # t3 missing

        lb = b.build(expected_topic_ids=self.expected_topics, on_missing="default")

        # Per-topic entry SHOULD exist for t3
        t3_entries = [e for e in lb.entries if e.topic_id == "t3" and e.run_id == "run-A"]
        self.assertEqual(len(t3_entries), 1)
        self.assertAlmostEqual(t3_entries[0].values["SCORE"], 0.0)  # default value

        # Aggregate also correct
        all_entry = [e for e in lb.entries if e.topic_id == "all" and e.run_id == "run-A"][0]
        self.assertAlmostEqual(all_entry.values["SCORE"], 0.6)