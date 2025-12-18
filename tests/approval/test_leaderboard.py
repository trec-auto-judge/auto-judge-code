import unittest

# from trec_auto_judge import Leaderboard, LeaderboardEntry
from trec_auto_judge import *
from tempfile import TemporaryDirectory
from pathlib import Path
# from approvaltests import verify_file

class TestLeaderboard(unittest.TestCase):
    def test_valid_leaderboard(self):
        
        MY_SPEC = LeaderboardSpec(measures=(
            MeasureSpec("measure-01", aggregate=mean_of_floats, cast=float),
        ))
        b = LeaderboardBuilder(MY_SPEC)

        b.add(run_id="run-01",topic_id="topic-01",values={"measure-01": 2.0})
        b.add(run_id="run-01",topic_id="topic-02",values={"measure-01": 0.0})
    
        l = b.build()
        
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