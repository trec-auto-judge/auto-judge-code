import unittest
from typing import Type
from trec_auto_judge import AutoJudge, Sequence, Report, Request, Leaderboard, Optional, Qrels, MeasureSpec, LeaderboardSpec, LeaderboardBuilder, NuggetBanks
from trec_auto_judge import auto_judge_to_click_command, mean_of_floats
from trec_auto_judge.nugget_data import NuggetBanksProtocol
from trec_auto_judge.llm import MinimaLlmConfig
from click.testing import CliRunner
from . import TREC_25_DATA
from pathlib import Path
from tempfile import TemporaryDirectory

class NaiveJudge(AutoJudge):
    """A simple judge that uses NuggetBanks format."""
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional["NuggetBanks"] = None,
        **kwargs
    ) -> Optional["NuggetBanks"]:
        return None

    def judge(self, rag_responses: Sequence["Report"], rag_topics: Sequence["Request"],  llm_config: MinimaLlmConfig, nugget_banks: Optional[NuggetBanks] = None, **kwargs) -> tuple["Leaderboard", Optional["Qrels"]]:
        ret = LeaderboardBuilder(self.leaderboard_spec())

        for r in rag_responses:
            ret.add(
                topic_id=r.metadata.topic_id,
                run_id=r.metadata.run_id,
                values={"measure-01": 1}
            )

        return ret.build(), None

    def leaderboard_spec(self) -> LeaderboardSpec:
        return LeaderboardSpec(measures=(
            MeasureSpec("measure-01", aggregate=mean_of_floats, cast=float),
        ))

class TestAutoJudgeProtocoll(unittest.TestCase):
    def test_minimal_auto_judge(self):
        cmd = auto_judge_to_click_command(NaiveJudge(), "my-command")

        runner = CliRunner()

        with TemporaryDirectory() as tmp_dir:
            target_file = Path(tmp_dir) / "leaderboard.trec"
            result = runner.invoke(cmd, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs", "--output", str(target_file)])

            print(result.output)
            print(result.exception)
            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(target_file.is_file())
            actual_leaderboard = target_file.read_text()
            self.assertIn("measure-01\t28\t1", actual_leaderboard)
            self.assertIn("measure-01\tall\t1.0", actual_leaderboard)

