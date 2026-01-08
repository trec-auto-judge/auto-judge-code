#!/usr/bin/env python3
from trec_auto_judge import AutoJudge, Sequence, Optional, Report, Request, LeaderboardSpec, LeaderboardBuilder, LeaderboardVerification, mean_of_floats, MeasureSpec, auto_judge_to_click_command, Leaderboard, Qrels, MinimaLlmConfig, NuggetBanks
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import mean
import random


def rand(seed: str) -> float:
    random.seed(seed)
    return random.random()


NAIVE_LEADERBOARD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("LENGTH", aggregate=mean_of_floats, cast=float),
    MeasureSpec("RANDOM", aggregate=mean_of_floats, cast=float),
))


class NaiveJudge(AutoJudge):
    def create_nuggets(
        self,
        rag_responses: Sequence["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional["NuggetBanks"] = None,
        **kwargs
    ) -> Optional["NuggetBanks"]:
        return None

    def judge(self, rag_responses: Sequence["Report"]
              , rag_topics: Sequence["Request"]
              , llm_config:MinimaLlmConfig
              , nugget_banks: Optional[NuggetBanks] = None
              , **kwargs) -> tuple["Leaderboard", Optional["Qrels"]]:
        ret = LeaderboardBuilder(NAIVE_LEADERBOARD_SPEC)

        for rag_response in tqdm(rag_responses, "Process RAG Responses"):
            vals = {
                "LENGTH": len(rag_response.get_report_text().split()),
                "RANDOM": rand(rag_response.metadata.run_id + rag_response.metadata.topic_id)
            }
            ret.add(run_id=rag_response.metadata.run_id, topic_id=rag_response.metadata.topic_id, values=vals)

        leaderboard = ret.build()
        LeaderboardVerification(leaderboard, on_missing="ignore").all()
        return leaderboard, None


if __name__ == '__main__':
    auto_judge_to_click_command(NaiveJudge(), "naive-judge")()
