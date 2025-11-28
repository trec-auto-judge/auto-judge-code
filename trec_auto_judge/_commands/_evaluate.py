import click
from pathlib import Path
import pandas as pd
from ..evaluation import TrecLeaderboardEvaluation
from typing import List


@click.option(
    "--truth-leaderboard",
    type=Path,
    required=True,
    help="The ground truth leaderboards congruent to 'trec_eval -q' format.",
)
@click.option(
    "--truth-metric",
    type=Path,
    required=True,
    help="The metric from the ground truth leaderboard that .",
)
@click.option(
    "--input",
    type=Path,
    required=True,
    multiple=True,
    help="The to-be-evaluated leaderboard(s) congruent to 'trec_eval -q' format.",
)
def evaluate(truth_leaderboard: Path, truth_metric: str, input: List[Path]) -> int:
    """Evaluate the input leaderboards against the ground-truth leaderboards."""
    te = TrecLeaderboardEvaluation(truth_leaderboard, truth_metric)
    df = []

    for c in input:
        result = te.evaluate(c)
        
        for i in result:
            tmp = {"Judge": c.name, "Metric": i}
            for k, v in result[i].items():
                tmp[k] = v
            df.append(tmp)

    print(pd.DataFrame(df).to_string(index=False))

    return 0