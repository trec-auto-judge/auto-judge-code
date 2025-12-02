import click
from pathlib import Path
import pandas as pd
from ..evaluation import TrecLeaderboardEvaluation
from typing import Any, List, Optional
from tira.io_utils import to_prototext


def persist_output(df: pd.DataFrame, output: Path) -> None:
    if output.name.endswith(".jsonl"):
        df.to_json(output, lines=True, orient="records")
    elif output.name.endswith(".prototext"):
        ret = {k: v for k, v in df.iloc[0].to_dict().items()}
        ret = to_prototext([ret])
        output.write_text(ret)
    else:
        raise ValueError(f"Can not handle output file format {output}")

@click.option(
    "--truth-leaderboard",
    type=Path,
    required=False,
    help="The ground truth leaderboards congruent to 'trec_eval -q' format.",
)
@click.option(
    "--truth-metric",
    type=Path,
    required=False,
    help="The metric from the ground truth leaderboard that .",
)
@click.option(
    "--input",
    type=Path,
    required=True,
    multiple=True,
    help="The to-be-evaluated leaderboard(s) congruent to 'trec_eval -q' format.",
)
@click.option(
    "--output",
    type=Path,
    required=False,
    help="The file where the evaluation should be persisted.",
)
@click.option(
    "--aggregate",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Should only aggregates scores be reported.",
)
def evaluate(truth_leaderboard: Optional[Path], truth_metric: Optional[str], input: List[Path], output: Path, aggregate: bool) -> int:
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

    df = pd.DataFrame(df)

    if aggregate:
        df_aggr = {"Judges": len(df)}
        for k in df.columns:
            if k in ("Judge", "Metric"):
                continue
            df_aggr[k] = df[k].mean()
        df = pd.DataFrame([df_aggr])

    print(df.to_string(index=False))

    if output:
        persist_output(df, output)

    return 0