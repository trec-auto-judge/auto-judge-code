import click
from pathlib import Path


@click.option(
    "--inputs",
    type=Path,
    required=True,
    help="The to-be-evaluated leaderboards congruent to 'trec_eval -q' format.",
)
@click.option(
    "--truths",
    type=Path,
    required=True,
    help="The truth leaderboards congruent to 'trec_eval -q' format.",
)
def evaluate(inputs: Path, truths: Path) -> int:
    """Evaluate the input leaderboards against the ground-truth leaderboards."""
    print("ToDo implement this")
    return 1
