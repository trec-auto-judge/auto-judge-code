#!/usr/bin/env python3
from trec_auto_judge.click import option_rag_responses
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import mean


@click.command("naive_baseline")
@click.option("--output", type=Path, help="The output file.", required=True)
@option_rag_responses()
def main(rag_responses: list[dict], output: Path):
    """
    A naive rag response assessor that just orders each response by its length.
    """
    ret = []
    run_to_lengths = defaultdict(list)
    for rag_response in tqdm(rag_responses, "Process RAG Responses"):
        metadata = rag_response["metadata"]
        run_id = metadata["run_id"]
        topic_id = metadata["narrative_id"]

        text = " ".join([i["text"] for i in rag_response["answer"]])
        text_length = len(text.split())

        run_to_lengths[run_id].append(text_length)

        ret.append(f"{run_id} LENGTH {topic_id} {text_length}")

    for run_id, text_lengths in run_to_lengths.items():
        ret.append(f"{run_id} LENGTH all {mean(text_lengths)}")

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text("\n".join(ret))


if __name__ == '__main__':
    main()