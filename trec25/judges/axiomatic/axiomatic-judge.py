#!/usr/bin/env python3
from trec_auto_judge.click import option_rag_responses
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import mean
from ir_axioms.axiom.generation.coherence import WordLengthDeviationCoherenceAxiom, SubjectVerbClosenessCoherenceAxiom
from ir_axioms.model.generation import GenerationResponse
from ir_axioms.utils.lazy import lazy_inject

class COH1():
    def __init__(self):
        self.coh1_axiom = lazy_inject(WordLengthDeviationCoherenceAxiom)()

    def calculate(self, o: GenerationResponse):
        return self.coh1_axiom.average_word_lengths_stdev(o)

class COH2():
    def __init__(self):
        self.coh2_axiom = lazy_inject(SubjectVerbClosenessCoherenceAxiom)()

    def calculate(self, o: GenerationResponse):
        return self.coh2_axiom.avg_max_sv_distance(o)


@click.command("naive_baseline")
@click.option("--output", type=Path, help="The output file.", required=True)
@option_rag_responses()
def main(rag_responses: list[dict], output: Path):
    """
    A naive rag response assessor that just orders each response by its length.
    """
    measures = {
        "COH1": COH1(),
        "COH2": COH2(),
    }
    vals = {k: defaultdict(list) for k in measures}
    runs = set()

    ret = []

    for rag_response in tqdm(rag_responses, "Process RAG Responses"):
        metadata = rag_response["metadata"]
        run_id = metadata["run_id"]
        topic_id = metadata["narrative_id"]
        runs.add(run_id)
        text = " ".join([i["text"] for i in rag_response["answer"]])
        gen_output = GenerationResponse(id=f"{run_id}-{topic_id}", text=text)

        for m in measures:
            val = measures[m].calculate(gen_output)
            ret.append(f"{run_id} {m} {topic_id} {val}")
            vals[m][run_id].append(val)

    for run_id in runs:
        for m in measures:
            ret.append(f"{run_id} {m} all {mean(vals[m][run_id])}")

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text("\n".join(ret))


if __name__ == '__main__':
    main()