#!/usr/bin/env python3
from trec_auto_judge.click import option_rag_responses, option_rag_topics
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import median
import pandas as pd

import pyterrier as pt

def group_by_topic_id(rag_responses: list[dict]) -> float:
    ret = defaultdict(dict)
    for rag_response in rag_responses:
        metadata = rag_response["metadata"]
        run_id = metadata["run_id"]
        topic_id = metadata["narrative_id"]
        ret[topic_id][run_id] = " ".join([i["text"] for i in rag_response["answer"]])
    return ret

# Some semi-random selected weighting models from http://terrier.org/docs/v4.2/javadoc/org/terrier/matching/models/WeightingModel.html
WEIGHTING_MODELS = ["BM25", "DirichletLM", "Hiemstra_LM", "DFIC", "DPH", "DLH", "Tf", "TF_IDF", "PL2", "InL2"]


@click.command("retrieval-judge")
@click.option("--output", type=Path, help="The output file.", required=True)
@option_rag_responses()
@option_rag_topics()
def main(rag_responses: list[dict], rag_topics: list, output: Path):
    """
    fooo.
    """
    pt.java.init()
    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    def pt_tokenize(text):
        return ' '.join(tokeniser.getTokens(text))

    topic_id_to_title = {i.request_id: pt_tokenize(i.title) for i in rag_topics}
    topic_id_to_responses = group_by_topic_id(rag_responses)
    all_systems = set([i["metadata"]["run_id"] for i in rag_responses])

    ret = []
    
    system_to_wmodel_to_scores = defaultdict(lambda: defaultdict(list))
    for topic in tqdm(topic_id_to_responses.keys(), "Process Topics"):
        query_text = topic_id_to_title[topic]
        docs = [{"docno": system, "text": system_response} for system, system_response in topic_id_to_responses[topic].items()]
        index = pt.IterDictIndexer("/not-needed/for-memory-index", meta={'docno' : 100}, type=pt.IndexingType.MEMORY).index(docs)

        for wmodel in WEIGHTING_MODELS:
            retriever = pt.terrier.Retriever(index, wmodel=wmodel)
            rtr = retriever.search(query_text)
            run_id_to_score = defaultdict(lambda: 0)
            for _, i in rtr.iterrows():
                run_id_to_score[i["docno"]] = max(0, 1000 - i["rank"])
            
            for system in all_systems:
                system_to_wmodel_to_scores[system][wmodel].append(run_id_to_score.get(system))
                ret.append(f"{system} {wmodel} {topic} {run_id_to_score.get(system)}")

    for run_id in all_systems:
        for wmodel in WEIGHTING_MODELS:
            ret.append(f"{run_id} {wmodel} all {median(system_to_wmodel_to_scores[run_id][wmodel])}")

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text("\n".join(ret))


if __name__ == '__main__':
    main()