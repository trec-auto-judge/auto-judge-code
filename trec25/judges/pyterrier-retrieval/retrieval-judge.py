#!/usr/bin/env python3
from trec_auto_judge import Report, LeaderboardSpec, LeaderboardBuilder, LeaderboardVerification, mean_of_floats, MeasureSpec, AutoJudge, auto_judge_to_click_command, Leaderboard, Qrels, Sequence, Request, Optional, MinimaLlmConfig, NuggetBanks
from collections import defaultdict
from tqdm import tqdm
from tira.third_party_integrations import ensure_pyterrier_is_loaded
import pyterrier as pt


def group_by_topic_id(rag_responses: list[Report]) -> float:
    ret = defaultdict(dict)
    for rag_response in rag_responses:
        run_id = rag_response.metadata.run_id
        topic_id = rag_response.metadata.topic_id
        ret[topic_id][run_id] = rag_response.get_report_text()
    return ret


# Some semi-random selected weighting models from http://terrier.org/docs/v4.2/javadoc/org/terrier/matching/models/WeightingModel.html
LEADERBOARD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("BM25", aggregate=mean_of_floats, cast=float),
    MeasureSpec("DirichletLM", aggregate=mean_of_floats, cast=float),
    MeasureSpec("Hiemstra_LM", aggregate=mean_of_floats, cast=float),
    MeasureSpec("DFIC", aggregate=mean_of_floats, cast=float),
    MeasureSpec("DPH", aggregate=mean_of_floats, cast=float),
    MeasureSpec("DLH", aggregate=mean_of_floats, cast=float),
    MeasureSpec("Tf", aggregate=mean_of_floats, cast=float),
    MeasureSpec("TF_IDF", aggregate=mean_of_floats, cast=float),
    MeasureSpec("PL2", aggregate=mean_of_floats, cast=float),
    MeasureSpec("InL2", aggregate=mean_of_floats, cast=float),
))


class RetrievalJudge(AutoJudge):
    def create_nuggets(
        self,
        rag_responses: Sequence["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional["NuggetBanks"] = None,
        **kwargs
    ) -> Optional["NuggetBanks"]:
        return None
            
    
    def judge(self, rag_responses: Sequence["Report"], rag_topics: Sequence["Request"], llm_config:MinimaLlmConfig, nugget_banks: Optional[NuggetBanks] = None, **kwargs) -> tuple["Leaderboard", Optional["Qrels"]]:
        ensure_pyterrier_is_loaded()
        tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
        def pt_tokenize(text):
            return ' '.join(tokeniser.getTokens(text))

        topic_id_to_title = {i.request_id: pt_tokenize(i.title) for i in rag_topics}
        topic_id_to_responses = group_by_topic_id(rag_responses)
        all_systems = set([i.metadata.run_id for i in rag_responses])

        ret = LeaderboardBuilder(LEADERBOARD_SPEC)

        for topic in tqdm(topic_id_to_responses.keys(), "Process Topics"):
            query_text = topic_id_to_title[topic]
            docs = [{"docno": system, "text": system_response} for system, system_response in topic_id_to_responses[topic].items()]
            system_to_wmodel_to_score = defaultdict(lambda: defaultdict(lambda: 1000))
            index = pt.IterDictIndexer("/not-needed/for-memory-index", meta={'docno' : 100}, type=pt.IndexingType.MEMORY).index(docs)

            for wmodel in LEADERBOARD_SPEC.measures:
                retriever = pt.terrier.Retriever(index, wmodel=wmodel.name)
                rtr = retriever.search(query_text)
                run_id_to_score = defaultdict(lambda: 0)
                for _, i in rtr.iterrows():
                    run_id_to_score[i["docno"]] = max(0, 1000 - i["rank"])
                
                for system in all_systems:
                    system_to_wmodel_to_score[system][wmodel.name] = run_id_to_score.get(system)
            
            for system in all_systems:
                ret.add(run_id=system, topic_id=topic, values=system_to_wmodel_to_score[system])

        leaderboard = ret.build()
        LeaderboardVerification(leaderboard, on_missing="ignore").all()
        return leaderboard, None


if __name__ == '__main__':
    auto_judge_to_click_command(RetrievalJudge(), "retrieval-judge")()