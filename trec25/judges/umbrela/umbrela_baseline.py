#!/usr/bin/env python3
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import mean
from dataclasses import dataclass, field



from trec_auto_judge.click import option_rag_responses, option_rag_topics
from trec_auto_judge.request import Request
from trec_auto_judge.report import Report
from trec_auto_judge.leaderboard import *
from trec_auto_judge.qrels import *
from trec_auto_judge import *

# crucible
from nuggety.align import evaluator_run, Umbrela
from nuggety.alignment_result import *
import dspy
from typing import *
from pydantic import BaseModel



class UmbrelaAnnotation(BaseModel):
    run_id:str
    query_id:str
    title_query:str
    source_document:str
    problem_statement:str=""
    background:str=""
    confidence:Optional[float] = None
    reasoning:Optional[str] = None
    answerability:Optional[str] = None
    is_match:Optional[bool] = None
    match_score:Optional[float] = None



UMBRELA_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("GRADE", aggregate=mean_of_floats, cast=float),
    MeasureSpec("IS_MATCH", aggregate=mean_of_bools, cast=bool),
))


UMBRELA_QRELS = QrelsSpec["UmbrelaAnnotation"](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: doc_id_md5(r.source_document),
    grade=lambda r: r.match_score,
    on_duplicate="error"
)

class UmbrelaJudge:
    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
    ) -> tuple[Leaderboard, Optional[Qrels]]:
            
        """
        Umbrela response assessor that just orders each response by its length.
        """
        topic_dict = {request.request_id: request for request in rag_topics}

        def prepare_prompts()->List[UmbrelaAnnotation]:
            alignment_input_list = list()
            for rag_response in rag_responses:
                metadata = rag_response["metadata"]
                run_id = metadata["run_id"]
                topic_id = metadata["narrative_id"]

                text = " ".join([i["text"] for i in rag_response["answer"]])  # todo use the Report format

                topic = topic_dict[topic_id]
                if topic is None:
                    raise RuntimeError("Could not identify request object for topic {topic_id}")

                if topic.title is None:
                    raise RuntimeError(f"Missing fields in report request: title {topic.title}, background:{topic.background}, problem_statement: {topic.problem_statement}.")

                problem_statement = topic.problem_statement if topic.problem_statement else f"Identify information that is relevant to the query {topic.title}"
                background = topic.background if topic.background else f"I want to find relevant information on {topic.title}"

                prompt_objs = UmbrelaAnnotation(query_id = topic_id
                                                , run_id = run_id
                                                ,source_document = text
                                                ,metadata= metadata
                                                ,title_query = topic.title
                                                ,background = background
                                                ,problem_statement = problem_statement
                                                )
                alignment_input_list.append(prompt_objs)
            # return (alignment_input_list, None)
            return alignment_input_list


        def umbrela_to_leaderboard(prompt_output):
            b = LeaderboardBuilder(UMBRELA_SPEC)
            b.add_records(
                prompt_output,
                run_id=lambda r: r.run_id,
                topic_id=lambda r: r.query_id,
                get_values=lambda r: {
                    "GRADE": r.match_score,
                    "IS_MATCH": r.is_match,
                },
            )
            
            verify_all(
                measure_names=UMBRELA_SPEC.names,
                entries=b.entries(),                  # staged per-topic entries
                all_topic_id=UMBRELA_SPEC.all_topic_id,
                require_all_row_complete=False,       # no all-rows yet
                require_same_topics_per_run=True,
            )            
            return b.build()

        prompt_input = prepare_prompts()
        print("Debug in", "\n".join(str(p) for p in prompt_input[0:1]))
        
        prompt_output = evaluator_run(prompt=Umbrela, output_converter=Umbrela.convert_output, alignment_input_list=prompt_input)
        print("Debug out", "\n".join(str(p) for p in prompt_input[0:1]))

        leaderboard = umbrela_to_leaderboard(prompt_output=prompt_output)
        qrels = umbrela_to_qrels(  prompt_output, grade_fn=lambda res: res.answerability)
        return (leaderboard, qrels)
# =======

#         def prepare_prompts()->List[UmbrelaAnnotation]:
#             alignment_input_list = list()
#             for rag_response in rag_responses:
#                 metadata = rag_response["metadata"]
#                 run_id = metadata["run_id"]
#                 topic_id = metadata["narrative_id"]

#                 text = " ".join([i["text"] for i in rag_response["answer"]])  # todo use the Report format

#                 topic = topic_dict[topic_id]
#                 if topic is None:
#                     raise RuntimeError("Could not identify request object for topic {topic_id}")

#                 if topic.title is None:
#                     raise RuntimeError(f"Missing fields in report request: title {topic.title}, background:{topic.background}, problem_statement: {topic.problem_statement}.")

#                 problem_statement = topic.problem_statement if topic.problem_statement else f"Identify information that is relevant to the query {topic.title}"
#                 background = topic.background if topic.background else f"I want to find relevant information on {topic.title}"

#                 prompt_objs = UmbrelaAnnotation(query_id = topic_id
#                                                 , run_id = run_id
#                                                 ,source_document = text
#                                                 ,metadata= metadata
#                                                 ,title_query = topic.title
#                                                 ,background = background
#                                                 ,problem_statement = problem_statement
#                                                 )
#                 alignment_input_list.append(prompt_objs)
#             # return (alignment_input_list, None)
#             return alignment_input_list


    #     def umbrela_to_leaderboard(prompt_output: Iterable["UmbrelaAnnotation"], measures) -> Leaderboard:
    #             per_topic_entries = [
    #                 LeaderboardEntry(
    #                     run_id=res.run_id,
    #                     topic_id=res.query_id,
    #                     values={
    #                         "GRADE": float(res.match_score),
    #                         "IS_MATCH": bool(res.is_match),
    #                     },
    #                 )
    #                 for res in prompt_output
    #             ]
    #             return Leaderboard.from_entries_with_all(measures=measures, entries=per_topic_entries)

    #     # prompt_input = prepare_prompts()
    #     # print("Debug in", "\n".join(str(p) for p in prompt_input[0:1]))
        
    #     # prompt_output = evaluator_run(prompt=Umbrela, output_converter=Umbrela.convert_output, alignment_input_list=prompt_input)
    #     # print("Debug out", "\n".join(str(p) for p in prompt_input[0:1]))

    # leaderboard = umbrela_to_leaderboard(prompt_output=prompt_output, measures = MEASURES)
    # qrels = umbrela_to_qrels(  prompt_output, grade_fn=lambda res: res.answerability)
    # return (leaderboard, qrels)

# below here all should move into the TIRA CLI, or be a main class with its own CLI for development

@click.command("umbrela_baseline")
@option_rag_responses()
@option_rag_topics()
@click.option("--output", type=Path, help="The output file.", required=True)
def main(rag_responses: List[Report], rag_topics: List[Request], output:Path):
    qrels_opt=None

    (leaderboard, qrels_opt) = UmbrelaJudge().judge(rag_responses=rag_responses, rag_topics=rag_topics)
    # (leaderboard, qrels_opt) = judge(rag_responses=rag_responses, rag_topics=rag_topics)
    # write_leaderboard(leaderboard=leaderboard, output=output)
    leaderboard.write(output=output)
    if qrels_opt is not None:
        write_qrel_file(qrel_out_file=output.with_suffix(".qrels"), qrels= qrels_opt)

if __name__ == '__main__':
    main()