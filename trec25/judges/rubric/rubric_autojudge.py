#!/usr/bin/env python3
"""
Rubric-based AutoJudge that:
1. Generates nugget questions from query (create_nuggets)
2. Grades how well each response answers each nugget (judge)
3. Derives evaluation score from nugget coverage
"""
from trec_auto_judge import *
from trec_auto_judge.nugget_data import (
    NuggetBank, NuggetBanks, NuggetQuestion
)

import dspy
import asyncio
import re
import json
from typing import *
from pydantic import BaseModel

from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch
from trec_auto_judge import MinimaLlmConfig, OpenAIMinimaLlm


# =============================================================================
# DSPy Signatures
# =============================================================================

class GenerateNuggetQuestions(dspy.Signature):
    """Break a query into concise questions that must be answered."""

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    query_problem: str = dspy.InputField(desc="Problem statement to be addressed")

    questions: list[str] = dspy.OutputField(
        desc="List of concise questions that must be answered to address the query"
    )


class GradeNuggetAnswer(dspy.Signature):
    """
    Grade how well a passage answers a specific question.

    Can the question be answered based on the available context? Choose one:
    - 5: The answer is highly relevant, complete, and accurate.
    - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
    - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
    - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
    - 1: The answer is minimally relevant or complete, with substantial shortcomings.
    - 0: The answer is not relevant or complete at all.
    """

    question: str = dspy.InputField(desc="The question to be answered")
    passage: str = dspy.InputField(desc="The passage that may contain the answer")

    grade: Literal["0", "1", "2", "3", "4", "5"] = dspy.OutputField(
        desc="Grade from 0-5 indicating how well the passage answers the question"
    )
    reasoning: Optional[str] = dspy.OutputField(
        desc="Brief explanation of the grade", default=None, required=False
    )


# =============================================================================
# Data Models (combined input/output)
# =============================================================================

class NuggetGenerationData(BaseModel):
    """Combined input/output for nugget question generation."""
    # Input fields
    query_id: str
    query_title: str
    query_background: str
    query_problem: str
    # Output fields (populated by LLM)
    questions: List[str] = []


class NuggetGradeData(BaseModel):
    """Combined input/output for grading a nugget against a passage."""
    # Input fields
    run_id: str
    query_id: str
    nugget_id: str
    question: str
    passage: str
    # Output fields (populated by LLM)
    grade: int = 0
    reasoning: Optional[str] = None


# =============================================================================
# Leaderboard & Qrels Specs
# =============================================================================

RUBRIC_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("NUGGET_COVERAGE", aggregate=mean_of_floats, cast=float),
    MeasureSpec("AVG_GRADE", aggregate=mean_of_floats, cast=float),
    MeasureSpec("COVERED_COUNT", aggregate=mean_of_floats, cast=float),
))


RUBRIC_QRELS = QrelsSpec["NuggetGradeData"](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: doc_id_md5(r.passage),
    grade=lambda r: float(r.grade),
    on_duplicate="keep_max"
)


# =============================================================================
# Conversion Functions
# =============================================================================

def _parse_grade(s: str) -> int:
    """Extract grade 0-5 from string."""
    m = re.search(r'\b([0-5])\b', s)
    if not m:
        return 0  # Default to 0 if no valid grade found
    return int(m.group(1))


# =============================================================================
# RubricJudge Implementation
# =============================================================================

class RubricJudge(AutoJudge):
    """
    Rubric-based judge that:
    1. Generates nugget questions from topics
    2. Grades responses against each nugget
    3. Computes coverage score based on grade threshold
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks
    
    def __init__(self, grade_threshold: int = 3):
        """
        Args:
            grade_threshold: Minimum grade to consider a nugget "covered" (default: 3)
        """
        self.grade_threshold = grade_threshold

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanks] = None,
        **kwargs
    ) -> Optional[NuggetBanks]:
        """Generate nugget questions for each topic using LLM."""

        # Prepare generation data
        gen_data = [
            NuggetGenerationData(
                query_id=topic.request_id,
                query_title=topic.title or "",
                query_background=topic.background or "",
                query_problem=topic.problem_statement or ""
            )
            for topic in rag_topics
        ]

        # Convert output handler
        def convert_gen_output(prediction: dspy.Prediction, data: NuggetGenerationData) -> None:
            questions = prediction.questions if hasattr(prediction, 'questions') else []
            # print("questions debug 1", type(questions), questions)
            # DSPy may return list as JSON string - parse it
            if isinstance(questions, str):
                try:
                    parsed = json.loads(questions)
                    if isinstance(parsed, list):
                        questions = [str(q).strip() for q in parsed if q]
                    else:
                        # Fallback: split by newlines
                        questions = [q.strip() for q in questions.split('\n') if q.strip()]
                except json.JSONDecodeError:
                    # Fallback: split by newlines
                    questions = [q.strip() for q in questions.split('\n') if q.strip()]
            # print("questions debug 2", type(questions), questions)
            # print("questions debug 3",  questions[0])
            # print("questions", questions)
            data.questions = questions

        # Run LLM generation
        print(f"Rubric: Generating questions...")
        gen_data = asyncio.run(run_dspy_batch(
            GenerateNuggetQuestions,
            gen_data,
            convert_gen_output,
            backend=OpenAIMinimaLlm(llm_config)
            
        ))
        print(f"Rubric: Finished gnerating questions")

        # Build NuggetBanks from generated questions
        banks = []
        for data in gen_data:
            bank = NuggetBank(
                query_id=data.query_id,
                title_query=data.query_title
            )

            nuggets = [
                NuggetQuestion(
                    query_id=data.query_id,
                    question=question_text,
                    question_id=f"{data.query_id}-q{i}"
                )
                for i, question_text in enumerate(data.questions)
            ]

            bank.add_nuggets(nuggets)
            bank.index_nuggets()
            banks.append(bank)

        return NuggetBanks.from_banks_list(banks)

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanks] = None,
        **kwargs
    ) -> tuple[Leaderboard, Optional[Qrels]]:
        """
        Grade each response against all nuggets for its topic.

        Stores per-nugget grades in Report.evaldata with format:
        {
            "nugget_grades": {
                "<nugget_id>": {"grade": int, "reasoning": str},
                ...
            },
            "coverage_score": float,
            "avg_grade": float,
            "covered_count": int,
            "total_nuggets": int
        }
        """
        if nugget_banks is None:
            raise ValueError("RubricJudge requires nugget_banks. Run create_nuggets first or provide --nugget-banks.")

        # Prepare grading data (one per response-nugget pair)
        grade_data: List[NuggetGradeData] = []
        response_nugget_map: Dict[str, List[NuggetGradeData]] = {}  # run_id:topic_id -> data list

        for response in rag_responses:
            metadata = response.metadata
            run_id = metadata.run_id
            topic_id = metadata.topic_id
            text = response.get_report_text()

            bank = nugget_banks.banks.get(topic_id)
            if bank is None:
                print(f"Warning: No nugget bank for topic {topic_id}, skipping")
                continue

            response_key = f"{run_id}:{topic_id}"
            response_nugget_map[response_key] = []

            # Create grade data for each nugget question
            for nugget in bank.nuggets_as_list():
                if isinstance(nugget, NuggetQuestion):
                    data = NuggetGradeData(
                        run_id=run_id,
                        query_id=topic_id,
                        nugget_id=nugget.question_id or nugget.question,
                        question=nugget.question,
                        passage=text
                    )
                    grade_data.append(data)
                    response_nugget_map[response_key].append(data)

        # Convert output handler
        def convert_grade_output(prediction: dspy.Prediction, data: NuggetGradeData) -> None:
            data.grade = _parse_grade(prediction.grade)
            data.reasoning = getattr(prediction, 'reasoning', None)

        # Run LLM grading
        print(f"Rubric: Grading responses...")
        if grade_data:
            grade_data = asyncio.run(run_dspy_batch(
                GradeNuggetAnswer,
                grade_data,
                convert_grade_output,
                backend=OpenAIMinimaLlm(llm_config)
            ))
        print(f"Rubric: Finished grading")


        # Aggregate grades per response and store in evaldata
        response_grades: Dict[str, Dict[str, Any]] = {}  # response_key -> evaldata

        for data in grade_data:
            response_key = f"{data.run_id}:{data.query_id}"
            if response_key not in response_grades:
                response_grades[response_key] = {
                    "nugget_grades": {},
                    "grades_list": []
                }

            response_grades[response_key]["nugget_grades"][data.nugget_id] = {
                "grade": data.grade,
                "reasoning": data.reasoning
            }
            response_grades[response_key]["grades_list"].append(data.grade)

        # Compute coverage scores
        for response_key, evaldata in response_grades.items():
            grades = evaldata["grades_list"]
            if grades:
                covered = sum(1 for g in grades if g >= self.grade_threshold)
                evaldata["coverage_score"] = covered / len(grades)
                evaldata["avg_grade"] = sum(grades) / len(grades)
                evaldata["covered_count"] = covered
                evaldata["total_nuggets"] = len(grades)
            else:
                evaldata["coverage_score"] = 0.0
                evaldata["avg_grade"] = 0.0
                evaldata["covered_count"] = 0
                evaldata["total_nuggets"] = 0
            del evaldata["grades_list"]  # Remove temporary field

        # Update Report.evaldata
        for response in rag_responses:
            response_key = f"{response.metadata.run_id}:{response.metadata.topic_id}"
            if response_key in response_grades:
                response.evaldata = response_grades[response_key]

        # Build leaderboard
        leaderboard = self._build_leaderboard(response_grades)

        # Build qrels from grade data
        qrels = build_qrels(records=grade_data, spec=RUBRIC_QRELS) if grade_data else None

        return (leaderboard, qrels)

    def _build_leaderboard(self, response_grades: Dict[str, Dict[str, Any]]) -> Leaderboard:
        """Build leaderboard from aggregated response grades."""
        b = LeaderboardBuilder(RUBRIC_SPEC)

        for response_key, evaldata in response_grades.items():
            run_id, topic_id = response_key.split(":", 1)
            b.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "NUGGET_COVERAGE": evaldata["coverage_score"],
                    "AVG_GRADE": evaldata["avg_grade"],
                    "COVERED_COUNT": float(evaldata["covered_count"]),
                }
            )

        leaderboard = b.build()
        LeaderboardVerification(leaderboard).complete_measures(include_all_row=False).same_topics_per_run()
        return leaderboard


if __name__ == '__main__':
    auto_judge_to_click_command(RubricJudge(), "rubric_autojudge")()