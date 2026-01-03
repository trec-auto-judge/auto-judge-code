"""
JudgeRunner: Orchestrates AutoJudge execution with nugget lifecycle management.

Consolidates repeated functionality for nugget creation, saving, and judging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .nugget_data import (
    NuggetBanksProtocol,
    NuggetBanksVerification,
    write_nugget_banks_generic,
)
from .qrels.qrels import Qrels, QrelsVerification, write_qrel_file
from .leaderboard.leaderboard import Leaderboard, LeaderboardVerification
from .llm.minima_llm import MinimaLlmConfig
from .report import Report
from .request import Request


@dataclass
class JudgeResult:
    """Result of a judge run."""
    leaderboard: Optional[Leaderboard]
    qrels: Optional[Qrels]
    nuggets: Optional[NuggetBanksProtocol]


def run_judge(
    auto_judge,
    rag_responses: Iterable[Report],
    rag_topics: Sequence[Request],
    llm_config: MinimaLlmConfig,
    nugget_banks: Optional[NuggetBanksProtocol] = None,
    output_path: Optional[Path] = None,
    store_nuggets_path: Optional[Path] = None,
    do_create_nuggets: bool = False,
    do_judge: bool = True,
) -> JudgeResult:
    """
    Execute judge workflow with nugget lifecycle management.

    Args:
        auto_judge: AutoJudge implementation
        rag_responses: RAG responses to evaluate
        rag_topics: Topics/queries to evaluate
        llm_config: LLM configuration
        nugget_banks: Input nugget banks (any NuggetBanksProtocol implementation)
        output_path: Leaderboard output path
        store_nuggets_path: Path to store created/refined nuggets
        do_create_nuggets: If True, call create_nuggets()
        do_judge: If True, call judge()

    Returns:
        JudgeResult with leaderboard, qrels, and final nuggets
    """
    current_nuggets = nugget_banks
    leaderboard = None
    qrels = None

    # Step 1: Create/refine nuggets if requested
    if do_create_nuggets:
        current_nuggets = auto_judge.create_nuggets(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            llm_config=llm_config,
            nugget_banks=nugget_banks,
        )
        # Verify created nuggets
        if current_nuggets is not None:
            NuggetBanksVerification(current_nuggets, rag_topics).all()
            # Save immediately for crash recovery
            if store_nuggets_path:
                write_nugget_banks_generic(current_nuggets, store_nuggets_path)

    # Step 2: Judge if requested
    if do_judge:
        leaderboard, qrels = auto_judge.judge(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            llm_config=llm_config,
            nugget_banks=current_nuggets,
        )

        # Step 3: Write outputs
        if output_path:
            _write_outputs(
                leaderboard=leaderboard,
                qrels=qrels,
                rag_topics=rag_topics,
                output_path=output_path,
            )

    return JudgeResult(
        leaderboard=leaderboard,
        qrels=qrels,
        nuggets=current_nuggets,
    )


def _write_outputs(
    leaderboard: Leaderboard,
    qrels: Optional[Qrels],
    rag_topics: Sequence[Request],
    output_path: Path,
) -> None:
    """Verify and write leaderboard and qrels."""
    topic_ids = [t.request_id for t in rag_topics]

    LeaderboardVerification(leaderboard, expected_topic_ids=topic_ids).all()
    leaderboard.write(output_path)

    if qrels is not None:
        QrelsVerification(qrels, expected_topic_ids=topic_ids).all()
        write_qrel_file(qrel_out_file=output_path.with_suffix(".qrels"), qrels=qrels)