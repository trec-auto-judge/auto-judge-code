"""
JudgeRunner: Orchestrates AutoJudge execution with nugget lifecycle management.

Consolidates repeated functionality for nugget creation, saving, and judging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Type

from .nugget_data import (
    AnyNuggetBanks,
    NuggetBank,
    NuggetBanks,
    NuggetizerNuggetBank,
    NuggetizerNuggetBanks,
    make_io_functions,
)
from .qrels.qrels import Qrels, verify_qrels, write_qrel_file
from .leaderboard.leaderboard import Leaderboard, verify_leaderboard_topics
from .llm.minima_llm import MinimaLlmConfig
from .report import Report
from .request import Request
from .workflow import NuggetFormat


@dataclass
class JudgeResult:
    """Result of a judge run."""
    leaderboard: Optional[Leaderboard]
    qrels: Optional[Qrels]
    nuggets: Optional[AnyNuggetBanks]


def _get_nugget_models(nugget_format: NuggetFormat) -> Tuple[Type, Type]:
    """Get (BankModel, ContainerModel) tuple for a given nugget format."""
    if nugget_format == NuggetFormat.NUGGETIZER:
        return NuggetizerNuggetBank, NuggetizerNuggetBanks
    return NuggetBank, NuggetBanks


def _write_nuggets(
    nuggets: AnyNuggetBanks,
    path: Path,
    nugget_format: NuggetFormat,
) -> None:
    """Write nuggets using factory-derived writer based on format."""
    bank_model, container_model = _get_nugget_models(nugget_format)
    _, _, write_fn = make_io_functions(bank_model, container_model)
    write_fn(nuggets, path)


def run_judge(
    auto_judge,
    rag_topics: Sequence[Request],
    llm_config: MinimaLlmConfig,
    rag_responses: Optional[Iterable[Report]] = None,
    nugget_banks: Optional[AnyNuggetBanks] = None,
    output_path: Optional[Path] = None,
    store_nuggets_path: Optional[Path] = None,
    create_nuggets: bool = False,
    modify_nuggets: bool = False,
    nugget_format: NuggetFormat = NuggetFormat.AUTOARGUE,
) -> JudgeResult:
    """
    Execute judge workflow with nugget lifecycle management.

    Args:
        auto_judge: AutoJudge implementation
        rag_topics: Topics/queries to evaluate
        llm_config: LLM configuration
        rag_responses: RAG responses to judge (None for nuggify-only)
        nugget_banks: Input nugget banks
        output_path: Leaderboard output path (None for nuggify-only)
        store_nuggets_path: Path to store nuggets
        create_nuggets: If True, call create_nuggets() before judging
        modify_nuggets: If True, judge() may modify nuggets (save after judge)
        nugget_format: Format for writing nuggets (autoargue or nuggetizer)

    Returns:
        JudgeResult with leaderboard, qrels, and final nuggets
    """
    current_nuggets = nugget_banks
    leaderboard = None
    qrels = None

    # Step 1: Create nuggets if requested
    if create_nuggets:
        current_nuggets = auto_judge.create_nuggets(
            rag_topics=rag_topics,
            llm_config=llm_config,
            nugget_banks=nugget_banks,
        )
        # Save immediately for crash recovery
        if current_nuggets is not None and store_nuggets_path:
            _write_nuggets(current_nuggets, store_nuggets_path, nugget_format)

    # Step 2: Judge if we have responses
    if rag_responses is not None:
        leaderboard, qrels, emitted_nuggets = auto_judge.judge(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            llm_config=llm_config,
            nugget_banks=current_nuggets,
        )

        # Step 3: Handle modified nuggets
        if modify_nuggets and emitted_nuggets is not None:
            current_nuggets = emitted_nuggets
            if store_nuggets_path:
                _write_nuggets(emitted_nuggets, store_nuggets_path, nugget_format)

        # Step 4: Write outputs
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
    topic_ids = {t.request_id for t in rag_topics}

    verify_leaderboard_topics(
        expected_topic_ids=topic_ids,
        entries=leaderboard.entries,
        include_all_row=True,
        require_no_extras=True,
    )
    leaderboard.write(output_path)

    if qrels is not None:
        verify_qrels(qrels=qrels, expected_topic_ids=topic_ids, require_no_extras=True)
        write_qrel_file(qrel_out_file=output_path.with_suffix(".qrels"), qrels=qrels)