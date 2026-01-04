"""
JudgeRunner: Orchestrates AutoJudge execution with nugget lifecycle management.

Consolidates repeated functionality for nugget creation, saving, and judging.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .nugget_data import (
    NuggetBanksProtocol,
    NuggetBanksVerification,
    write_nugget_banks_generic,
    load_nugget_banks_generic,
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


# Framework-consumed settings (extracted before passing to AutoJudge)
FRAMEWORK_SETTINGS = frozenset({"llm_model"})


def _strip_framework_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Remove framework-consumed settings before passing to AutoJudge."""
    return {k: v for k, v in settings.items() if k not in FRAMEWORK_SETTINGS}


def _resolve_nugget_file_path(filebase: Path) -> Path:
    """
    Resolve nugget file path from filebase.

    If filebase already has a recognized extension (.jsonl, .json), use as-is.
    Otherwise, append .nuggets.jsonl extension.
    """
    if filebase.suffix in (".jsonl", ".json"):
        return filebase
    return filebase.parent / f"{filebase.name}.nuggets.jsonl"


def _resolve_judgment_file_paths(filebase: Path) -> tuple[Path, Path]:
    """
    Resolve leaderboard and qrels file paths from filebase.

    Returns:
        Tuple of (leaderboard_path, qrels_path):
        - {filebase}.judgment.json
        - {filebase}.judgment.qrels
    """
    if filebase.suffix in (".json",):
        # Already has extension, derive qrels from it
        return filebase, filebase.with_suffix(".qrels")
    return (
        filebase.parent / f"{filebase.name}.judgment.json",
        filebase.parent / f"{filebase.name}.judgment.qrels",
    )


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
    # Settings dicts passed to AutoJudge methods as **kwargs
    settings: Optional[dict[str, Any]] = None,
    nugget_settings: Optional[dict[str, Any]] = None,
    judge_settings: Optional[dict[str, Any]] = None,
    # Lifecycle flags
    force_recreate_nuggets: bool = False,
    nugget_depends_on_responses: bool = True,
    judge_uses_nuggets: bool = True,
    # NuggetBanks type for loading (dotted import path)
    nugget_banks_type: Optional[str] = None,
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
        settings: Shared settings dict passed to both phases (fallback)
        nugget_settings: Settings dict passed to create_nuggets() (overrides settings)
        judge_settings: Settings dict passed to judge() (overrides settings)
        force_recreate_nuggets: If True, recreate even if file exists
        nugget_depends_on_responses: If True, pass responses to create_nuggets()
        judge_uses_nuggets: If True, pass nuggets to judge()
        nugget_banks_type: Dotted import path for NuggetBanks class (for loading)

    Returns:
        JudgeResult with leaderboard, qrels, and final nuggets
    """
    # Extract framework-consumed settings
    effective_llm_config = llm_config
    if settings and "llm_model" in settings:
        effective_llm_config = llm_config.with_model(settings["llm_model"])
        print(f"[judge_runner] Model override from settings: {effective_llm_config.model}", file=sys.stderr)

    current_nuggets = nugget_banks
    leaderboard = None
    qrels = None

    # Resolve nugget file path (add .nuggets.jsonl extension if needed)
    nugget_file_path = _resolve_nugget_file_path(store_nuggets_path) if store_nuggets_path else None

    # Step 1: Resolve nuggets (auto-load or create)
    if do_create_nuggets:
        # Check if we can auto-load existing nuggets
        if nugget_file_path and nugget_file_path.exists() and not force_recreate_nuggets:
            # Auto-load existing nuggets
            print(f"[judge_runner] Loading existing nuggets: {nugget_file_path}", file=sys.stderr)
            if nugget_banks_type:
                current_nuggets = load_nugget_banks_generic(nugget_file_path, nugget_banks_type)
            else:
                # Try to get type from auto_judge
                nbt = getattr(auto_judge, "nugget_banks_type", None)
                if nbt:
                    current_nuggets = load_nugget_banks_generic(nugget_file_path, nbt)
                else:
                    raise ValueError(
                        f"Cannot load nuggets: no nugget_banks_type specified. "
                        f"Provide nugget_banks_type or use --force-recreate-nuggets."
                    )
        else:
            # Create nuggets
            nugget_kwargs = _strip_framework_settings(nugget_settings or settings or {})
            if nugget_kwargs:
                print(f"[judge_runner] create_nuggets settings: {nugget_kwargs}", file=sys.stderr)

            # Pass responses based on nugget_depends_on_responses flag
            responses_for_nuggets = rag_responses if nugget_depends_on_responses else None

            current_nuggets = auto_judge.create_nuggets(
                rag_responses=responses_for_nuggets,
                rag_topics=rag_topics,
                llm_config=effective_llm_config,
                nugget_banks=nugget_banks,
                **nugget_kwargs,
            )
            # Verify created nuggets
            if current_nuggets is not None:
                NuggetBanksVerification(current_nuggets, rag_topics).all()
                # Save immediately for crash recovery
                if nugget_file_path:
                    write_nugget_banks_generic(current_nuggets, nugget_file_path)
                    print(f"[judge_runner] Nuggets saved to: {nugget_file_path}", file=sys.stderr)
    elif not do_create_nuggets and nugget_file_path and nugget_file_path.exists():
        # do_create_nuggets=False but file exists - load it
        print(f"[judge_runner] Loading nuggets (create_nuggets=false): {nugget_file_path}", file=sys.stderr)
        nbt = nugget_banks_type or getattr(auto_judge, "nugget_banks_type", None)
        if nbt:
            current_nuggets = load_nugget_banks_generic(nugget_file_path, nbt)
        # If no type available and no nugget_banks passed, current_nuggets stays as nugget_banks

    # Step 2: Judge if requested
    if do_judge:
        judge_kwargs = _strip_framework_settings(judge_settings or settings or {})
        if judge_kwargs:
            print(f"[judge_runner] judge settings: {judge_kwargs}", file=sys.stderr)

        # Pass nuggets based on judge_uses_nuggets flag
        nuggets_for_judge = current_nuggets if judge_uses_nuggets else None

        leaderboard, qrels = auto_judge.judge(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            llm_config=effective_llm_config,
            nugget_banks=nuggets_for_judge,
            **judge_kwargs,
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

    # Resolve output paths from filebase
    leaderboard_path, qrels_path = _resolve_judgment_file_paths(output_path)

    LeaderboardVerification(leaderboard, expected_topic_ids=topic_ids).all()
    leaderboard.write(leaderboard_path)
    print(f"[judge_runner] Leaderboard saved to: {leaderboard_path}", file=sys.stderr)

    if qrels is not None:
        QrelsVerification(qrels, expected_topic_ids=topic_ids).all()
        write_qrel_file(qrel_out_file=qrels_path, qrels=qrels)
        print(f"[judge_runner] Qrels saved to: {qrels_path}", file=sys.stderr)