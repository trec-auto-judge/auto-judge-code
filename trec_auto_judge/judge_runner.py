"""
JudgeRunner: Orchestrates AutoJudge execution with nugget lifecycle management.

Consolidates repeated functionality for nugget creation, saving, and judging.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml

from .utils import get_git_info
from .nugget_data import (
    NuggetBanksProtocol,
    write_nugget_banks_generic,
)
from .qrels.qrels import Qrels, write_qrel_file
from .leaderboard.leaderboard import Leaderboard
from .llm.minima_llm import MinimaLlmConfig
from .report import Report
from .request import Request
from .workflow.paths import (
    resolve_nugget_file_path,
    resolve_judgment_file_paths,
    resolve_config_file_path,
    load_nugget_banks_from_path,
)


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
    nugget_banks_path: Optional[Path] = None,
    judge_output_path: Optional[Path] = None,
    nugget_output_path: Optional[Path] = None,
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
    # Configuration name for reproducibility tracking
    config_name: str = "default",
) -> JudgeResult:
    """
    Execute judge workflow with nugget lifecycle management.

    Args:
        auto_judge: AutoJudge implementation
        rag_responses: RAG responses to evaluate
        rag_topics: Topics/queries to evaluate
        llm_config: LLM configuration
        nugget_banks_path: Path to input nugget banks (file or directory)
        judge_output_path: Leaderboard/qrels output path
        nugget_output_path: Path to store created/refined nuggets
        do_create_nuggets: If True, call create_nuggets()
        do_judge: If True, call judge()
        settings: Shared settings dict passed to both phases (fallback)
        nugget_settings: Settings dict passed to create_nuggets() (overrides settings)
        judge_settings: Settings dict passed to judge() (overrides settings)
        force_recreate_nuggets: If True, recreate even if file exists
        nugget_depends_on_responses: If True, pass responses to create_nuggets()
        judge_uses_nuggets: If True, pass nuggets to judge()
        config_name: Variant/sweep name for reproducibility tracking (default: "default")

    Returns:
        JudgeResult with leaderboard, qrels, and final nuggets
    """
    # Get nugget_banks_type from auto_judge (required for loading/saving nuggets)
    nugget_banks_type = getattr(auto_judge, "nugget_banks_type", None)

    # Resolve nugget output file path (add .nuggets.jsonl extension if needed)
    nugget_file_path = resolve_nugget_file_path(nugget_output_path) if nugget_output_path else None

    # Validate nugget_banks_type is available when needed for loading or saving
    needs_nugget_type = (
        (nugget_banks_path and nugget_banks_path.exists()) or  # Loading from input path
        (nugget_file_path and nugget_file_path.exists() and not force_recreate_nuggets) or  # Loading from output
        (do_create_nuggets and nugget_output_path)  # Will save nuggets (need type for future loading)
    )
    if needs_nugget_type and not nugget_banks_type:
        raise ValueError(
            "Cannot load/save nuggets: auto_judge does not define nugget_banks_type. "
            "Add nugget_banks_type class attribute to your AutoJudge implementation."
        )

    # Load input nuggets from path if provided
    input_nuggets: Optional[NuggetBanksProtocol] = None
    if nugget_banks_path and nugget_banks_path.exists() and nugget_banks_type:
        print(f"[judge_runner] Loading input nuggets: {nugget_banks_path}", file=sys.stderr)
        input_nuggets = load_nugget_banks_from_path(nugget_banks_path, nugget_banks_type)

    current_nuggets = input_nuggets
    leaderboard = None
    qrels = None

    # Step 1: Create or load nuggets
    if do_create_nuggets:
        # Check if output file exists and we can skip creation
        if nugget_file_path and nugget_file_path.exists() and not force_recreate_nuggets:
            print(f"[judge_runner] Loading existing nuggets (skipping creation): {nugget_file_path}", file=sys.stderr)
            current_nuggets = load_nugget_banks_from_path(nugget_file_path, nugget_banks_type)
        else:
            # Create nuggets
            nugget_kwargs = nugget_settings or settings or {}
            if nugget_kwargs:
                print(f"[judge_runner] create_nuggets settings: {nugget_kwargs}", file=sys.stderr)

            # Pass responses based on nugget_depends_on_responses flag
            responses_for_nuggets = rag_responses if nugget_depends_on_responses else None

            current_nuggets = auto_judge.create_nuggets(
                rag_responses=responses_for_nuggets,
                rag_topics=rag_topics,
                llm_config=llm_config,
                nugget_banks=input_nuggets,
                **nugget_kwargs,
            )
            # Verify created nuggets
            if current_nuggets is not None:
                # Verify type matches what auto_judge declared
                if nugget_banks_type and not isinstance(current_nuggets, nugget_banks_type):
                    print(
                        f"create_nuggets() returned {type(current_nuggets).__name__}, "
                        f"but auto_judge declares nugget_banks_type={nugget_banks_type.__name__}. "
                        f"Ensure create_nuggets() returns the declared type to avoid problems in nugget loading."
                        , sys.stderr
                    )
                topic_ids = [t.request_id for t in rag_topics]
                current_nuggets.verify(topic_ids)
                # Save immediately for crash recovery
                if nugget_file_path:
                    write_nugget_banks_generic(current_nuggets, nugget_file_path)
                    print(f"[judge_runner] Nuggets saved to: {nugget_file_path}", file=sys.stderr)

    # Step 2: Judge if requested
    if do_judge:
        judge_kwargs = judge_settings or settings or {}
        if judge_kwargs:
            print(f"[judge_runner] judge settings: {judge_kwargs}", file=sys.stderr)

        # Pass nuggets based on judge_uses_nuggets flag
        nuggets_for_judge = current_nuggets if judge_uses_nuggets else None

        leaderboard, qrels = auto_judge.judge(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            llm_config=llm_config,
            nugget_banks=nuggets_for_judge,
            **judge_kwargs,
        )

        # Step 3: Write outputs
        if judge_output_path:
            _write_outputs(
                leaderboard=leaderboard,
                qrels=qrels,
                rag_topics=rag_topics,
                judge_output_path=judge_output_path,
            )
            _write_run_config(
                judge_output_path=judge_output_path,
                config_name=config_name,
                do_create_nuggets=do_create_nuggets,
                do_judge=do_judge,
                llm_model=llm_config.model,
                settings=settings,
                nugget_settings=nugget_settings,
                judge_settings=judge_settings,
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
    judge_output_path: Path,
) -> None:
    """Verify and write leaderboard and qrels."""
    topic_ids = [t.request_id for t in rag_topics]

    # Resolve output paths from filebase
    leaderboard_path, qrels_path = resolve_judgment_file_paths(judge_output_path)
    leaderboard.verify(expected_topic_ids=topic_ids, on_missing="fix_aggregate")  # Reconsider setting this to "fix_aggregate" if want to ensure all run/topic combinations have values.
    leaderboard.write(leaderboard_path)
    print(f"[judge_runner] Leaderboard saved to: {leaderboard_path}", file=sys.stderr)

    if qrels is not None:
        qrels.verify(expected_topic_ids=topic_ids)
        write_qrel_file(qrel_out_file=qrels_path, qrels=qrels)
        print(f"[judge_runner] Qrels saved to: {qrels_path}", file=sys.stderr)


def _write_run_config(
    judge_output_path: Path,
    config_name: str,
    do_create_nuggets: bool,
    do_judge: bool,
    llm_model: str,
    settings: Optional[dict[str, Any]],
    nugget_settings: Optional[dict[str, Any]],
    judge_settings: Optional[dict[str, Any]],
) -> None:
    """
    Write run configuration for reproducibility.

    Creates {filebase}.config.yml with:
    - Workflow parameters (name, flags, settings)
    - LLM model used
    - Git info (commit, dirty, remote)
    - Timestamp
    """
    config_path = resolve_config_file_path(judge_output_path)

    config: dict[str, Any] = {
        "name": config_name,
        "create_nuggets": do_create_nuggets,
        "judge": do_judge,
        "llm_model": llm_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
    }

    # Only include non-empty settings
    if settings:
        config["settings"] = settings
    if nugget_settings:
        config["nugget_settings"] = nugget_settings
    if judge_settings:
        config["judge_settings"] = judge_settings

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[judge_runner] Config saved to: {config_path}", file=sys.stderr)