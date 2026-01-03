"""
Workflow declaration for AutoJudge nugget/judge pipelines.

Participants declare their workflow in workflow.yml to enable TIRA orchestration.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel


class WorkflowMode(str, Enum):
    """Available workflow modes for judge execution."""

    JUDGE_ONLY = "judge-only"
    """Just judge, no nuggets involved."""

    NUGGIFY_THEN_JUDGE = "nuggify-then-judge"
    """Create nuggets first, store them, then judge using created nuggets."""

    JUDGE_EMITS_NUGGETS = "judge-emits-nuggets"
    """Judge creates nuggets as side output (no separate nuggify step)."""

    NUGGIFY_AND_REFINE = "nuggify-and-refine"
    """Create nuggets first, then judge refines and emits more nuggets."""


# Built-in NuggetBanks type paths (for convenience)
NUGGET_BANKS_AUTOARGUE = "trec_auto_judge.nugget_data.NuggetBanks"
NUGGET_BANKS_NUGGETIZER = "trec_auto_judge.nugget_data.NuggetizerNuggetBanks"

# Default type
DEFAULT_NUGGET_BANKS_TYPE = NUGGET_BANKS_AUTOARGUE


class Workflow(BaseModel):
    """Workflow configuration loaded from workflow.yml."""

    mode: WorkflowMode = WorkflowMode.JUDGE_ONLY
    nugget_banks_type: str = DEFAULT_NUGGET_BANKS_TYPE
    """Dotted import path for NuggetBanks container class."""
    nugget_input: Optional[str] = None
    nugget_output: Optional[str] = None

    @property
    def calls_create_nuggets(self) -> bool:
        """Whether this workflow calls create_nuggets() before judge()."""
        return self.mode in (
            WorkflowMode.NUGGIFY_THEN_JUDGE,
            WorkflowMode.NUGGIFY_AND_REFINE,
        )

    @property
    def judge_emits_nuggets(self) -> bool:
        """Whether judge() emits nuggets (requires --store-nuggets)."""
        return self.mode in (
            WorkflowMode.JUDGE_EMITS_NUGGETS,
            WorkflowMode.NUGGIFY_AND_REFINE,
        )

    @property
    def judge_uses_nuggets(self) -> bool:
        """Whether judge() expects nugget_banks input."""
        return self.mode in (
            WorkflowMode.NUGGIFY_THEN_JUDGE,
            WorkflowMode.NUGGIFY_AND_REFINE,
        )


def load_workflow(source: Union[str, Path]) -> Workflow:
    """
    Load workflow configuration from a YAML file.

    Args:
        source: Path to workflow.yml

    Returns:
        Workflow configuration

    Example workflow.yml:
        mode: "nuggify-then-judge"
    """
    path = Path(source)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Workflow.model_validate(data)


def load_workflow_from_directory(directory: Union[str, Path]) -> Optional[Workflow]:
    """
    Load workflow.yml from a judge directory if it exists.

    Args:
        directory: Judge directory (e.g., trec25/judges/my-judge/)

    Returns:
        Workflow if workflow.yml exists, None otherwise
    """
    path = Path(directory) / "workflow.yml"
    if path.is_file():
        return load_workflow(path)
    return None


# Default workflow for judges that don't declare one
DEFAULT_WORKFLOW = Workflow(mode=WorkflowMode.JUDGE_ONLY)