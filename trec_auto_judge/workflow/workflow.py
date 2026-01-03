"""
Workflow declaration for AutoJudge nugget/judge pipelines.

Participants declare their workflow in workflow.yml to enable TIRA orchestration.
"""

from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel


# Built-in NuggetBanks type paths (for convenience)
NUGGET_BANKS_AUTOARGUE = "trec_auto_judge.nugget_data.NuggetBanks"
NUGGET_BANKS_NUGGETIZER = "trec_auto_judge.nugget_data.NuggetizerNuggetBanks"

# Default type
DEFAULT_NUGGET_BANKS_TYPE = NUGGET_BANKS_AUTOARGUE


class Workflow(BaseModel):
    """
    Workflow configuration loaded from workflow.yml.

    Controls which steps are executed:
    - create_nuggets: Whether to call create_nuggets() to generate/refine nuggets
    - judge: Whether to call judge() to produce leaderboard/qrels

    Example workflow.yml:
        create_nuggets: true
        judge: true
        nugget_banks_type: "trec_auto_judge.nugget_data.NuggetBanks"
    """

    create_nuggets: bool = False
    """Whether to call create_nuggets() to generate/refine nuggets."""

    judge: bool = True
    """Whether to call judge() to produce leaderboard/qrels."""

    nugget_banks_type: str = DEFAULT_NUGGET_BANKS_TYPE
    """Dotted import path for NuggetBanks container class."""

    nugget_input: Optional[str] = None
    """Path to existing nugget banks to load (for refinement or judge input)."""

    nugget_output: Optional[str] = None
    """Path to store created/refined nugget banks."""


def load_workflow(source: Union[str, Path]) -> Workflow:
    """
    Load workflow configuration from a YAML file.

    Args:
        source: Path to workflow.yml

    Returns:
        Workflow configuration

    Example workflow.yml:
        create_nuggets: true
        judge: true
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


# Default workflow for judges that don't declare one (judge only, no nuggets)
DEFAULT_WORKFLOW = Workflow()