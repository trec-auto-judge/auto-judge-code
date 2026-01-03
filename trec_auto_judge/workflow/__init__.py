"""Workflow declaration for AutoJudge nugget/judge pipelines."""

from .workflow import (
    Workflow,
    load_workflow,
    load_workflow_from_directory,
    DEFAULT_WORKFLOW,
    # Built-in NuggetBanks type paths
    NUGGET_BANKS_AUTOARGUE,
    NUGGET_BANKS_NUGGETIZER,
    DEFAULT_NUGGET_BANKS_TYPE,
)
