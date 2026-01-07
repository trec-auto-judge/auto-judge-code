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
    # Configuration resolution
    ResolvedConfiguration,
    resolve_default,
    resolve_variant,
    resolve_sweep,
)

from .paths import (
    resolve_nugget_file_path,
    resolve_judgment_file_paths,
    resolve_config_file_path,
    load_nugget_banks_from_path,
)

from .settings import (
    KeyValueType,
    create_cli_default_workflow,
    apply_cli_workflow_overrides,
)
