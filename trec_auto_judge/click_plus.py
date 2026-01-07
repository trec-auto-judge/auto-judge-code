from pathlib import Path
from typing import Optional
from .io import load_runs_failsave
from .request import load_requests_from_irds, load_requests_from_file
from .llm import MinimaLlmConfig
from .llm_resolver import ModelPreferences, ModelResolver, ModelResolutionError
from .workflow import (
    load_workflow,
    resolve_default,
    resolve_variant,
    resolve_sweep,
    KeyValueType,
    create_cli_default_workflow,
    apply_cli_workflow_overrides,
)
from .judge_runner import run_judge
from .cli_default_group import DefaultGroup
import click
from . import AutoJudge


class ClickRagResponses(click.ParamType):
    name = "dir"

    def convert(self, value, param, ctx):
        if not value or not Path(value).is_dir():
            self.fail(f"The directory {value} does not exist, so I can not load rag responses from this directory.", param, ctx)
        runs = load_runs_failsave(Path(value))

        if len(runs) > 0:
            return runs

        self.fail(f"{value!r} contains no rag runs.", param, ctx)


def option_rag_responses():
    """Rag Run directory click option."""
    def decorator(func):
        func = click.option(
            "--rag-responses",
            type=ClickRagResponses(),
            required=True,
            help="The directory that contains the rag responses to evaluate."
        )(func)

        return func

    return decorator


class ClickIrDataset(click.ParamType):
    name = "ir-dataset"

    def fail_if_ir_datasets_is_not_installed(self, param, ctx, msg=""):
        try:
            import ir_datasets
            from ir_datasets import registry
        except:
            msg += " ir_datasets is not installed, so I can not try to load the data via ir_datasets. Please install ir_datasets to load data from there."
            self.fail(msg.strip(), param, ctx)

        try:
            import tira
            from tira.third_party_integrations import ir_datasets
        except:
            msg += " tira is not installed, so I can not try to load the data via tira ir_datasets integration. Please install tira to load data from there."
            self.fail(msg.strip(), param, ctx)

    def convert(self, value, param, ctx):
        self.fail_if_ir_datasets_is_not_installed(param, ctx)

        from ir_datasets import registry
        from tira.third_party_integrations import ir_datasets
        from .io import irds_from_dir, load_hf_dataset_config_or_none

        if value == "infer-dataset-from-context":
            candidate_files = set()
            if "rag_responses" in ctx.params:
                for r in ctx.params["rag_responses"]:
                    if r and r.path:
                        p = Path(r.path).parent
                        candidate_files.add(p / "README.md")
                        candidate_files.add(p.parent / "README.md")
                        candidate_files.add(p.parent.parent / "README.md")
  
            irds_config = None
            base_path = None
            for c in candidate_files:
                irds_config = load_hf_dataset_config_or_none(c, ["ir_dataset"])
                if irds_config:
                    base_path = c.parent
                    irds_config = irds_config["ir_dataset"]
                    break

            if not irds_config:
                raise ValueError("ToDo: Better error handling of wrong configurations")

            if "ir_datasets_id" in irds_config:
                return ir_datasets.load(irds_config["ir_datasets_id"])
            elif "directory" in irds_config:
                return irds_from_dir(str(base_path / irds_config["directory"]))
            else:
                raise ValueError("ToDo: Better error handling of incomplete configurations")

        if value and value in registry:
            return ir_datasets.load(value)

        if value and Path(value).is_dir() and (Path(value) / "queries.jsonl").is_file() and (Path(value) / "corpus.jsonl.gz").is_file():
            return irds_from_dir(value)

        if len(str(value).split("/")) == 2:
            return ir_datasets.load(value)
        else:
            raise ValueError("ToDo: Better error handling of incomplete configurations")


def option_ir_dataset():
    """Ir-dataset click option."""
    def decorator(func):
        func = click.option(
            "--ir-dataset",
            type=ClickIrDataset(),
            required=False,
            default="infer-dataset-from-context",
            help="The ir-datasets ID or a directory that contains the ir-dataset or TODO...."
        )(func)

        return func

    return decorator


class ClickRagTopics(ClickIrDataset):
    name = "file-or-ir-dataset"

    def fail_if_empty_or_return_otherwise(self, value, param, ctx, ret):
        if len(ret) == 0:
            self.fail(f"{value!r} contains 0 RAG topics.", param, ctx)
        else:
            return ret

    def convert(self, value, param, ctx):
        if value and Path(value).is_file():
            try:
                ret = load_requests_from_file(Path(value))
            except Exception as e:
                self.fail(f"The file {value} is not valid, no rag-topics could be loaded. {e}", param, ctx)
            return self.fail_if_empty_or_return_otherwise(value, param, ctx, ret)

        fail_msg = "The argument passed to --rag-topics is not a file."

        self.fail_if_ir_datasets_is_not_installed(param, ctx, msg=fail_msg)

        try:
            ds = super().convert(value, param, ctx)
        except:
            fail_msg += " The argument is also not a valid ir_datasets identifier that could be loaded."
            self.fail(fail_msg, param, ctx)

        ret = load_requests_from_irds(ds)
        return self.fail_if_empty_or_return_otherwise(value, param, ctx, ret)


def option_rag_topics():
    """Provide RAG topics CLI option."""
    def decorator(func):
        func = click.option(
            "--rag-topics",
            type=ClickRagTopics(),
            required=False,
            default="infer-dataset-from-context",
            help="The rag topics. Please either pass a local file that contains Requests in jsonl format (requires fields title and request_id). Alternatively, pass an ir-datasets ID to load the topics from ir_datasets."
        )(func)

        return func

    return decorator


def option_llm_config():
    """Optional llm-config.yml for model preferences."""
    def decorator(func):
        func = click.option(
            "--llm-config",
            type=click.Path(exists=True, path_type=Path),
            required=False,
            default=None,
            help="Path to llm-config.yml (dev: base_url+model, submission: model_preferences)"
        )(func)
        return func
    return decorator


def option_submission():
    """Flag to enable submission mode (resolve model_preferences)."""
    def decorator(func):
        func = click.option(
            "--submission",
            is_flag=True,
            default=False,
            help="Submission mode: resolve model_preferences against organizer's available models"
        )(func)
        return func
    return decorator


class ClickNuggetBanksPath(click.ParamType):
    """Click parameter type for nugget banks path (file or directory)."""
    name = "file-or-dir"

    def convert(self, value, param, ctx):
        if value is None:
            return None

        path = Path(value)

        if not path.exists():
            self.fail(f"Path {value} does not exist", param, ctx)

        if not path.is_file() and not path.is_dir():
            self.fail(f"Path {value} is neither a file nor directory", param, ctx)

        return path


def option_nugget_banks():
    """Optional nugget banks path CLI option (file or directory)."""
    def decorator(func):
        func = click.option(
            "--nugget-banks",
            type=ClickNuggetBanksPath(),
            required=False,
            default=None,
            help="Path to nugget banks file (JSON/JSONL) or directory."
        )(func)
        return func
    return decorator


def option_workflow():
    """Optional workflow.yml for declaring judge workflow."""
    def decorator(func):
        func = click.option(
            "--workflow",
            type=click.Path(exists=True, path_type=Path),
            required=False,
            default=None,
            help="Path to workflow.yml declaring the judge's nugget/judge pipeline."
        )(func)
        return func
    return decorator


def _resolve_llm_config(llm_config_path: Optional[Path], submission: bool = False) -> MinimaLlmConfig:
    """
    Resolve LLM config from llm-config.yml or environment.

    Two modes:
    - Dev mode (default): Load direct config (base_url + model) from file or env.
      For judge developers testing with their local LLM.
    - Submission mode (--submission): Resolve model_preferences against available
      models provided by the organizer.

    Args:
        llm_config_path: Path to llm-config.yml
        submission: If True, use submission mode (resolve model_preferences)
    """
    if submission:
        # Submission mode: resolve model_preferences against organizer's available models
        if llm_config_path is None:
            raise click.ClickException(
                "Submission mode requires --llm-config with model_preferences"
            )
        try:
            prefs = ModelPreferences.from_yaml(llm_config_path)
            resolver = ModelResolver.from_env()
            config = resolver.resolve(prefs)
            click.echo(f"Submission mode - resolved model: {config.model} from {config.base_url}", err=True)
            return config
        except ModelResolutionError as e:
            raise click.ClickException(str(e))
        except Exception as e:
            raise click.ClickException(f"Could not resolve model preferences from {llm_config_path}: {e}")

    # Dev mode: load direct config (base_url + model)
    if llm_config_path is not None:
        try:
            config = MinimaLlmConfig.from_yaml(llm_config_path)
            click.echo(f"Dev mode - loaded config: {config.model} from {config.base_url}", err=True)
            return config
        except FileNotFoundError:
            click.echo(f"Warning: Config file not found: {llm_config_path}", err=True)
        except ValueError as e:
            click.echo(f"Warning: {e}", err=True)

    # Fallback to environment-based config
    return MinimaLlmConfig.from_env()


def _apply_llm_model_override(
    llm_config: MinimaLlmConfig,
    settings: dict,
    submission: bool,
) -> tuple[MinimaLlmConfig, dict]:
    """
    Apply llm_model from settings to llm_config and strip it from settings.

    Steps:
    1. Extract llm_model from settings
    2. Apply it to llm_config via with_model()
    3. In submission mode, validate the model is allowed by organizer
    4. Strip llm_model from settings before passing to AutoJudge

    Args:
        llm_config: Base LLM configuration
        settings: Settings dict (may contain 'llm_model')
        submission: Whether we're in submission mode

    Returns:
        Tuple of (updated_llm_config, settings_without_llm_model)
    """
    llm_model = settings.get("llm_model")
    stripped_settings = {k: v for k, v in settings.items() if k != "llm_model"}

    if not llm_model:
        return llm_config, stripped_settings

    # In submission mode, validate the model is allowed
    if submission:
        try:
            resolver = ModelResolver.from_env()
            available = resolver.available
            enabled_models = available.get_enabled_models()

            # Check if model is available (directly or via alias)
            canonical = available.resolve_alias(llm_model)
            if canonical not in available.models:
                click.echo(
                    f"Warning: llm_model '{llm_model}' is not available in submission mode. "
                    f"Available models: {enabled_models}. Ignoring llm_model setting.",
                    err=True,
                )
                return llm_config, stripped_settings

        except Exception as e:
            click.echo(
                f"Warning: Could not validate llm_model against available models: {e}. "
                f"Ignoring llm_model setting.",
                err=True,
            )
            return llm_config, stripped_settings

    # Apply the model override
    updated_config = llm_config.with_model(llm_model)
    click.echo(f"Model override from settings: {llm_model}", err=True)
    return updated_config, stripped_settings


def auto_judge_to_click_command(auto_judge: AutoJudge, cmd_name: str):
    """
    Create a Click command group for an AutoJudge with subcommands:
    - nuggify: Create/refine nugget banks only
    - judge: Judge with existing nugget banks
    - run: Execute according to workflow.yml (DEFAULT)

    Invoking without a subcommand runs the 'run' command.
    """
    from .request import Request
    from .report import Report
    from typing import Iterable

    @click.group(cmd_name, cls=DefaultGroup, default_cmd_name="run")
    @click.pass_context
    def cli(ctx):
        """AutoJudge command group."""
        ctx.ensure_object(dict)
        ctx.obj["auto_judge"] = auto_judge

    @cli.command("judge")
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @option_submission()
    @click.option("--output", type=Path, help="Leaderboard output file.", required=True)
    def judge_cmd(
        rag_topics: Iterable[Request],
        rag_responses: Iterable[Report],
        nugget_banks,
        llm_config: Optional[Path],
        submission: bool,
        output: Path,
    ):
        """Judge RAG responses using existing nugget banks."""
        resolved_config = _resolve_llm_config(llm_config, submission)

        run_judge(
            auto_judge=auto_judge,
            rag_responses=rag_responses,
            rag_topics=list(rag_topics),
            llm_config=resolved_config,
            nugget_banks_path=nugget_banks,
            judge_output_path=output,
            do_create_nuggets=False,
            do_judge=True,
        )

    @cli.command("nuggify")
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @option_submission()
    @click.option("--store-nuggets", type=Path, help="Output nuggets file.", required=True)
    def nuggify_cmd(
        rag_responses: Iterable[Report],
        rag_topics: Iterable[Request],
        nugget_banks,
        llm_config: Optional[Path],
        submission: bool,
        store_nuggets: Path
    ):
        """Create or refine nugget banks based on RAG responses."""
        resolved_config = _resolve_llm_config(llm_config, submission)

        result = run_judge(
            auto_judge=auto_judge,
            rag_responses=rag_responses,
            rag_topics=list(rag_topics),
            llm_config=resolved_config,
            nugget_banks_path=nugget_banks,
            judge_output_path=None,
            nugget_output_path=store_nuggets,
            do_create_nuggets=True,
            do_judge=False,
        )

    @click.option("--output", type=Path, help="The output file.", required=True)
    def run(rag_topics:Iterable[Request], rag_responses:Iterable[Report], output:Path):
        leaderboard, qrels = auto_judge.judge(rag_responses, rag_topics)
        
        topic_ids = {t.request_id for t in rag_topics}
        verify_leaderboard_topics(expected_topic_ids=topic_ids,
            entries=leaderboard.entries,
            include_all_row=True,
            require_no_extras=True
        )

        if result.nuggets is None:
            click.echo("Warning: Judge doesn't create nuggets (create_nuggets returned None)", err=True)
        else:
            click.echo(f"Nuggets written to {store_nuggets}", err=True)

    @cli.command("run")
    @option_workflow()
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @option_submission()
    @click.option("--out-dir", type=Path, help="Parent directory for all output files.", required=False)
    @click.option("--filebase", type=str, help="Override workflow filebase (e.g., 'my-run').", required=False)
    @click.option("--store-nuggets", type=Path, help="Override nugget output path.", required=False)
    @click.option("--variant", type=str, help="Run a named variant from workflow.yml (e.g., --variant $name).", required=False)
    @click.option("--sweep", type=str, help="Run a parameter sweep from workflow.yml (e.g., --sweep $name).", required=False)
    @click.option("--all-variants", is_flag=True, help="Run all variants defined in workflow.yml.")
    @click.option("--force-recreate-nuggets", is_flag=True, help="Recreate nuggets even if file exists.")
    @click.option("--create-nuggets/--no-create-nuggets", default=None, help="Override workflow create_nuggets flag.")
    @click.option("--judge/--no-judge", "do_judge", default=None, help="Override workflow judge flag.")
    @click.option("--set", "-S", "settings_overrides", multiple=True, type=KeyValueType(),
                  help="Override shared settings: --set key=value")
    @click.option("--nset", "-N", "nugget_settings_overrides", multiple=True, type=KeyValueType(),
                  help="Override nugget settings: --nset key=value")
    @click.option("--jset", "-J", "judge_settings_overrides", multiple=True, type=KeyValueType(),
                  help="Override judge settings: --jset key=value")
    @click.option("--nugget-depends-on-responses/--no-nugget-depends-on-responses",
                  default=None, help="Override nugget_depends_on_responses lifecycle flag.")
    @click.option("--nugget-judge/--no-nugget-judge", "judge_uses_nuggets",
                  default=None, help="Judge uses nuggets (REQUIRED when --workflow omitted).")
    def run_cmd(
        workflow: Optional[Path],
        rag_responses: Iterable[Report],
        rag_topics: Iterable[Request],
        nugget_banks,
        llm_config: Optional[Path],
        submission: bool,
        out_dir: Optional[Path],
        filebase: Optional[str],
        store_nuggets: Optional[Path],
        variant: Optional[str],
        sweep: Optional[str],
        all_variants: bool,
        force_recreate_nuggets: bool,
        create_nuggets: Optional[bool],
        do_judge: Optional[bool],
        settings_overrides: tuple,
        nugget_settings_overrides: tuple,
        judge_settings_overrides: tuple,
        nugget_depends_on_responses: Optional[bool],
        judge_uses_nuggets: Optional[bool],
    ):
        """Run judge according to workflow.yml (default command)."""
        # Load workflow or create default based on CLI flags
        if workflow:
            wf = load_workflow(workflow)
            click.echo(f"Loaded workflow: create_nuggets={wf.create_nuggets}, judge={wf.judge}", err=True)
        else:
            # No workflow file - require --nugget-judge flag
            if judge_uses_nuggets is None:
                raise click.UsageError(
                    "No --workflow file provided.\n\n"
                    "When running without workflow.yml, you must specify:\n"
                    "  --nugget-judge     (creates nuggets, then judges with them)\n"
                    "  --no-nugget-judge  (judge only, no nugget creation)\n"
                )
            wf = create_cli_default_workflow(judge_uses_nuggets)
            click.echo(f"Using CLI defaults: create_nuggets={wf.create_nuggets}, judge={wf.judge}", err=True)

        # Apply CLI overrides to workflow
        apply_cli_workflow_overrides(
            wf,
            settings_overrides,
            nugget_settings_overrides,
            judge_settings_overrides,
            nugget_depends_on_responses,
            judge_uses_nuggets,
        )

        # CLI --filebase overrides workflow settings.filebase
        if filebase:
            wf.settings["filebase"] = filebase
            click.echo(f"Filebase override: {filebase}", err=True)

        # Validate mutually exclusive options
        options_set = sum([bool(variant), bool(sweep), all_variants])
        if options_set > 1:
            raise click.UsageError("--variant, --sweep, and --all-variants are mutually exclusive.")

        # Resolve configurations based on CLI options
        if variant:
            configs = [resolve_variant(wf, variant)]
        elif sweep:
            configs = resolve_sweep(wf, sweep)
        elif all_variants:
            if not wf.variants:
                raise click.UsageError("No variants defined in workflow.")
            configs = [resolve_variant(wf, name) for name in wf.variants]
        else:
            configs = [resolve_default(wf)]


        # Step 1: Load base LLM config from file/env
        base_llm_config = _resolve_llm_config(llm_config, submission)

        # Convert rag_topics to list once (it's an iterable)
        topics_list = list(rag_topics)

        # Run each configuration
        for config in configs:
            click.echo(f"\n=== Running configuration: {config.name} ===", err=True)

            # Step 2-4: Apply llm_model from settings, validate in submission mode, strip from settings
            effective_llm_config, clean_settings = _apply_llm_model_override(
                base_llm_config, config.settings, submission
            )

            # Determine output paths: --store-nuggets overrides, otherwise use resolved config
            # (--filebase was already injected into wf.settings before resolution)
            nugget_output_path = store_nuggets or config.nugget_output_path
            judge_output_path = config.judge_output_path

            # Apply --out-dir prefix to output paths
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                if nugget_output_path:
                    nugget_output_path = out_dir / nugget_output_path
                if judge_output_path:
                    judge_output_path = out_dir / judge_output_path

            if nugget_output_path:
                click.echo(f"Nugget output: {nugget_output_path}", err=True)
            if judge_output_path:
                click.echo(f"Judge output: {judge_output_path}", err=True)

            # CLI flags override workflow settings (None means use workflow default)
            effective_create_nuggets = create_nuggets if create_nuggets is not None else wf.create_nuggets
            effective_do_judge = do_judge if do_judge is not None else wf.judge

            # each returns a JudgeResult object to contain Leaderboard, Qrels, Nuggets for meta-evaluation
            run_judge(
                auto_judge=auto_judge,
                rag_responses=rag_responses,
                rag_topics=topics_list,
                llm_config=effective_llm_config,
                nugget_banks_path=nugget_banks,
                judge_output_path=judge_output_path,
                nugget_output_path=nugget_output_path,
                do_create_nuggets=effective_create_nuggets,
                do_judge=effective_do_judge,
                settings=clean_settings,
                nugget_settings=config.nugget_settings,
                judge_settings=config.judge_settings,
                # Lifecycle flags
                force_recreate_nuggets=force_recreate_nuggets or wf.force_recreate_nuggets,
                nugget_depends_on_responses=wf.nugget_depends_on_responses,
                judge_uses_nuggets=wf.judge_uses_nuggets,
                config_name=config.name,
            )

            click.echo(f"Done configuration: {config.name}", err=True)

        click.echo(f"\nAll configurations complete.", err=True)

    return cli
