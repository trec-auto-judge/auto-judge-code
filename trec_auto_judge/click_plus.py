from pathlib import Path
from typing import Optional
from .io import load_runs_failsave
from .request import load_requests_from_irds, load_requests_from_file
from .llm import MinimaLlmConfig
from .llm_resolver import ModelPreferences, ModelResolver, ModelResolutionError
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

        if len(str(value).split("/") == 2):
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
            help="Path to llm-config.yml with model preferences"
        )(func)
        return func
    return decorator


class ClickNuggetBanks(click.ParamType):
    """Click parameter type for loading nugget banks from file or directory."""
    name = "file-or-dir"

    def convert(self, value, param, ctx):
        if value is None:
            return None

        from .nugget_data.nugget_banks import (
            load_nugget_banks_from_file,
            load_nugget_banks_from_directory,
        )

        path = Path(value)

        if path.is_file():
            try:
                return load_nugget_banks_from_file(path)
            except Exception as e:
                self.fail(f"Could not load nugget banks from {value}: {e}", param, ctx)

        if path.is_dir():
            try:
                return load_nugget_banks_from_directory(path)
            except Exception as e:
                self.fail(f"Could not load nugget banks from directory {value}: {e}", param, ctx)

        self.fail(f"Path {value} is neither a file nor directory", param, ctx)


def option_nugget_banks():
    """Optional nugget banks CLI option."""
    def decorator(func):
        func = click.option(
            "--nugget-banks",
            type=ClickNuggetBanks(),
            required=False,
            default=None,
            help="Nugget banks file (JSON/JSONL) or directory. Optional input for judges that use nuggets."
        )(func)
        return func
    return decorator


def _resolve_llm_config(llm_config_path: Optional[Path]) -> MinimaLlmConfig:
    """
    Resolve LLM config from llm-config.yml or environment.

    Supports two formats:
    1. Direct config (dev mode): base_url + model → use as-is
    2. Preferences (submission mode): model_preferences → resolve against available

    Fallback: MinimaLlmConfig.from_env()
    """
    import yaml

    if llm_config_path is not None:
        try:
            with open(llm_config_path) as f:
                data = yaml.safe_load(f) or {}

            # Dev mode: direct config with base_url and model
            if "base_url" in data and "model" in data:
                config = MinimaLlmConfig(
                    base_url=data["base_url"].rstrip("/"),
                    model=data["model"],
                    api_key=data.get("api_key", ""),
                )
                click.echo(f"Using direct config: {config.model} from {config.base_url}", err=True)
                return config

            # Submission mode: resolve preferences against available models
            if "model_preferences" in data:
                prefs = ModelPreferences.from_dict(data)
                resolver = ModelResolver.from_env()
                config = resolver.resolve(prefs)
                click.echo(f"Resolved model: {config.model} from {config.base_url}", err=True)
                return config

        except ModelResolutionError as e:
            raise click.ClickException(str(e))
        except Exception as e:
            click.echo(f"Warning: Could not load config from {llm_config_path}: {e}", err=True)

    # Fallback to environment-based config
    return MinimaLlmConfig.from_env()


class DefaultGroup(click.Group):
    """Click group that invokes a default subcommand when none is specified."""

    def __init__(self, *args, default_cmd_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def parse_args(self, ctx, args):
        # Show group help if --help is passed without subcommand
        if '--help' in args or '-h' in args:
            return super().parse_args(ctx, args)
        # If no args or first arg looks like an option, prepend default command
        if self.default_cmd_name and (not args or args[0].startswith('-')):
            args = [self.default_cmd_name] + list(args)
        return super().parse_args(ctx, args)

    def format_help(self, ctx, formatter):
        """Format help to include default command options."""
        super().format_help(ctx, formatter)

        # Append default command's options
        if self.default_cmd_name and self.default_cmd_name in self.commands:
            default_cmd = self.commands[self.default_cmd_name]
            formatter.write_paragraph()
            formatter.write_text(
                f"Default command: {self.default_cmd_name}"
            )
            formatter.write_paragraph()

            # Get default command's options
            with formatter.section("Default command options"):
                default_cmd.format_options(ctx, formatter)


def auto_judge_to_click_command(auto_judge: AutoJudge, cmd_name: str):
    """
    Create a Click command group for an AutoJudge with subcommands:
    - nuggify: Create nugget banks only
    - judge: Judge with existing nugget banks
    - nuggify-and-judge: Create nuggets then judge (DEFAULT)

    For backwards compatibility, invoking without a subcommand runs nuggify-and-judge.
    """
    from .qrels.qrels import write_qrel_file, verify_qrels
    from .leaderboard.leaderboard import verify_leaderboard_topics
    from .nugget_data.nugget_banks import NuggetBanks, write_nugget_banks
    from .request import Request
    from .report import Report
    from typing import Iterable

    @click.group(cmd_name, cls=DefaultGroup, default_cmd_name="nuggify-and-judge")
    def cli():
        """AutoJudge command group."""
        pass

    @cli.command("judge")
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @click.option("--output", type=Path, help="Leaderboard output file.", required=True)
    def judge_cmd(
        rag_topics: Iterable[Request],
        rag_responses: Iterable[Report],
        nugget_banks: Optional[NuggetBanks],
        llm_config: Optional[Path],
        output: Path
    ):
        """Judge RAG responses using existing nugget banks."""
        resolved_config = _resolve_llm_config(llm_config)

        leaderboard, qrels = auto_judge.judge(
            rag_responses, rag_topics, resolved_config, nugget_banks=nugget_banks
        )

        topic_ids = {t.request_id for t in rag_topics}
        verify_leaderboard_topics(
            expected_topic_ids=topic_ids,
            entries=leaderboard.entries,
            include_all_row=True,
            require_no_extras=True
        )
        leaderboard.write(output)

        if qrels is not None:
            verify_qrels(qrels=qrels, expected_topic_ids=topic_ids, require_no_extras=True)
            write_qrel_file(qrel_out_file=output.with_suffix(".qrels"), qrels=qrels)

    @cli.command("nuggify")
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @click.option("--store-nuggets", type=Path, help="Output nuggets file.", required=True)
    def nuggify_cmd(
        rag_topics: Iterable[Request],
        nugget_banks: Optional[NuggetBanks],
        llm_config: Optional[Path],
        store_nuggets: Path
    ):
        """Create nugget banks from topics (optionally refining existing nuggets)."""
        resolved_config = _resolve_llm_config(llm_config)

        created_nuggets = auto_judge.create_nuggets(
            rag_topics=rag_topics, resolved_config=resolved_config, nugget_banks=nugget_banks, llm_config=llm_config
        )

        if created_nuggets is None:
            click.echo("Warning: Judge doesn't create nuggets (create_nuggets returned None)", err=True)
        else:
            write_nugget_banks(created_nuggets, store_nuggets)
            click.echo(f"Nuggets written to {store_nuggets}", err=True)

    @cli.command("nuggify-and-judge")
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @click.option("--output", type=Path, help="Leaderboard output file.", required=True)
    @click.option("--store-nuggets", type=Path, help="Optional: store created nuggets.", required=False)
    def nuggify_and_judge_cmd(
        rag_topics: Iterable[Request],
        rag_responses: Iterable[Report],
        nugget_banks: Optional[NuggetBanks],
        llm_config: Optional[Path],
        output: Path,
        store_nuggets: Optional[Path]
    ):
        """Create nuggets, then judge RAG responses (default command)."""
        resolved_config = _resolve_llm_config(llm_config)

        # Create nuggets (optionally refining existing)
        created_nuggets = auto_judge.create_nuggets(
            rag_topics, resolved_config, nugget_banks=nugget_banks
        )

        # Use created nuggets for judging, or fall back to input nuggets
        nuggets_for_judging = created_nuggets if created_nuggets is not None else nugget_banks

        leaderboard, qrels = auto_judge.judge(
            rag_responses, rag_topics, resolved_config, nugget_banks=nuggets_for_judging
        )

        # Store nuggets if requested and available
        if store_nuggets:
            if created_nuggets is not None:
                write_nugget_banks(created_nuggets, store_nuggets)
                click.echo(f"Nuggets written to {store_nuggets}", err=True)
            else:
                click.echo("Warning: --store-nuggets ignored (judge doesn't create nuggets)", err=True)

        topic_ids = {t.request_id for t in rag_topics}
        verify_leaderboard_topics(
            expected_topic_ids=topic_ids,
            entries=leaderboard.entries,
            include_all_row=True,
            require_no_extras=True
        )
        leaderboard.write(output)

        if qrels is not None:
            verify_qrels(qrels=qrels, expected_topic_ids=topic_ids, require_no_extras=True)
            write_qrel_file(qrel_out_file=output.with_suffix(".qrels"), qrels=qrels)

    return cli