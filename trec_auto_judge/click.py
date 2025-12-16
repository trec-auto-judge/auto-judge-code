from pathlib import Path
from .io import load_runs_failsave
from .request import Request, load_requests

def option_rag_responses():
    import click
    class ClickRagResponses(click.ParamType):
        name = "dir"

        def convert(self, value, param, ctx):
            if not value or not Path(value).is_dir():
                self.fail(f"The directory {value} does not exist, so I can not load rag responses from this directory.", param, ctx)
            runs = load_runs_failsave(Path(value))

            if len(runs) > 0:
                return runs

            self.fail(f"{value!r} contains no rag runs.", param, ctx)

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

def option_rag_topics():
    import click
    class ClickRagTopics(click.ParamType):
        name = "file"

        def convert(self, value, param, ctx):
            if not value or not Path(value).is_file():
                self.fail(f"Trying to load RAG topics from {value}, but file does not exist.", param, ctx)
            topics = load_requests(Path(value))

            if len(topics) > 0:
                return topics

            self.fail(f"{value!r} contains no RAG topics.", param, ctx)

    """Provide RAG topics CLI option."""
    def decorator(func):
        func = click.option(
            "--rag-topics",
            type=ClickRagTopics(),
            required=True,
            help="The file that contains run_as_taskd_group topics for evaluation."
        )(func)

        return func

    return decorator

def option_ir_dataset():
    import click
    from tira.third_party_integrations import ir_datasets
    from tira.ir_datasets_util import load_ir_dataset_from_local_file
    from ir_datasets import registry

    def irds_from_dir(directory):
        ds = load_ir_dataset_from_local_file(Path(directory), str(directory))
        if str(directory) not in registry:
            registry.register(str(directory), ds)
        return ds

    def _load_cfg(c):
        from yaml import safe_load

        if c.is_file():
            txt = c.read_text()
            for cfg in txt.split("---"):
                try:
                    print(cfg)
                    return safe_load(cfg)["ir_dataset"]
                except:
                    pass

    class ClickIrDataset(click.ParamType):
        name = "dir"

        def convert(self, value, param, ctx):
            if value == "infer-dataset-from-context":
                candidate_files = set()
                if "rag_responses" in ctx.params:
                    for r in ctx.params["rag_responses"]:
                        if "path" in r:
                            p = Path(r["path"]).parent
                            candidate_files.add(p / "README.md")
                            candidate_files.add(p.parent / "README.md")
                irds_config = None
                base_path = None
                for c in candidate_files:
                    irds_config = _load_cfg(c)
                    if irds_config:
                        base_path = c.parent
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

            return ir_datasets.load(value)

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
