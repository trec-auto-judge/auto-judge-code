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
