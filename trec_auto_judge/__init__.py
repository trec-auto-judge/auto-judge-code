from typing import Iterable, Protocol, Sequence, Optional, Type

from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file
from .leaderboard.leaderboard import Leaderboard, LeaderboardEntry, MeasureSpec, LeaderboardSpec, LeaderboardBuilder, LeaderboardVerification, LeaderboardVerificationError, mean_of_bools, mean_of_floats, mean_of_ints
from .qrels.qrels import QrelsSpec, QrelRow, Qrels, build_qrels, QrelsVerification, QrelsVerificationError, write_qrel_file, doc_id_md5
from .llm.minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from .nugget_data import NuggetBanks, NuggetBanksProtocol
__version__ = '0.0.1'


# === The interface for AutoJudges to implement ====

class AutoJudge(Protocol):
    """Protocol for AutoJudge implementations."""

    nugget_banks_type: Type[NuggetBanksProtocol]
    """The NuggetBanks container type this judge uses."""

    def judge(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> tuple["Leaderboard", Optional["Qrels"]]:
        """
        Judge RAG responses against topics.

        Returns:
            - Leaderboard: Rankings/scores for runs
            - Qrels: Optional fine-grained relevance judgments
        """
        ...

    def create_nuggets(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[NuggetBanksProtocol]:
        """
        Create or refine nugget banks based on RAG responses.

        Args:
            rag_responses: RAG system outputs to analyze for nugget creation/refinement
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration for nugget generation
            nugget_banks: Optional existing nuggets to refine/extend

        Returns:
            NuggetBanks container, or None if judge doesn't support nuggets
        """
        ...


# === The click interface to the trec-auto-judge command line ====

from ._commands._evaluate import evaluate
from ._commands._export_corpus import export_corpus
from ._commands._list_models import list_models
from click import group
from .click_plus import option_rag_responses, option_rag_topics, option_ir_dataset, auto_judge_to_click_command

@group()
def main():
    pass


main.command()(evaluate)
main.command()(export_corpus)
main.add_command(list_models)


if __name__ == '__main__':
    main()