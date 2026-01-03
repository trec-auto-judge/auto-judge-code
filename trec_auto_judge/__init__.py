from typing import Iterable, Protocol, Sequence, Optional, Type

from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file
from .leaderboard.leaderboard import Leaderboard, LeaderboardEntry, MeasureSpec, LeaderboardSpec,  LeaderboardBuilder, VerificationError, verify_complete_measures, verify_complete_topics_per_run, verify_all, mean_of_bools, mean_of_floats, mean_of_ints
from .qrels.qrels import QrelsSpec, QrelRow, Qrels, build_qrels, verify_all_topics_present, verify_no_unexpected_topics, verify_qrels, write_qrel_file, doc_id_md5
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
    ) -> tuple["Leaderboard", Optional["Qrels"], Optional[NuggetBanksProtocol]]:
        """
        Judge RAG responses against topics.

        Returns:
            - Leaderboard: Rankings/scores for runs
            - Qrels: Optional fine-grained relevance judgments
            - NuggetBanks: Optional modified/emitted nuggets (for judge-emits-nuggets mode)
        """
        ...

    def create_nuggets(
        self,
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[NuggetBanksProtocol]:
        """
        Create nugget banks for the given topics.
        If existing_nuggets is provided, refine/extend them.
        Default implementation raises NotImplementedError.
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