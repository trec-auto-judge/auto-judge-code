from typing import Iterable, Protocol, Sequence, Optional, Type

from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file
from .leaderboard.leaderboard import Leaderboard, LeaderboardEntry, MeasureSpec, LeaderboardSpec, LeaderboardBuilder, LeaderboardVerification, LeaderboardVerificationError, mean_of_bools, mean_of_floats, mean_of_ints
from .qrels.qrels import QrelsSpec, QrelRow, Qrels, build_qrels, QrelsVerification, QrelsVerificationError, write_qrel_file, doc_id_md5
from .llm.minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from .nugget_data import NuggetBanks, NuggetBanksProtocol
__version__ = '0.0.1'
from ir_datasets.indices.base import Docstore
from ir_datasets import Dataset

# === Dependency Injection ====

class HasLLM():
    llm_config: Optional[MinimaLlmConfig] = None

    def set_llm_config(self, llm_config: MinimaLlmConfig) -> None:
        self.llm_config = llm_config

    def get_llm_config(self) -> Optional[MinimaLlmConfig]:
        return self.llm_config

    def get_llm_backend(self) -> OpenAIMinimaLlm:
        return OpenAIMinimaLlm(self.get_llm_config())

class HasNuggetBank():
    nugget_bank: Optional[NuggetBanksProtocol] = None

    def set_nugget_bank(self, nugget_bank: NuggetBanksProtocol) -> None:
        self.nugget_bank = nugget_bank

    def get_nugget_bank(self) -> Optional[NuggetBanksProtocol]:
        return self.nugget_bank

class HasIrDataset():
    ir_dataset: Optional[Dataset] = None

    def set_dataset(self, ir_dataset: Dataset) -> None:
        self.ir_dataset = ir_dataset

    def get_dataset(self) -> Optional[Dataset]:
        return self.ir_dataset

class HasDocsstore(HasIrDataset):
    def docs_store(self) -> Docstore:
        return self.ir_dataset.docs_store()

# === The interfaces for AutoJudges to implement ====

class RagAutoJudge(Protocol):
    """The main protocol for AutoJudge implementations."""
    def judge(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        **kwargs
    ) -> Leaderboard:
        """
        Judge RAG responses against topics.

        Returns:
            Leaderboard: Rankings/scores for runs
        """
        ...

class QrelAutoJudge(Protocol):
    """The protocol for AutoJudge implementations that creates qrels."""
    def judge(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        **kwargs
    ) -> Qrels:
        """
        Create qrels for responses against topics.

        Returns:
            Qrels: fine-grained relevance judgments
        """
        ...

class NuggetAutoJudge(Protocol):
    """The protocol for AutoJudge implementations that creates or refines nugget banks."""
    def create_nuggets(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        **kwargs
    ) -> NuggetBanksProtocol:
        """
        Create or refine nugget banks based on RAG responses.

        Args:
            rag_responses: RAG system outputs to analyze for nugget creation/refinement
            rag_topics: Evaluation topics/queries

        Returns:
            NuggetBanks container
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