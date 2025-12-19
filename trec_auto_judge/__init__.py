from click import group
from typing import Protocol, Sequence, Optional

from .click_plus import option_rag_responses, option_rag_topics, option_ir_dataset
from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file
from .leaderboard.leaderboard import Leaderboard, LeaderboardEntry, MeasureSpec, LeaderboardSpec,  LeaderboardBuilder, VerificationError, verify_complete_measures, verify_complete_topics_per_run, verify_all, mean_of_bools, mean_of_floats, mean_of_ints 
from .qrels.qrels import QrelsSpec, QrelRow, Qrels, build_qrels, verify_all_topics_present, verify_no_unexpected_topics, verify_qrels_topics, write_qrel_file, doc_id_md5
from ._commands._evaluate import evaluate
from ._commands._export_corpus import export_corpus

__version__ = '0.0.1'


@group()
def main():
    pass


main.command()(evaluate)
main.command()(export_corpus)


class AutoJudge(Protocol):
    def judge(
        self,
        rag_responses: Sequence["Report"],
        rag_topics: Sequence["Request"],
    ) -> tuple["Leaderboard", Optional["Qrels"]]:
        ...

if __name__ == '__main__':
    main()
