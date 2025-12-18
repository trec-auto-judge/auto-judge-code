from click import group
from .request import Request, load_requests_from_irds, load_requests_from_file, load_report
from typing import Protocol, Sequence, Optional

class AutoJudge(Protocol):
    def judge(
        self,
        rag_responses: Sequence["Report"],
        rag_topics: Sequence["Request"],
    ) -> tuple["Leaderboard", Optional["Qrels"]]:
        ...
from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests
from .leaderboard import Leaderboard, LeaderboardEntry, MeasureDef, MeasureName, write_leaderboard
from .qrels import Qrels, QrelEntry, write_qrel_file, read_qrel_file
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
