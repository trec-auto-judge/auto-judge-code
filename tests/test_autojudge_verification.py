"""
Verification test suite for AutoJudge implementations.

Runs each verifier separately to provide granular feedback on issues.
This serves as a test driver similar to judge_runner but with individual
test cases for each verification check.

Note: These tests require a configured LLM backend. Set up via:
- Direct config: llm-config.yml with base_url and model
- Environment: OPENAI_BASE_URL and OPENAI_MODEL
"""

import pytest
from pathlib import Path
from typing import List, Optional

from trec_auto_judge import Request, Report, Leaderboard, Qrels
from trec_auto_judge.llm import MinimaLlmConfig
from trec_auto_judge.nugget_data import (
    NuggetBanks,
    NuggetBanksVerification,
    NuggetBanksVerificationError,
)
from trec_auto_judge.leaderboard.leaderboard import (
    LeaderboardVerification,
    LeaderboardVerificationError,
)
from trec_auto_judge.qrels.qrels import (
    QrelsVerification,
    QrelsVerificationError,
)
from trec_auto_judge.report import ReportMetaData, Rag24ReportSentence


# =============================================================================
# Fixtures
# =============================================================================

RUBRIC_JUDGE_DIR = Path(__file__).parent.parent / "trec25" / "judges" / "rubric"


@pytest.fixture
def llm_config() -> MinimaLlmConfig:
    """Load LLM config from judge's config file or environment."""
    config_path = RUBRIC_JUDGE_DIR / "llm-config.yml"
    try:
        return MinimaLlmConfig.from_yaml(config_path)
    except (FileNotFoundError, ValueError):
        # Fall back to environment variables
        return MinimaLlmConfig.from_env()


@pytest.fixture
def sample_topics() -> List[Request]:
    """Create sample topics for testing."""
    return [
        Request(
            request_id="topic-001",
            title="Climate change impacts",
            problem_statement="What are the main impacts of climate change?",
            background="Understanding climate change effects on ecosystems.",
        ),
        Request(
            request_id="topic-002",
            title="Renewable energy",
            problem_statement="What are the benefits of renewable energy?",
            background="Exploring sustainable energy alternatives.",
        ),
    ]


@pytest.fixture
def sample_responses(sample_topics) -> List[Report]:
    """Create sample RAG responses for testing."""
    responses = []
    for topic in sample_topics:
        for run_id in ["run-A", "run-B"]:
            metadata = ReportMetaData(
                run_id=run_id,
                narrative_id=topic.request_id,
                narrative=topic.problem_statement,
                team_id="test-team",
                type="automatic",
            )
            answer = [
                Rag24ReportSentence(
                    text=f"Sample response for {topic.title} from {run_id}.",
                    citations=[0, 1],
                ),
                Rag24ReportSentence(
                    text="This discusses the topic in detail with relevant information.",
                    citations=[1, 2],
                ),
            ]
            report = Report(
                metadata=metadata,
                answer=answer,
                references=["doc-0", "doc-1", "doc-2"],
            )
            responses.append(report)
    return responses


# =============================================================================
# Test Driver Class
# =============================================================================

class AutoJudgeTestDriver:
    """
    Test driver that runs an AutoJudge and stores results for verification.

    Usage:
        driver = AutoJudgeTestDriver(judge, topics, responses, llm_config)
        driver.run_create_nuggets()
        driver.run_judge()

        # Then use driver.nuggets, driver.leaderboard, driver.qrels in tests
    """

    def __init__(
        self,
        auto_judge,
        rag_topics: List[Request],
        rag_responses: List[Report],
        llm_config: MinimaLlmConfig,
    ):
        self.auto_judge = auto_judge
        self.rag_topics = rag_topics
        self.rag_responses = rag_responses
        self.llm_config = llm_config

        # Results populated by run methods
        self.nuggets: Optional[NuggetBanks] = None
        self.leaderboard: Optional[Leaderboard] = None
        self.qrels: Optional[Qrels] = None

    def run_create_nuggets(self, existing_nuggets: Optional[NuggetBanks] = None):
        """Run create_nuggets and store results."""
        self.nuggets = self.auto_judge.create_nuggets(
            rag_responses=self.rag_responses,
            rag_topics=self.rag_topics,
            llm_config=self.llm_config,
            nugget_banks=existing_nuggets,
        )
        return self.nuggets

    def run_judge(self, nugget_banks: Optional[NuggetBanks] = None):
        """Run judge and store results."""
        nuggets_to_use = nugget_banks or self.nuggets

        self.leaderboard, self.qrels = self.auto_judge.judge(
            rag_responses=self.rag_responses,
            rag_topics=self.rag_topics,
            llm_config=self.llm_config,
            nugget_banks=nuggets_to_use,
        )

        return self.leaderboard, self.qrels


# =============================================================================
# RubricJudge Tests
# =============================================================================

class TestRubricJudgeVerification:
    """Verification tests for RubricJudge implementation."""

    @pytest.fixture
    def rubric_judge(self):
        """Create RubricJudge instance."""
        from trec25.judges.rubric.rubric_autojudge import RubricJudge
        return RubricJudge(grade_threshold=3)

    @pytest.fixture
    def driver(self, rubric_judge, sample_topics, sample_responses, llm_config):
        """Create test driver for RubricJudge."""
        return AutoJudgeTestDriver(
            auto_judge=rubric_judge,
            rag_topics=sample_topics,
            rag_responses=sample_responses,
            llm_config=llm_config,
        )

    # -------------------------------------------------------------------------
    # create_nuggets verification tests
    # -------------------------------------------------------------------------

    @pytest.fixture
    def nuggets_created(self, driver):
        """Run create_nuggets and return driver with results."""
        driver.run_create_nuggets()
        return driver

    def test_create_nuggets_returns_nuggets(self, nuggets_created):
        """Verify create_nuggets returns a NuggetBanks object."""
        assert nuggets_created.nuggets is not None
        assert isinstance(nuggets_created.nuggets, NuggetBanks)

    def test_create_nuggets_complete_topics(self, nuggets_created, sample_topics):
        """Verify every topic has a nugget bank entry."""
        NuggetBanksVerification(
            nuggets_created.nuggets, sample_topics
        ).complete_topics()

    def test_create_nuggets_no_extra_topics(self, nuggets_created, sample_topics):
        """Verify no nugget banks exist for non-existent topics."""
        NuggetBanksVerification(
            nuggets_created.nuggets, sample_topics
        ).no_extra_topics()

    def test_create_nuggets_non_empty_banks(self, nuggets_created, sample_topics):
        """Verify each nugget bank has at least one nugget."""
        NuggetBanksVerification(
            nuggets_created.nuggets, sample_topics
        ).non_empty_banks()

    def test_create_nuggets_all_verification(self, nuggets_created, sample_topics):
        """Run all nugget verification checks."""
        NuggetBanksVerification(
            nuggets_created.nuggets, sample_topics
        ).all()

    # -------------------------------------------------------------------------
    # judge verification tests
    # -------------------------------------------------------------------------

    @pytest.fixture
    def judge_results(self, nuggets_created):
        """Run judge and return driver with results."""
        nuggets_created.run_judge()
        return nuggets_created

    def test_judge_returns_leaderboard(self, judge_results):
        """Verify judge returns a Leaderboard object."""
        assert judge_results.leaderboard is not None
        assert isinstance(judge_results.leaderboard, Leaderboard)

    def test_judge_leaderboard_complete_measures(self, judge_results):
        """Verify every leaderboard entry has all measures."""
        LeaderboardVerification(
            judge_results.leaderboard
        ).complete_measures()

    def test_judge_leaderboard_complete_measures_excluding_all_row(self, judge_results):
        """Verify per-topic entries have all measures (excluding 'all' row)."""
        LeaderboardVerification(
            judge_results.leaderboard
        ).complete_measures(include_all_row=False)

    def test_judge_leaderboard_same_topics_per_run(self, judge_results):
        """Verify all runs have the same set of topics."""
        LeaderboardVerification(
            judge_results.leaderboard
        ).same_topics_per_run()

    def test_judge_leaderboard_complete_topics(self, judge_results, sample_topics):
        """Verify every expected topic has leaderboard entries."""
        topic_ids = [t.request_id for t in sample_topics]
        LeaderboardVerification(
            judge_results.leaderboard, expected_topic_ids=topic_ids
        ).complete_topics()

    def test_judge_leaderboard_no_extra_topics(self, judge_results, sample_topics):
        """Verify no leaderboard entries for unexpected topics."""
        topic_ids = [t.request_id for t in sample_topics]
        LeaderboardVerification(
            judge_results.leaderboard, expected_topic_ids=topic_ids
        ).no_extra_topics()

    def test_judge_leaderboard_all_verification(self, judge_results, sample_topics):
        """Run all leaderboard verification checks."""
        topic_ids = [t.request_id for t in sample_topics]
        LeaderboardVerification(
            judge_results.leaderboard, expected_topic_ids=topic_ids
        ).all()

    # -------------------------------------------------------------------------
    # qrels verification tests (if judge returns qrels)
    # -------------------------------------------------------------------------

    def test_judge_qrels_returned(self, judge_results):
        """Verify judge returns qrels (may be None for some judges)."""
        assert judge_results.qrels is not None

    def test_judge_qrels_complete_topics(self, judge_results, sample_topics):
        """Verify every expected topic has qrel entries."""
        if judge_results.qrels is None:
            pytest.skip("Judge does not return qrels")

        topic_ids = [t.request_id for t in sample_topics]
        QrelsVerification(
            judge_results.qrels, expected_topic_ids=topic_ids
        ).complete_topics()

    def test_judge_qrels_no_extra_topics(self, judge_results, sample_topics):
        """Verify no qrel entries for unexpected topics."""
        if judge_results.qrels is None:
            pytest.skip("Judge does not return qrels")

        topic_ids = [t.request_id for t in sample_topics]
        QrelsVerification(
            judge_results.qrels, expected_topic_ids=topic_ids
        ).no_extra_topics()

    def test_judge_qrels_no_duplicates(self, judge_results, sample_topics):
        """Verify no duplicate (topic_id, doc_id) pairs in qrels."""
        if judge_results.qrels is None:
            pytest.skip("Judge does not return qrels")

        topic_ids = [t.request_id for t in sample_topics]
        QrelsVerification(
            judge_results.qrels, expected_topic_ids=topic_ids
        ).no_duplicates()

    def test_judge_qrels_all_verification(self, judge_results, sample_topics):
        """Run all qrels verification checks."""
        if judge_results.qrels is None:
            pytest.skip("Judge does not return qrels")

        topic_ids = [t.request_id for t in sample_topics]
        QrelsVerification(
            judge_results.qrels, expected_topic_ids=topic_ids
        ).all()


# =============================================================================
# Verification Failure Tests (ensure verifiers catch problems)
# =============================================================================

class TestVerificationCatchesProblems:
    """Tests that verify the verifiers actually catch problems."""

    def test_leaderboard_verification_catches_missing_topic(self, sample_topics):
        """Verify LeaderboardVerification catches missing topics."""
        from trec_auto_judge import LeaderboardSpec, LeaderboardBuilder, MeasureSpec, mean_of_floats

        spec = LeaderboardSpec(measures=(
            MeasureSpec("SCORE", aggregate=mean_of_floats, cast=float),
        ))
        builder = LeaderboardBuilder(spec)

        # Only add entry for first topic
        builder.add(run_id="run-A", topic_id="topic-001", values={"SCORE": 0.5})
        leaderboard = builder.build()

        topic_ids = [t.request_id for t in sample_topics]

        with pytest.raises(LeaderboardVerificationError):
            LeaderboardVerification(
                leaderboard, expected_topic_ids=topic_ids
            ).complete_topics()

    def test_qrels_verification_catches_duplicates(self, sample_topics):
        """Verify QrelsVerification catches duplicate entries."""
        from trec_auto_judge.qrels.qrels import Qrels, QrelRow

        # Create qrels with duplicate
        rows = [
            QrelRow(topic_id="topic-001", doc_id="doc-1", grade=1),
            QrelRow(topic_id="topic-001", doc_id="doc-1", grade=2),  # Duplicate!
        ]
        qrels = Qrels(rows=rows)

        topic_ids = [t.request_id for t in sample_topics]

        with pytest.raises(QrelsVerificationError):
            QrelsVerification(qrels, expected_topic_ids=topic_ids).no_duplicates()