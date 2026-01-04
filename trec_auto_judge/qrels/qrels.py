import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Generic, Callable, Iterable, Sequence, Union

from trec_auto_judge.utils import format_preview


def doc_id_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class QrelsVerificationError(Exception):
    """Raised when qrels verification fails."""
    pass



R = TypeVar("R")

@dataclass(frozen=True)
class QrelRow:
    topic_id: str
    doc_id: str
    grade: int

@dataclass(frozen=True)
class QrelsSpec(Generic[R]):
    topic_id: Callable[[R], str]
    doc_id: Callable[[R], str]   
    grade: Callable[[R], int]
    # todo this should be an enum
    on_duplicate: str = "error"    # "error" | "keep_max" | "keep_last"

@dataclass(frozen=True)
class Qrels:
    """
    Collection of relevance judgments.

    Qrels are intentionally policy-free:
      - doc_id is opaque
      - no assumptions about corpus vs generated content
      - no assumptions about how grades are produced
    """
    rows: Sequence[QrelRow]
    
    def verify(self, expected_topic_ids: Optional[Sequence[str]], warn:Optional[bool]=False):
        QrelsVerification(self, expected_topic_ids=expected_topic_ids, warn=warn).all()
        
        
# === Qrel builder ===

def build_qrels(*, records: Iterable[R], spec: QrelsSpec[R]) -> list[QrelRow]:
    seen: dict[tuple[str, str], int] = {}
    for r in records:
        tid = spec.topic_id(r)
        did = spec.doc_id(r)
        g = int(spec.grade(r))

        key = (tid, did)
        if key in seen:
            if spec.on_duplicate == "error":
                raise ValueError(f"Duplicate qrel for {key}: old={seen[key]} new={g}")
            elif spec.on_duplicate == "keep_max":
                seen[key] = max(seen[key], g)
            elif spec.on_duplicate == "keep_last":
                seen[key] = g
            else:
                raise ValueError(f"Unknown on_duplicate: {spec.on_duplicate}")
        else:
            seen[key] = g

    return Qrels(rows=[QrelRow(topic_id=tid, doc_id=did, grade=g) for (tid, did), g in seen.items()])

#  === Qrel verification ===


class QrelsVerification:
    """
    Fluent verifier for qrels.

    Chain verification methods to run multiple checks:

        QrelsVerification(qrels, expected_topic_ids).complete_topics().no_duplicates()

    Or run all checks:

        QrelsVerification(qrels, expected_topic_ids).all()

    Each method raises QrelsVerificationError on failure (fail-fast).
    """

    def __init__(
        self,
        qrels: "Qrels",
        expected_topic_ids: Sequence[str],
        warn: Optional[bool]=False
    ):
        """
        Initialize verifier.

        Args:
            qrels: The qrels to verify
            expected_topic_ids: The expected topic IDs to verify against
        """
        self.qrels = qrels
        self.expected_topic_ids = expected_topic_ids
        self.warn = warn

    def _raise_or_warn(self, err: QrelsVerificationError):
        if self.warn:
            print(f"Qrels Verification Warning: {err}", file=sys.stderr)
        else:
            raise err


    def complete_topics(self) -> "QrelsVerification":
        """
        Verify every expected topic has at least one qrel row.

        Raises:
            QrelsVerificationError: If any topic is missing qrels
        """
        expected = set(self.expected_topic_ids)
        seen = set()

        for r in self.qrels.rows:
            if r.topic_id in expected:
                seen.add(r.topic_id)

        missing = expected - seen
        if missing:
            missing_list = sorted(missing)
            raise QrelsVerificationError(
                f"Missing qrels for {len(missing_list)} topic(s): {format_preview(missing_list)}"
            )

        return self

    def no_extra_topics(self) -> "QrelsVerification":
        """
        Verify no qrels exist for non-expected topics.

        Raises:
            QrelsVerificationError: If qrels exist for unknown topics
        """
        expected = set(self.expected_topic_ids)
        extras = {r.topic_id for r in self.qrels.rows} - expected

        if extras:
            extra_list = sorted(extras)
            raise QrelsVerificationError(
                f"Qrels for {len(extra_list)} unexpected topic(s): {format_preview(extra_list)}"
            )

        return self

    def no_duplicates(self) -> "QrelsVerification":
        """
        Verify no duplicate (topic_id, doc_id) pairs exist.

        Raises:
            QrelsVerificationError: If duplicates are found
        """
        checked: Dict[Tuple[str, str], QrelRow] = {}

        for r in self.qrels.rows:
            key = (r.topic_id, r.doc_id)
            if key in checked:
                raise QrelsVerificationError(
                    f"Duplicate qrel for topic={r.topic_id} doc_id={r.doc_id}: "
                    f"first={checked[key]}, duplicate={r}"
                )
            checked[key] = r

        return self

    def all(self) -> "QrelsVerification":
        """
        Run all verification checks.

        Checks run in order (fail-fast):
        1. no_duplicates - no duplicate (topic_id, doc_id) pairs
        2. complete_topics - every expected topic has qrels
        3. no_extra_topics - no qrels for unknown topics

        Returns:
            self for chaining
        """
        return self.no_duplicates().complete_topics().no_extra_topics()


# === serialization ===


def write_qrel_file(
    *,
    qrel_out_file: Union[str, Path],
    qrels: Qrels,
    system_id: str = "0",
) -> None:
    """
    Write qrels in standard TREC format:

        topic_id  system_id  doc_id  grade

    Notes:
    - `system_id` is conventionally '0' for human or synthetic judgments.
    - Ordering is deterministic (sorted by topic_id, then doc_id).
    """
    path = Path(qrel_out_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort for reproducibility
    rows = sorted(
        qrels.rows,
        key=lambda r: (r.topic_id, r.doc_id),
    )

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r.topic_id} {system_id} {r.doc_id} {r.grade}\n")
