from typing import TypeVar, Generic, Callable, Iterable, Sequence, Iterable, Union

from pathlib import Path
from dataclasses import dataclass
import hashlib
from pathlib import Path

def doc_id_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()



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

def verify_all_topics_present(
    *,
    expected_topic_ids: Sequence[str],
    qrels: Qrels,
) -> None:
    """
    Verify that every expected topic_id has at least one qrel row.
    """
    expected = set(expected_topic_ids)
    seen = set()

    for r in qrels.rows:
        if r.topic_id in expected:
            seen.add(r.topic_id)

    missing = expected - seen
    if missing:
        # Keep message small but actionable
        raise ValueError(f"Missing qrels for {len(missing)} topic_id(s), e.g. {sorted(missing)[:5]}")
    
def verify_no_unexpected_topics(
    *,
    expected_topic_ids: Sequence[str],
    qrels: Qrels,
) -> None:
    expected = set(expected_topic_ids)
    extras = {r.topic_id for r in qrels.rows} - expected
    if extras:
        raise ValueError(f"Found unexpected topic_id(s) in qrels, e.g. {sorted(extras)[:5]}")


def verify_qrels_topics(
    *,
    expected_topic_ids: Sequence[str],
    qrels: Qrels,
    require_no_extras: bool = False,
) -> None:
    verify_all_topics_present(expected_topic_ids=expected_topic_ids, qrels=qrels)
    if require_no_extras:
        verify_no_unexpected_topics(expected_topic_ids=expected_topic_ids, qrels=qrels)


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
