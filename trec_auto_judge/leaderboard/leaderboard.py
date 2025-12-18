from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MeasureName = str
AggFn = Callable[[Sequence[Any]], Any]
CastFn = Callable[[Any], Any]


#  ==== DataClasses for data storage and serialization ===  

@dataclass(frozen=True)
class LeaderboardEntry:
    """One row in a leaderboard: (run_id, topic_id) plus a mapping of measure -> value."""
    run_id: str
    topic_id: str
    values: Dict[MeasureName, Any]


@dataclass(frozen=True)
class Leaderboard:
    """
    Thin serialization vessel for leaderboard results.

    - `measures` defines the measure names.
    - `entries` contains per-topic rows and and per-measure `all_topic_id` rows.
    
    Developer note:
    - Aggregation logic lives in LeaderboardBuilder.
    """
    measures: Tuple[MeasureName, ...]
    entries: Tuple[LeaderboardEntry, ...]
    all_topic_id: str = "all"

    def all_measure_names(self) -> Tuple[MeasureName, ...]:
        """Return measure names in schema order."""
        return self.measures

    def write(self, output: Path) -> None:
        """
        Write the leaderboard as white-space separated lines: run_id <tab> measure <tab> topic_id <tab> value.

        Only measures present in each entry are written (allows sparse rows).
        """
        lines: List[str] = []
        for e in self.entries:
            for m in self.all_measure_names():
                if m in e.values:
                    lines.append("\t".join([e.run_id, m, e.topic_id, str(e.values[m])]))

        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing leaderboard to {output.absolute()}")   # ToDo: use a logger
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MeasureSpec:
    """
    Build-time definition of a measure.

    - `name`: key used in entry.values and output.
    - `aggregate`: computes the synthetic per-run value from per-topic values.
    - `cast`: normalizes/validates per-topic values when rows are added (output type will be input for aggregate function). Default no-op.
    """
    name: MeasureName
    aggregate: AggFn
    cast: CastFn = lambda x: x


@dataclass(frozen=True)
class LeaderboardSpec:
    """
    Build-time schema for a leaderboard.

    The spec defines all valid measure names with aggregator and caster. 
    Storing values for different names will raise an error.
    """
    measures: Tuple[MeasureSpec, ...]
    all_topic_id: str = "all"

    @property
    def names(self) -> Tuple[MeasureName, ...]:
        """Measure names in schema order."""
        return tuple(m.name for m in self.measures)

    @property
    def name_set(self) -> set[MeasureName]:
        """Measure names as a set for fast validation."""
        return set(self.names)

    def cast_values(self, values: Mapping[MeasureName, Any]) -> Dict[MeasureName, Any]:
        """
        Cast/normalize measure values using each MeasureSpec.cast.

        Assumes `values` contains all required measure keys.
        """
        return {m.name: m.cast(values[m.name]) for m in self.measures}


#  ==== Convenient Builder for Leaderboards ===

class LeaderboardBuilder:
    """
    Builder/assembler for Leaderboard.

    Responsibilities:
    - Collect per-topic rows (hand-filled or record-derived).
    - Validate measure keys (fail fast on typos/missing keys).
    - Cast values according to the spec.
    - Compute synthetic per-run `all_topic_id` rows using each measure's aggregator.
    """

    def __init__(self, spec: LeaderboardSpec):
        """Create a builder for a specific leaderboard specification."""
        self.spec = spec
        self._rows: List[LeaderboardEntry] = []

    def add(
        self,
        *,
        run_id: str,
        topic_id: str,
        values: Optional[Dict[MeasureName, Any]] = None,
        **kw: Any,
    ) -> None:
        """
        Add one per-topic row.

        Provide either:
        - `values={...}` (a dict of measure -> value), OR
        - keyword args (e.g., GRADE=..., IS_MATCH=...).

        This method is strict by default:
        - Unknown measure keys raise KeyError.
        - Missing measure keys raise KeyError.
        """
        if values is None:
            values = kw
        elif kw:
            raise TypeError("Pass either values= or keyword measures, not both.")

        extra = set(values) - self.spec.name_set
        missing = self.spec.name_set - set(values)
        if extra:
            raise KeyError(f"Unknown measure(s): {sorted(extra)}")
        if missing:
            raise KeyError(f"Missing measure(s): {sorted(missing)}")

        casted = self.spec.cast_values(values)
        self._rows.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=casted))

    def add_records(
        self,
        records: Iterable[Any],
        *,
        run_id: Callable[[Any], str],
        topic_id: Callable[[Any], str],
        get_values: Callable[[Any], Dict[MeasureName, Any]],
    ) -> None:
        """
        Add multiple rows from an iterable of arbitrary record objects.

        The caller supplies functions to extract:
        - `run_id(record)`
        - `topic_id(record)`
        - `get_values(record)` -> {measure_name: value, ...}
        """
        for r in records:
            self.add(run_id=run_id(r), topic_id=topic_id(r), values=get_values(r))

    def entries(self) -> tuple[LeaderboardEntry, ...]:
        """Return the currently staged per-topic entries (no synthetic 'all' rows)."""
        return tuple(self._rows)


    def build(self) -> Leaderboard:
        """
        Build a Leaderboard with synthetic per-run `all_topic_id` rows.

        The returned Leaderboard contains:
        - all per-topic rows that were added
        - plus one additional row per run_id with topic_id == spec.all_topic_id
        """
        by_run: Dict[str, Dict[MeasureName, List[Any]]] = defaultdict(lambda: defaultdict(list))

        for e in self._rows:
            if e.topic_id == self.spec.all_topic_id:
                # Avoid accidental double-aggregation if someone added "all" rows manually.
                continue
            for k, v in e.values.items():
                by_run[e.run_id][k].append(v)

        all_rows: List[LeaderboardEntry] = []
        for run_id, m2vals in by_run.items():
            agg_vals: Dict[MeasureName, Any] = {}
            for ms in self.spec.measures:
                vals = m2vals.get(ms.name, [])
                if vals:
                    agg_vals[ms.name] = ms.aggregate(vals)
            all_rows.append(LeaderboardEntry(run_id=run_id, topic_id=self.spec.all_topic_id, values=agg_vals))

        return Leaderboard(
            measures=self.spec.names,
            entries=tuple(self._rows + all_rows),
            all_topic_id=self.spec.all_topic_id,
        )


# === Verification ====



@dataclass(frozen=True)
class VerificationError(Exception):
    """Raised when leaderboard completeness constraints are violated."""
    message: str

    def __str__(self) -> str:
        return self.message


def verify_complete_measures(
    *,
    measure_names: Sequence[MeasureName],
    entries: Iterable[LeaderboardEntry],
    all_topic_id: str = "all",
    include_all_row: bool = True,
) -> None:
    """
    Verify that every (run_id, topic_id) entry contains all measures.

    By default, also checks the synthetic all_topic_id rows if present.
    """
    required = set(measure_names)

    missing_reports: list[str] = []
    for e in entries:
        if not include_all_row and e.topic_id == all_topic_id:
            continue
        present = set(e.values.keys())
        missing = required - present
        extra = present - required
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing={sorted(missing)}")
            if extra:
                parts.append(f"extra={sorted(extra)}")
            missing_reports.append(f"{e.run_id}/{e.topic_id}: " + ", ".join(parts))

    if missing_reports:
        preview = "\n  ".join(missing_reports[:25])
        more = "" if len(missing_reports) <= 25 else f"\n  ... ({len(missing_reports) - 25} more)"
        raise VerificationError(
            "Leaderboard entries do not match the measure schema:\n  " + preview + more
        )


def verify_complete_topics_per_run(
    *,
    entries: Iterable[LeaderboardEntry],
    all_topic_id: str = "all",
    include_all_row: bool = False,
) -> None:
    """
    Verify that all runs have the same set of topic_ids.

    This catches cases where run A has topics {t1,t2} but run B has {t1,t3},
    which makes per-run comparisons misleading.
    """
    by_run: dict[str, Set[str]] = {}
    for e in entries:
        if not include_all_row and e.topic_id == all_topic_id:
            continue
        by_run.setdefault(e.run_id, set()).add(e.topic_id)

    if not by_run:
        return

    runs = sorted(by_run.keys())
    reference_run = runs[0]
    reference_topics = by_run[reference_run]

    diffs: list[str] = []
    for r in runs[1:]:
        tset = by_run[r]
        missing = reference_topics - tset
        extra = tset - reference_topics
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing_topics={sorted(missing)}")
            if extra:
                parts.append(f"extra_topics={sorted(extra)}")
            diffs.append(f"{r} vs {reference_run}: " + ", ".join(parts))

    if diffs:
        preview = "\n  ".join(diffs[:25])
        more = "" if len(diffs) <= 25 else f"\n  ... ({len(diffs) - 25} more)"
        raise VerificationError(
            f"Runs do not share the same topic set (reference={reference_run}):\n  {preview}{more}"
        )


def verify_all(
    *,
    measure_names: Sequence[MeasureName],
    entries: Iterable[LeaderboardEntry],
    all_topic_id: str = "all",
    require_all_row_complete: bool = True,
    require_same_topics_per_run: bool = True,
) -> None:
    """
    Convenience: verify both
      (1) every entry has all measures
      (2) every run has the same set of topics
    """
    verify_complete_measures(
        measure_names=measure_names,
        entries=entries,
        all_topic_id=all_topic_id,
        include_all_row=require_all_row_complete,
    )
    if require_same_topics_per_run:
        verify_complete_topics_per_run(
            entries=entries,
            all_topic_id=all_topic_id,
            include_all_row=False,
        )
        



#  === Example aggregators (optional helpers) ====

def mean_of_floats(values: Sequence[Any]) -> float:
    """Aggregate numeric values via arithmetic mean (values cast to float)."""
    return mean(float(v) for v in values)

def mean_of_ints(values: Sequence[Any]) -> float:
    """Aggregate numeric values via arithmetic mean (values cast to float)."""
    return mean(float(v) for v in values)


def mean_of_bools(values: Sequence[Any]) -> float:
    """Aggregate booleans via mean of {0.0, 1.0}."""
    return mean(1.0 if bool(v) else 0.0 for v in values)
