from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Set

from trec_auto_judge.utils import format_preview

MeasureName = str
AggFn = Callable[[Sequence[Any]], Any]
CastFn = Callable[[Any], Any]
OnMissing = Literal["default", "warn", "error", "fix_aggregate"]


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

    def verify(self,  on_missing:OnMissing, expected_topic_ids: Sequence[str], warn:Optional[bool]=False):
        LeaderboardVerification(leaderboard = self, warn=warn, expected_topic_ids=expected_topic_ids, on_missing=on_missing) \
            .complete_measures(include_all_row=True) \
            .complete_topics()
@dataclass(frozen=True)
class MeasureSpec:
    """
    Build-time definition of a measure.

    - `name`: key used in entry.values and output.
    - `aggregate`: computes the synthetic per-run value from per-topic values.
    - `cast`: normalizes/validates per-topic values when rows are added (output type will be input for aggregate function). Default no-op.
    - `default`: value used for missing (run_id, topic_id) pairs when build() fills gaps.
      Must be the same type as cast's output (i.e., valid input for aggregate). If None, no default is available.
    """
    name: MeasureName
    aggregate: AggFn
    cast: CastFn = lambda x: x
    default: Any = None


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


    def _detect_missing_run_topic(
        self,
        expected_topic_ids: Sequence[str],
    ) -> List[tuple[str, str]]:
        """
        Detect missing (run_id, topic_id) pairs.

        Returns list of (run_id, topic_id) tuples for each run that is missing
        expected topics.
        """
        existing_run_topic: Dict[str, Set[str]] = defaultdict(set)
        for e in self._rows:
            if e.topic_id != self.spec.all_topic_id:
                existing_run_topic[e.run_id].add(e.topic_id)

        expected_set = set(expected_topic_ids)
        missing: List[tuple[str, str]] = []
        for run_id in existing_run_topic.keys():
            for topic_id in expected_set - existing_run_topic[run_id]:
                missing.append((run_id, topic_id))
        return missing

    def _compute_aggregates(
        self,
        entries: List[LeaderboardEntry],
        phantom_defaults: List[tuple[str, str]],
    ) -> List[LeaderboardEntry]:
        """
        Compute "all" row aggregates from entries and phantom defaults.

        Args:
            entries: Per-topic entries to aggregate
            phantom_defaults: (run_id, topic_id) pairs to include in aggregation
                using MeasureSpec.default values (no actual entries created)

        Returns:
            List of aggregate "all" row entries, one per run_id
        """
        by_run: Dict[str, Dict[MeasureName, List[Any]]] = defaultdict(lambda: defaultdict(list))

        # Collect values from actual entries
        for e in entries:
            if e.topic_id == self.spec.all_topic_id:
                continue
            for k, v in e.values.items():
                by_run[e.run_id][k].append(v)

        # Include phantom defaults
        for run_id, _ in phantom_defaults:
            for ms in self.spec.measures:
                if ms.default is not None:
                    by_run[run_id][ms.name].append(ms.default)

        # Build aggregate rows
        all_rows: List[LeaderboardEntry] = []
        for run_id, m2vals in by_run.items():
            agg_vals: Dict[MeasureName, Any] = {}
            for ms in self.spec.measures:
                vals = m2vals.get(ms.name, [])
                if vals:
                    agg_vals[ms.name] = ms.aggregate(vals)
            all_rows.append(LeaderboardEntry(run_id=run_id, topic_id=self.spec.all_topic_id, values=agg_vals))

        return all_rows

    def build(
        self,
        expected_topic_ids: Optional[Sequence[str]] = None,
        on_missing: OnMissing = "default",
    ) -> Leaderboard:
        """
        Build a Leaderboard with synthetic per-run `all_topic_id` rows.

        The returned Leaderboard contains:
        - all per-topic rows that were added
        - plus one additional row per run_id with topic_id == spec.all_topic_id

        Args:
            expected_topic_ids: If provided, handles missing (run_id, topic_id) pairs.
            on_missing: When expected_topic_ids is provided and gaps exist:
                - "default": silently create per-topic entries with defaults
                - "warn": create per-topic entries with defaults and print warning
                - "fix_aggregate": only fill defaults for "all" row aggregation (no per-topic entries)
                - "error": raise ValueError listing missing (run_id, topic_id) pairs
        """
        # Step 1: Detect missing pairs
        all_missing: List[tuple[str, str]] = []
        if expected_topic_ids is not None:
            all_missing = self._detect_missing_run_topic(expected_topic_ids)

        # Step 2: Handle missing based on mode
        filled_rows: List[LeaderboardEntry] = []
        phantom_defaults: List[tuple[str, str]] = []

        if all_missing:
            formatted_pairs = [f"({r}, {t})" for r, t in sorted(all_missing)]
            if on_missing == "error":
                raise ValueError(
                    f"Missing leaderboard entries for {len(all_missing)} (run_id, topic_id) pair(s): {format_preview(formatted_pairs)}"
                )

            if on_missing == "warn":
                print(f"Leaderboard Warning: {len(all_missing)} missing entries: {format_preview(formatted_pairs)}", file=sys.stderr)

            if on_missing in ("default", "warn"):
                # Create actual per-topic entries
                default_values = {ms.name: ms.default for ms in self.spec.measures if ms.default is not None}
                if default_values:
                    for run_id, topic_id in all_missing:
                        filled_rows.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=default_values))
            elif on_missing == "fix_aggregate":
                phantom_defaults = all_missing

        # Step 3: Compute aggregates
        all_entries = self._rows + filled_rows
        all_rows = self._compute_aggregates(all_entries, phantom_defaults)

        return Leaderboard(
            measures=self.spec.names,
            entries=tuple(all_entries + all_rows),
            all_topic_id=self.spec.all_topic_id,
        )



# === Verification ====



class LeaderboardVerificationError(Exception):
    """Raised when leaderboard verification fails."""
    pass


class LeaderboardVerification:
    """
    Fluent verifier for leaderboards.

    Chain verification methods to run multiple checks:

        LeaderboardVerification(leaderboard).complete_measures().same_topics_per_run()

    Or run all checks:

        LeaderboardVerification(leaderboard, expected_topic_ids=topics).all()

    Each method raises LeaderboardVerificationError on failure (fail-fast).
    """

    def __init__(
        self,
        leaderboard: Leaderboard,
        on_missing: OnMissing,
        expected_topic_ids: Optional[Sequence[str]] = None,
        warn: Optional[bool]=False
    ):
        """
        Initialize verifier.

        Args:
            leaderboard: The leaderboard to verify
            expected_topic_ids: Optional set of expected topic IDs
        """
        self.leaderboard = leaderboard
        self.expected_topic_ids = expected_topic_ids
        self.warn = warn
        self.on_missing = on_missing

    def _raise_or_warn(self, err:LeaderboardVerificationError):
        if self.warn:
            print(f"Leaderboard Verification Warning: {err}", file=sys.stderr)
        else:
            raise err

    def complete_measures(self, include_all_row: bool = True) -> "LeaderboardVerification":
        """
        Verify that every (run_id, topic_id) entry contains all measures.

        Args:
            include_all_row: If True, also check synthetic all_topic_id rows

        Raises:
            LeaderboardVerificationError: If any entry is missing or has extra measures
        """
        required = set(self.leaderboard.measures)
        all_topic_id = self.leaderboard.all_topic_id

        missing_reports: list[str] = []
        for e in self.leaderboard.entries:
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

        if missing_reports and self.on_missing is not "fix_aggregate":
            self._raise_or_warn(LeaderboardVerificationError(
                "Leaderboard entries do not match the measure schema:\n  " + format_preview(missing_reports, limit=25, separator="\n  ")
            ))

        return self

    def same_topics_per_run(self, include_all_row: bool = False) -> "LeaderboardVerification":
        """
        Verify that all runs have the same set of topic_ids.

        This catches cases where run A has topics {t1,t2} but run B has {t1,t3},
        which makes per-run comparisons misleading.

        Args:
            include_all_row: If True, include synthetic all_topic_id rows

        Raises:
            LeaderboardVerificationError: If runs have different topic sets
        """
        all_topic_id = self.leaderboard.all_topic_id
        by_run: dict[str, Set[str]] = {}

        for e in self.leaderboard.entries:
            if not include_all_row and e.topic_id == all_topic_id:
                continue
            by_run.setdefault(e.run_id, set()).add(e.topic_id)

        if not by_run:
            return self

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
            self._raise_or_warn(LeaderboardVerificationError(
                f"Runs do not share the same topic set (reference={reference_run}):\n  " + format_preview(diffs, limit=25, separator="\n  ")
            ))

        return self

    def complete_topics(self, include_all_row: bool = False) -> "LeaderboardVerification":
        """
        Verify that every expected topic has at least one entry.

        Requires expected_topic_ids to be set in constructor.

        Args:
            include_all_row: If True, include synthetic all_topic_id rows

        Raises:
            LeaderboardVerificationError: If any expected topic is missing
        """
        if self.on_missing is "fix_aggregate":
            # Not checking for complete topics
            return self
        
        if self.expected_topic_ids is None:
            raise RuntimeError("Must set `expected_topic_ids` to non-null in order to check `complete_topics`")

        all_topic_id = self.leaderboard.all_topic_id
        expected = set(self.expected_topic_ids)
        
        
        for run, leaderboard_run_entries in groupby(sorted(self.leaderboard.entries
                                                           , key=lambda e: e.run_id)
                                                    , key=lambda e:e.run_id):
            seen = set()

            for e in leaderboard_run_entries:
                if not include_all_row and e.topic_id == all_topic_id:
                    continue
                if e.topic_id in expected:
                    seen.add(e.topic_id)

            missing = expected - seen
            if missing:
                missing_list = sorted(missing)
                self._raise_or_warn(LeaderboardVerificationError(
                    f"Run {run}: Missing leaderboard entries for {len(missing_list)} topic(s): {format_preview(missing_list)}"
                ))

        return self

    def no_extra_topics(self, include_all_row: bool = False) -> "LeaderboardVerification":
        """
        Verify no entries exist for non-expected topics.

        Requires expected_topic_ids to be set in constructor.

        Args:
            include_all_row: If True, include synthetic all_topic_id rows in check

        Raises:
            LeaderboardVerificationError: If entries exist for unknown topics
        """
        if self.expected_topic_ids is None:
            return self

        all_topic_id = self.leaderboard.all_topic_id
        expected = set(self.expected_topic_ids)
        extras = set()

        for e in self.leaderboard.entries:
            if not include_all_row and e.topic_id == all_topic_id:
                continue
            if e.topic_id not in expected and e.topic_id != all_topic_id:
                extras.add(e.topic_id)

        if extras:
            extra_list = sorted(extras)
            self._raise_or_warn(LeaderboardVerificationError(
                f"Leaderboard entries for {len(extra_list)} unexpected topic(s): {format_preview(extra_list)}"
            ))

        return self

    def all(self, include_all_row: bool = True) -> "LeaderboardVerification":
        """
        Run all verification checks.

        Checks run in order (fail-fast):
        1. complete_measures - every entry has all measures
        2. complete_topics - every expected topic has entries (if expected_topic_ids set)
        3. no_extra_topics - no entries for unknown topics (if expected_topic_ids set)
        4. same_topics_per_run - all runs have same topic set

        Args:
            include_all_row: If True, include synthetic all_topic_id rows in measure check

        Returns:
            self for chaining
        """
        
        if self.expected_topic_ids is not None:
            return (
                self.complete_measures(include_all_row=include_all_row)
                .complete_topics()
                .no_extra_topics()
                # .same_topics_per_run()   # omitted because that is already covered by the previous calls.
            )
        else:
            return (
                self.complete_measures(include_all_row=include_all_row)
                .same_topics_per_run()  # Fall back when we don't know the true topics.
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
