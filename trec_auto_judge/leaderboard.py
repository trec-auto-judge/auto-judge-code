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
