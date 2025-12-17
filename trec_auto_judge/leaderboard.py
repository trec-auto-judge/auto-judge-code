
from pathlib import Path
from collections import defaultdict
from statistics import mean
from dataclasses import dataclass, field

from typing import Union, Callable, Sequence, Dict, List, DefaultDict, Iterable


MeasureName = str


@dataclass(frozen=True)
class MeasureDef:
    """
    Generic definition of a measure.

    aggregate(values) should compute the per-run 'all' value from per-topic values.
    """
    aggregate: Callable[[Sequence[object]], object]


@dataclass(frozen=True)
class LeaderboardEntry:
    run_id: str
    topic_id: str  # include "all" as a conventional aggregate topic id
    values: Dict[MeasureName, object]


@dataclass
class Leaderboard:
    """
    Generic Leaderboard object that supports aggregators:
    - stores entries (including the synthetic topic_id == "all" rows)
    - knows how to create those "all" rows given MeasureDef.aggregate for each measure
    """
    measures: Dict[MeasureName, MeasureDef]
    entries: List[LeaderboardEntry] = field(default_factory=list)

    def all_measure_names(self):
        return self.measures.keys()

    @classmethod
    def from_entries_with_all(
        cls,
        *,
        measures: Dict[MeasureName, MeasureDef],
        entries: Iterable[LeaderboardEntry],
        all_topic_id: str = "all",
        include_existing_all: bool = False,
    ) -> "Leaderboard":
        base_entries = list(entries)

        if not include_existing_all:
            base_entries = [e for e in base_entries if e.topic_id != all_topic_id]

        # Collect per-run per-measure values
        by_run: DefaultDict[str, DefaultDict[MeasureName, List[object]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for e in base_entries:
            for m in measures.keys():
                if m in e.values:
                    by_run[e.run_id][m].append(e.values[m])

        # Create synthetic "all" rows per run
        all_entries: List[LeaderboardEntry] = []
        for run_id, m2vals in by_run.items():
            agg_vals: Dict[MeasureName, object] = {}
            for m, vals in m2vals.items():
                if vals:
                    agg_vals[m] = measures[m].aggregate(vals)
            all_entries.append(
                LeaderboardEntry(run_id=run_id, topic_id=all_topic_id, values=agg_vals)
            )

        return cls(measures=measures, entries=base_entries + all_entries)

    def write(self, output: Path) -> None:
        """
        Write the leaderboard to a file. 
        """
        lines: List[str] = []
        for e in self.entries:
            for m in self.all_measure_names():
                if m in e.values:
                    lines.append("\t".join([e.run_id,m,e.topic_id,str(e.values[m])]))

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n")

# ================
# Default Aggregators

class MeanOfFloats():
    def aggregate(self, values: Sequence[object]) -> float:
        return mean(float(v) for v in values)

class MeanOfBools():
    def aggregate(self, values: Sequence[object]) -> float:
        return mean(1.0 if bool(v) else 0.0 for v in values)

