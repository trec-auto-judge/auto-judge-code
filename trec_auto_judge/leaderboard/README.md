# Leaderboard + Builder

This module defines a simple, safe way to construct a leaderboard from per-topic scores while computing
per-run aggregate ("all") rows automatically.

The design avoids the common failure mode where measure names are repeated in multiple places as raw
strings (and drift silently). Instead, `LeaderboardSpec` defines:

- which measures exist
- how to cast/normalize measure values
- how to aggregate per-topic values into an `"all"` row

`Leaderboard` itself is a thin, serializable dataclass (plus a `write()` method).


## Concepts

### `Leaderboard` (dataclass)
A `Leaderboard` contains:

- `measures`: an ordered tuple of measure names (schema)
- `entries`: all rows, including synthetic `"all"` rows
- `all_topic_id`: the topic id used for aggregated rows (default `"all"`)

It intentionally does *not* know how to aggregate. It is safe to serialize and write.

### `LeaderboardSpec` (build-time schema)
A `LeaderboardSpec` defines the leaderboard schema and build-time behavior:

- `MeasureSpec.name`: the measure key
- `MeasureSpec.cast`: converts/normalizes input values when rows are added
- `MeasureSpec.aggregate`: computes the `"all"` value from per-topic values

### `LeaderboardBuilder` (assembler)
A `LeaderboardBuilder` is the only place where:

- per-topic rows are added
- measure keys are validated (fail-fast on typos / missing keys)
- values are cast
- synthetic `"all"` rows are computed

---

## Quickstart

### Define aggregators for `all'

or use from leaderboard module.

Example

```python
from statistics import mean

def mean_of_bools(values):
    return mean(1.0 if bool(v) else 0.0 for v in values)
```

## Define a Leaderboard spec


```python

MY_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("GRADE", aggregate=mean_of_floats, cast=float),
    MeasureSpec("IS_MATCH", aggregate=mean_of_bools, cast=bool),
))
```

## Build a Leaderboard

This is the simplest interface to build a leaderboard.

```python

from pathlib import Path

b = LeaderboardBuilder(MY_SPEC)

b.add(run_id="runA", topic_id="t1", GRADE=0.9, IS_MATCH=True)
b.add(run_id="runA", topic_id="t2", GRADE=0.4, IS_MATCH=False)

lb = b.build()
lb.write(Path("leaderboard.tsv"))

```


## Advanced: record-builder for leaderboard

If you already have a sequence of objects from which different measures are derived, you can use `add_records` of the leaderboard builder.

```python
    b = LeaderboardBuilder(MY_SPEC)
    b.add_records(
        prompt_output,
        run_id=lambda r: r.run_id,
        topic_id=lambda r: r.query_id,
        get_values=lambda r: {c
            "GRADE": r.match_score,
            "IS_MATCH": r.is_match,
        },
    )
    return b.build()
```

This is safe because measure names ("GRADE", "IS_MATCH") live in the spec. If get_values contains a typo key (e.g., "ISMATCH"), or are missing, then builder.add(...) raises an error.


## Output format of Leaderboard.write

`Leaderboard.write()` produces white-space separated lines akin to `trec_evals -q`:

```
run_id measure topic_id value
```

runA    GRADE    t1     0.9
runA    IS_MATCH t1     True
runA    GRADE    all    0.65
runA    IS_MATCH all    0.5


The leaderboard will automatically include an `all` topic based on the aggregator



## Verification helpers

The leaderboard module includes verification utilities to catch *silent* data issues early, especially when entries are produced by multiple components (parsers, judges, conversions) and measure names vary by use case.

These functions validate two common invariants:

1. **Every entry contains a complete set of measures** (prevents missing keys that otherwise cause partial aggregation or incorrect comparisons).

2. **Every run reports the same set of topics** (prevents unfair “wins” from evaluating on fewer topics, and prevents aggregation from mixing incomparable sets).

### `verify_complete_measures`

Checks that **each `LeaderboardEntry` contains all expected measure keys** in its `values` mapping.


```python
def verify_complete_measures(
    *,
    measure_names: Sequence[MeasureName],
    entries: Iterable[LeaderboardEntry],
    all_topic_id: str = "all",
    include_all_row: bool = True,
) -> None:
```


* `measure_names`: the authoritative list of measures that must appear in every row.
* `include_all_row`: if `True`, also requires the synthetic `topic_id == all_topic_id` row to be complete. If `False`, the “all” row is ignored for completeness checks (useful when verifying *before* building the “all” row).

Typical use:

* Right after constructing entries from a source format (e.g., converting judge outputs into `LeaderboardEntry` rows).
* Before computing “all” rows or exporting.

### `verify_complete_topics_per_run`

Checks that **each `run_id` contains the same set of `topic_id`s** as all other runs.


```python
def verify_complete_topics_per_run(
    *,
    entries: Iterable[LeaderboardEntry],
    all_topic_id: str = "all",
    include_all_row: bool = False,
) -> None:
```


* This prevents comparisons where one run is missing topics (intentionally or accidentally).
* `include_all_row` defaults to `False` because the `all` row is synthetic and should not participate in topic coverage checks.

Typical use:

* After merging multiple runs into a single leaderboard.
* Before computing run-level aggregates or ranking runs.

### `verify_all`

Checks all.

```python
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
```

Convenience wrapper that runs:

* `verify_complete_measures(..., include_all_row=require_all_row_complete)`
* and optionally `verify_complete_topics_per_run(...)` if `require_same_topics_per_run=True`.

Recommended usage patterns:

* **Before building the `all` row**:

  ```python
  verify_all(
      measure_names=list(measures.keys()),
      entries=per_topic_entries,
      require_all_row_complete=False,   # no all-row yet
      require_same_topics_per_run=True,
  )
  ```

* **After building the `all` row**:

  ```python
  verify_all(
      measure_names=list(measures.keys()),
      entries=leaderboard.entries,
      require_all_row_complete=True,    # ensure all-row is complete too
      require_same_topics_per_run=True,
  )
  ```

These checks are designed to raise failures immediately rather than silently propagating into incorrect aggregate scores or wrong/empty leaderboards.
