# Implementing an AutoJudge

This guide explains how to implement an AutoJudge, declare nugget bank formats, and configure workflow pipelines.

## AutoJudge Protocol

Every judge implements the `AutoJudge` protocol with two methods:

```python
class AutoJudge(Protocol):
    nugget_banks_type: Type[NuggetBanksProtocol]  # Optional: declare nugget format

    def judge(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs):
        """Score RAG responses. Returns (Leaderboard, Qrels)."""
        ...

    def create_nuggets(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs):
        """Create or refine nugget banks based on RAG responses. Returns NuggetBanks or None."""
        ...
```

## Minimal Judge (No Nuggets)

If your judge doesn't use nuggets, omit `nugget_banks_type`:

```python
from trec_auto_judge import Leaderboard

class SimpleJudge:
    def judge(self, rag_responses, rag_topics, llm_config, **kwargs):
        leaderboard = ...  # Score responses
        return leaderboard, None  # (Leaderboard, Qrels)

    def create_nuggets(self, rag_responses, rag_topics, llm_config, **kwargs):
        return None
```

## Judge with Nuggets

### Step 1: Declare the Nugget Format

Set `nugget_banks_type` to declare which format your judge uses:

```python
from typing import Type
from trec_auto_judge import NuggetBanks
from trec_auto_judge.nugget_data import NuggetBanksProtocol

class MyJudge:
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks
```

Available formats:
- `NuggetBanks` - AutoARGUE format (questions, claims, answers, references)
- `NuggetizerNuggetBanks` - Nuggetizer format (simpler text-based nuggets)

### Step 2: Implement create_nuggets()

```python
from trec_auto_judge.nugget_data import NuggetBanks, NuggetBank, NuggetQuestion

class MyJudge:
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs):
        banks = []
        for topic in rag_topics:
            bank = NuggetBank(query_id=topic.request_id, title_query=topic.title)

            # Generate/refine nuggets based on responses (e.g., via LLM)
            questions = generate_questions(topic, rag_responses, llm_config)
            bank.add_nuggets(questions)

            banks.append(bank)

        return NuggetBanks.from_banks_list(banks)
```

### Step 3: Implement judge()

```python
def judge(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs):
    scores = {}

    for response in rag_responses:
        topic_id = response.metadata.topic_id

        # Get nuggets for this topic
        if nugget_banks:
            topic_nuggets = nugget_banks.banks.get(topic_id)
            score = evaluate_with_nuggets(response, topic_nuggets, llm_config)
        else:
            score = evaluate_without_nuggets(response, llm_config)

        scores[topic_id] = score

    leaderboard = build_leaderboard(scores)
    return leaderboard, None  # (Leaderboard, Qrels)
```

### Step 4: Register the CLI

```python
# judge.py
from trec_auto_judge import auto_judge_to_click_command
from my_judge import MyJudge

cli = auto_judge_to_click_command(MyJudge(), "my-judge")

if __name__ == "__main__":
    cli()
```

## Workflow Declaration

Create `workflow.yml` to declare how your judge uses nuggets. The workflow uses boolean flags:

```yaml
create_nuggets: true   # Call create_nuggets() to generate/refine nuggets
judge: true            # Call judge() to produce leaderboard/qrels
```

### Common Workflow Configurations

| Configuration | create_nuggets | judge | Description |
|--------------|----------------|-------|-------------|
| Judge only | `false` | `true` | Judge doesn't use nuggets |
| Nuggify then judge | `true` | `true` | Create nuggets, then judge with them |
| Nuggify only | `true` | `false` | Just create nuggets, no judging |

### Judge Only (Default)

Judge doesn't use nuggets at all.

```yaml
create_nuggets: false
judge: true
```

### Nuggify Then Judge

Create nuggets first, then judge using them. Most common for nugget-based evaluation.

```yaml
create_nuggets: true
judge: true
```

### Nuggify Only

Create nuggets without judging (useful for nugget bank preparation).

```yaml
create_nuggets: true
judge: false
```

## Settings and Parameters

Pass hyperparameters to `create_nuggets()` and `judge()` via settings dicts:

```yaml
create_nuggets: true
judge: true

# Shared settings (passed to both phases as fallback)
settings:
  filebase: "{_name}"
  top_k: 20

# Phase-specific settings (override shared settings)
nugget_settings:
  extraction_style: "thorough"

judge_settings:
  threshold: 0.5
```

Settings are passed to AutoJudge methods as `**kwargs`:
- `judge()` receives `**(judge_settings or settings or {})`
- `create_nuggets()` receives `**(nugget_settings or settings or {})`

### Built-in Variables

Use `{variable}` syntax for template substitution in string values:

| Variable | Description |
|----------|-------------|
| `{_name}` | Configuration name ("default", variant name, or sweep name) |
| `{_nugget_filebase}` | Resolved nugget filebase (available in judge_settings) |

User-defined parameters are also available: `{top_k}` expands to the value of `top_k`.

**Note**: Parameters starting with `_` are reserved for built-ins and will cause a validation error.

### Naming Conventions

- **Parameters**: `snake_case` (e.g., `top_k`, `extraction_style`)
- **Variant/sweep names**: `train-case` (e.g., `ans-r`, `top-k-sweep`)

### Framework-Consumed Settings

Some settings are consumed by the framework and not passed to AutoJudge methods:

| Setting | Description |
|---------|-------------|
| `llm_model` | Override the model in `llm_config`. Applies to both phases. |

Example - run the same judge with different models:

```yaml
settings:
  llm_model: "gpt-4o"

sweeps:
  model-comparison:
    llm_model: ["gpt-4o", "claude-3-opus", "llama-3.1-70b"]
```

The `llm_model` setting overrides `llm_config.model` (from `llm-config.yml` or environment). This allows organizers to sweep over models without modifying the judge's LLM configuration file.

## Variants

Define named configurations that override base settings:

```yaml
create_nuggets: true
judge: true

settings:
  threshold: 0.5
  filebase: "{_name}"

variants:
  strict:
    threshold: 0.8

  ans-r:
    prompt: "AnswerR"
    threshold: 0.7
    judge_settings:
      use_citations: true
```

Run a specific variant:
```bash
./judge.py run --workflow workflow.yml --variant strict ...
```

Run all variants:
```bash
./judge.py run --workflow workflow.yml --all-variants ...
```

## Parameter Sweeps

Define parameter combinations for grid search:

```yaml
sweeps:
  top-k-sweep:
    top_k: [10, 20, 50]

  threshold-grid:
    top_k: [10, 20]
    threshold: [0.3, 0.5, 0.8]
```

Run a sweep (executes all combinations):
```bash
./judge.py run --workflow workflow.yml --sweep threshold-grid ...
```

The `threshold-grid` sweep produces 6 configurations (2 x 3 cartesian product).

## Lifecycle Flags

Control nugget creation and usage behavior:

```yaml
create_nuggets: true
judge: true

# Lifecycle flags
nugget_depends_on_responses: true   # Pass responses to create_nuggets() (default: true)
judge_uses_nuggets: true            # Pass nuggets to judge() (default: true)
force_recreate_nuggets: false       # Recreate even if file exists (default: false)
```

| Flag | Default | Description |
|------|---------|-------------|
| `nugget_depends_on_responses` | `true` | If false, `create_nuggets()` receives `rag_responses=None` |
| `judge_uses_nuggets` | `true` | If false, `judge()` receives `nugget_banks=None` |
| `force_recreate_nuggets` | `false` | If true, recreate nuggets even if output file exists |

### Auto-Load Behavior

When `create_nuggets: true` and a nugget file already exists:
- **Default**: Load existing nuggets instead of recreating (saves LLM calls)
- **With `--force-recreate-nuggets`**: Recreate nuggets anyway

CLI flag overrides workflow setting:
```bash
./judge.py run --workflow workflow.yml --force-recreate-nuggets ...
```

## Running the Judge

### CLI Subcommands

```bash
# Create/refine nuggets only
./judge.py nuggify --rag-responses runs/ --rag-topics topics.jsonl --store-nuggets nuggets.jsonl

# Judge with existing nuggets
./judge.py judge --rag-responses runs/ --nugget-banks nuggets.jsonl --output leaderboard.trec

# Execute based on workflow.yml (default command)
./judge.py run --workflow workflow.yml --rag-responses runs/ --output leaderboard.trec

# Run without subcommand uses 'run' with default workflow (judge only)
./judge.py --rag-responses runs/ --output leaderboard.trec
```

### Default Behavior

Running without a subcommand executes `run` with the default workflow (`judge=True, create_nuggets=False`):

```bash
./judge.py --rag-responses runs/ --output leaderboard.trec
```

## Directory Structure

```
trec25/judges/my-judge/
├── my-judge.py          # Judge implementation
├── judge.py             # CLI entry point
├── workflow.yml         # Workflow declaration
├── llm-config.yml       # LLM configuration (dev vs submission)
└── requirements.txt
```

## Configuration Files

**workflow.yml** - Declares the judge's pipeline. Fixed per judge implementation.

**llm-config.yml** - Configures LLM backend. Varies between environments:

```yaml
# Dev mode (direct config)
base_url: "http://localhost:8000/v1"
model: "meta-llama/Llama-3.1-8B-Instruct"

# Submission mode (preferences resolved by organizer)
model_preferences:
  - "gpt-4o"
  - "claude-3-opus"
```

## Output Files

Given a `filebase` setting (e.g., `filebase: "rubric"`), the framework generates:

| Output | Filename |
|--------|----------|
| Nugget banks | `{filebase}.nuggets.jsonl` |
| Leaderboard | `{filebase}.judgment.json` |
| Qrels | `{filebase}.judgment.qrels` |
| Run config | `{filebase}.config.yml` |

Example with `filebase: "rubric"`:
```
rubric.nuggets.jsonl
rubric.judgment.json
rubric.judgment.qrels
rubric.config.yml
```

If `filebase` already has an extension (`.json`, `.jsonl`), it's used as-is.

### Run Configuration File

The `.config.yml` file captures everything needed to reproduce the run:

```yaml
name: default                    # Variant/sweep name
create_nuggets: true
judge: true
llm_model: gpt-4o
timestamp: 2024-01-15T10:30:00+00:00
git:
  commit: abc123def456...
  dirty: "false"                 # "true", "false", or "unknown"
  remote: git@github.com:...     # or "none" or "unknown"
settings:
  top_k: 20
nugget_settings:
  extraction_style: thorough
judge_settings:
  threshold: 0.5
```

Git info fields:
- `commit`: Full SHA or "unknown" if not in a git repo
- `dirty`: Whether there are uncommitted changes
- `remote`: Origin URL, "none" if no remote, or "unknown" on error

## How Nugget Types Flow

1. **Judge declares**: `nugget_banks_type = NuggetBanks`
2. **CLI reads**: Stores judge in context, uses its type for `--nugget-banks` loading
3. **Framework loads**: `load_nugget_banks_generic(path, judge.nugget_banks_type)`
4. **Judge receives**: Correctly-typed `nugget_banks` in `judge()` and `create_nuggets()`
5. **Framework saves**: `write_nugget_banks_generic(nuggets, path)` works with any format

The framework handles format dispatch automatically based on your declared type.

## Using Nuggetizer Format

For simpler text-based nuggets:

```python
from trec_auto_judge.nugget_data import (
    NuggetizerNuggetBanks,
    NuggetizerNuggetBank,
    NuggetizerNugget
)

class NuggetizerJudge:
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetizerNuggetBanks

    def create_nuggets(self, rag_responses, rag_topics, llm_config, **kwargs):
        banks = []
        for topic in rag_topics:
            bank = NuggetizerNuggetBank(qid=topic.request_id, query=topic.title)
            bank.nuggets = [
                NuggetizerNugget(text="Key fact 1"),
                NuggetizerNugget(text="Key fact 2"),
            ]
            banks.append(bank)
        return NuggetizerNuggetBanks.from_banks_list(banks)
```

## Custom Nugget Formats

To create a custom format, implement `NuggetBanksProtocol`:

```python
from typing import ClassVar, Dict, List, Type
from pydantic import BaseModel
from trec_auto_judge.nugget_data.protocols import NuggetBankProtocol, NuggetBanksProtocol

class MyNuggetBank(BaseModel):
    topic_id: str
    facts: List[str]

    @property
    def query_id(self) -> str:
        return self.topic_id

class MyNuggetBanks(BaseModel):
    _bank_model: ClassVar[Type[MyNuggetBank]] = MyNuggetBank
    banks: Dict[str, MyNuggetBank] = {}

    @classmethod
    def from_banks_list(cls, banks: List[MyNuggetBank], overwrite: bool = False):
        result = {}
        for bank in banks:
            if bank.query_id in result and not overwrite:
                raise ValueError(f"Duplicate: {bank.query_id}")
            result[bank.query_id] = bank
        return cls(banks=result)
```

Key requirements:
- `NuggetBank` must have a `query_id` property
- `NuggetBanks` must have `_bank_model: ClassVar` pointing to the bank class
- `NuggetBanks` must have `banks: Dict[str, NuggetBank]` field
- `NuggetBanks` must have `from_banks_list(banks, overwrite=False)` classmethod