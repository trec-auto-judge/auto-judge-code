# Workflow Declaration

Participants declare their judge's workflow in `workflow.yml` to enable TIRA orchestration of nugget creation and judging pipelines.

## Quick Start

Create a `workflow.yml` in your judge directory:

```yaml
mode: "judge-only"
```

## Available Modes

| Mode | Description |
|------|-------------|
| `judge-only` | Just judge, no nuggets involved |
| `nuggify-then-judge` | Create nuggets first, store them, then judge using created nuggets |
| `judge-emits-nuggets` | Judge creates nuggets as a side output (no separate nuggify step) |
| `nuggify-and-refine` | Create nuggets first, then judge refines and emits more nuggets |

## Mode Details

### `judge-only`

Use when your judge doesn't use nuggets at all.

```yaml
mode: "judge-only"
```

**CLI equivalent:**
```bash
./judge.py judge --rag-responses $input --output $output
```

### `nuggify-then-judge`

Use when your judge creates nuggets in a separate step, then uses them for judging.

```yaml
mode: "nuggify-then-judge"
```

**CLI equivalent:**
```bash
./judge.py nuggify --rag-topics $topics --store-nuggets nuggets.jsonl
./judge.py judge --rag-responses $input --nugget-banks nuggets.jsonl --output $output
```

### `judge-emits-nuggets`

Use when your judge creates nuggets as part of the judging process (single pass).

```yaml
mode: "judge-emits-nuggets"
```

**CLI equivalent:**
```bash
./judge.py judge --rag-responses $input --output $output --store-nuggets nuggets.jsonl
```

### `nuggify-and-refine`

Use when your judge creates initial nuggets, then refines them during judging.

```yaml
mode: "nuggify-and-refine"
```

**CLI equivalent:**
```bash
./judge.py nuggify --rag-topics $topics --store-nuggets initial-nuggets.jsonl
./judge.py judge --rag-responses $input --nugget-banks initial-nuggets.jsonl --output $output --store-nuggets refined-nuggets.jsonl
```

## Using the `run` Command

The `run` command reads `workflow.yml` and automatically executes the appropriate steps:

```bash
./judge.py run --workflow workflow.yml --rag-responses $input --output $output --store-nuggets nuggets.jsonl
```

This is the recommended way to execute judges in TIRA, as it ensures the correct pipeline is followed based on the declared workflow.

## File Location

Place `workflow.yml` in your judge's root directory:

```
trec25/judges/my-judge/
├── workflow.yml
├── my-judge.py
├── llm-config.yml
└── ...
```

## Relationship to `llm-config.yml`

- **`workflow.yml`**: Declares the judge's pipeline (fixed per judge implementation)
- **`llm-config.yml`**: Configures LLM backend (varies between dev/submission environments)

These are separate files because workflow is intrinsic to the judge, while LLM config is environment-dependent.