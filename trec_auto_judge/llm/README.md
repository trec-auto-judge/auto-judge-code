# MinimaLLM

Minimal async LLM backend with batching support for OpenAI-compatible endpoints.

## Overview

MinimaLLM is a lightweight, dependency-minimal LLM client designed for:
- **OpenAI-compatible endpoints** (OpenAI, vLLM, Ollama, etc.)
- **Async batch execution** with worker pool pattern
- **Built-in retry/backoff** with exponential backoff and jitter
- **Rate limiting** with RPM-based pacing and semaphore backpressure
- **Progress monitoring** with heartbeat and stall detection
- **DSPy integration** via adapter

Uses only Python stdlib (`urllib`, `asyncio`) for HTTP transport—no external HTTP library dependencies.



### Environment Configuration

```bash
# Required
export OPENAI_BASE_URL="http://localhost:8000/v1"  # Your LLM endpoint
export OPENAI_MODEL="gpt-3.5-turbo"                # Model name
export OPENAI_API_KEY="sk-..."                     # API key (or use OPENAI_TOKEN)

# Optional batch configuration
export BATCH_NUM_WORKERS=64                        # Concurrent workers (default: 64)
export BATCH_MAX_FAILURES=25                       # Abort after N failures (default: 25, "none" to disable)
export BATCH_HEARTBEAT_S=10.0                      # Progress print interval (default: 10s)
export BATCH_STALL_S=300.0                         # Stall warning timeout (default: 300s)
export BATCH_PRINT_FIRST_FAILURES=5                # Print first N failures verbosely (default: 5)
export BATCH_KEEP_FAILURE_SUMMARIES=20             # Keep last N failure summaries (default: 20)

# Optional transport configuration
export MAX_OUTSTANDING=32                          # Max concurrent requests (default: 32)
export RPM=600                                     # Requests per minute limit (default: 600)
export TIMEOUT_S=60.0                              # Request timeout (default: 60s)

# Optional retry/backoff configuration
export MAX_ATTEMPTS=6                              # Max retry attempts (default: 6)
export BASE_BACKOFF_S=0.5                          # Initial backoff delay (default: 0.5s)
export MAX_BACKOFF_S=20.0                          # Max backoff ceiling (default: 20s)
export JITTER=0.2                                  # Backoff jitter factor (default: 0.2 = ±20%)

# Optional cooldown configuration (after 429/503/504)
export COOLDOWN_FLOOR_S=0.0                        # Min cooldown delay (default: 0s)
export COOLDOWN_CAP_S=30.0                         # Max cooldown delay (default: 30s)
export COOLDOWN_HALFLIFE_S=20.0                    # Cooldown decay half-life (default: 20s)

# Optional HTTP configuration
export COMPRESS_GZIP=0                             # Enable gzip compression (default: 0=disabled)

# Optional caching
export CACHE_DIR="./cache"                         # Enable SQLite prompt cache (default: disabled)
```

### Basic Usage

```python
import asyncio
from trec_auto_judge.llm import OpenAIMinimaLlm, MinimaLlmRequest

async def main():
    # Create backend from environment
    backend = OpenAIMinimaLlm.from_env()

    # Single request
    req = MinimaLlmRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "Say hello"}]
    )

    response = await backend.generate(req)
    print(response.text)

    # Cleanup
    await backend.aclose()

asyncio.run(main())
```

## Batch Execution

### Basic Batch Runner

```python
import asyncio
from trec_auto_judge.llm import OpenAIMinimaLlm, MinimaLlmRequest

async def main():
    backend = OpenAIMinimaLlm.from_env()

    # Create batch of requests
    requests = [
        MinimaLlmRequest(
            request_id=f"req-{i}",
            messages=[{"role": "user", "content": f"Count to {i}"}]
        )
        for i in range(1, 11)
    ]

    # Execute batch with automatic parallelization
    results = await backend.run_batched(requests)

    # Process results
    for result in results:
        if isinstance(result, MinimaLlmResponse):
            print(f"{result.request_id}: {result.text}")
        else:  # MinimaLlmFailure
            print(f"{result.request_id}: FAILED - {result.message}")

    await backend.aclose()

asyncio.run(main())
```

**Features:**
- Parallel execution with configurable worker pool
- Preserves input order in results
- Returns `MinimaLlmResponse` for successes, `MinimaLlmFailure` for errors
- Built-in retry with exponential backoff
- Progress heartbeat and stall detection
- Early abort on max failures

### Generic Batch Runner

For batch processing with custom async functions:

```python
import asyncio
from trec_auto_judge.llm import OpenAIMinimaLlm, run_batched_callable

async def my_async_function(item):
    # Your async processing logic
    backend = OpenAIMinimaLlm.from_env()
    response = await backend.generate(...)
    return process(response)

async def main():
    backend = OpenAIMinimaLlm.from_env()

    items = [1, 2, 3, 4, 5]

    # Execute batch with custom function
    results = await run_batched_callable(
        items,
        my_async_function,
        backend.cfg.batch  # BatchConfig
    )

    await backend.aclose()

asyncio.run(main())
```

## DSPy Integration

### DSPy Batch Processing with `run_dspy_batch()`

For batch DSPy execution with automatic field extraction:

```python
import asyncio
import dspy
from pydantic import BaseModel
from typing import Optional
from trec_auto_judge.llm import run_dspy_batch

# Define annotation model with fields matching DSPy signature
class MyAnnotation(BaseModel):
    question: str
    context: str
    answer: Optional[str] = None
    confidence: Optional[float] = None

# Define DSPy signature
class QA(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

    @classmethod
    def convert_output(cls, prediction, obj: MyAnnotation):
        obj.answer = prediction.answer
        obj.confidence = float(prediction.confidence)

async def main():
    # Prepare annotation objects
    annotations = [
        MyAnnotation(question="What is 2+2?", context="Math question"),
        MyAnnotation(question="What is Paris?", context="Geography question"),
    ]

    # Execute batch (automatically extracts question & context fields)
    results = await run_dspy_batch(
        QA,                    # Signature class
        annotations,           # Annotation objects
        QA.convert_output     # Output converter
    )

    # Results have outputs filled in
    for result in results:
        print(f"Q: {result.question}")
        print(f"A: {result.answer} (confidence: {result.confidence})")

asyncio.run(main())
```

**Key Features:**
- **Automatic field extraction**: Introspects signature to find InputFields
- **Field name matching**: Annotation fields match signature fields by name
- **No manual enumeration**: Fields extracted automatically via `model_dump(include=...)`
- **Built-in batching**: Uses `run_batched_callable` internally
- **Backend lifecycle**: Creates and cleans up backend automatically

### Sync Wrapper for DSPy Batch

If your code is synchronous:

```python
import asyncio
from trec_auto_judge.llm import run_dspy_batch

def process_batch(annotations):
    return asyncio.run(run_dspy_batch(
        MySignature,
        annotations,
        MySignature.convert_output
    ))
```

## Configuration

### LLM Configuration

```python
from trec_auto_judge.llm import MinimaLlmConfig, OpenAIMinimaLlm

# Manual configuration
config = MinimaLlmConfig(
    base_url="http://localhost:8000/v1",
    model="gpt-3.5-turbo",
    api_key="sk-...",
    max_outstanding=32,
    rpm=600,
    timeout_s=60.0
)

backend = OpenAIMinimaLlm(config)
```

### Batch Configuration

```python
from trec_auto_judge.llm import BatchConfig

# Manual batch config
batch_config = BatchConfig(
    num_workers=64,
    max_failures=25,
    heartbeat_s=10.0,
    stall_s=300.0,
    print_first_failures=5,
    keep_failure_summaries=20
)
```

### Combined Configuration

```python
config = MinimaLlmConfig(
    base_url="http://localhost:8000/v1",
    model="gpt-3.5-turbo",
    batch=batch_config  # Compose batch config
)
```

### Accessing Batch Configuration

Batch fields live in `BatchConfig`, but can be accessed via properties on `MinimaLlmConfig` for convenience:

```python
config = MinimaLlmConfig.from_env()

# Batch fields are in config.batch
config.batch.num_workers
config.batch.max_failures
config.batch.heartbeat_s

# Convenience properties (backward compatibility)
config.num_workers          # Same as config.batch.num_workers
config.max_failures         # Same as config.batch.max_failures
```

**Note:** The actual field only exists in `BatchConfig`. The properties on `MinimaLlmConfig` are for backward compatibility.

## Advanced Features

### Custom Retry Logic

Built-in exponential backoff with jitter:

```python
config = MinimaLlmConfig(
    base_url="http://localhost:8000/v1",
    model="gpt-3.5-turbo",
    max_attempts=6,           # Max retry attempts
    base_backoff_s=0.5,       # Initial backoff
    max_backoff_s=20.0,       # Max backoff ceiling
    jitter=0.2                # Jitter factor (±20%)
)
```

### Cooldown After Overload

Automatic cooldown after 429/503/504 responses:

```python
config = MinimaLlmConfig(
    base_url="http://localhost:8000/v1",
    model="gpt-3.5-turbo",
    cooldown_floor_s=0.0,     # Minimum cooldown
    cooldown_cap_s=30.0,      # Maximum cooldown
    cooldown_halflife_s=20.0  # Exponential decay half-life
)
```

### Request Compression

Enable gzip compression for request bodies:

```python
config = MinimaLlmConfig(
    base_url="http://localhost:8000/v1",
    model="gpt-3.5-turbo",
    compress_gzip=True        # Enable gzip compression
)
```

### Custom Request Parameters

Pass additional OpenAI-compatible parameters:

```python
req = MinimaLlmRequest(
    request_id="req-1",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=100,
    extra={"top_p": 0.9, "stop": ["\n"]}  # Additional params
)
```

### Prompt Caching

Enable SQLite-backed prompt caching for reproducible runs and faster re-execution:

```python
config = MinimaLlmConfig(
    base_url="http://localhost:8000/v1",
    model="gpt-3.5-turbo",
    cache_dir="./cache"           # Enable caching
)
```

Or via environment:
```bash
export CACHE_DIR="./cache"
```

**Features:**
- **Multi-process safe**: Uses SQLite WAL mode for concurrent access
- **Content-addressed**: Cache key is SHA-256 of (model, messages, temperature, max_tokens, extra)
- **Automatic**: Transparent caching—same API, just set `cache_dir`


See below for Cache Control on how to force refresh and retry on prompt parse errors.

## Error Handling

### The Result Type

`generate()` returns `Result = Union[MinimaLlmResponse, MinimaLlmFailure]`. Failures are returned (not raised) with full retry context for debugging.

### MinimaLlmFailure Fields

```python
@dataclass(frozen=True)
class MinimaLlmFailure:
    request_id: str                      # Original request ID
    error_type: str                      # "TimeoutError", "HTTPError", etc.
    message: str                         # Error details
    attempts: int                        # Number of attempts made
    status: Optional[int]                # HTTP status code (408 for timeout)
    body_snippet: Optional[str]          # First 300 chars of response body
    timeout_s: Optional[float]           # Configured timeout value
    attempt_timestamps: Tuple[float, ...]  # Monotonic timestamps of each attempt
```

### Handling Failures in Batch Execution

```python
from trec_auto_judge.llm import MinimaLlmResponse, MinimaLlmFailure

results = await backend.run_batched(requests)

for result in results:
    if isinstance(result, MinimaLlmResponse):
        # Success
        print(result.text)
    elif isinstance(result, MinimaLlmFailure):
        # Failure with full retry context
        print(f"Error: {result.error_type}")
        print(f"Message: {result.message}")
        print(f"Attempts: {result.attempts}")
        print(f"Timeout: {result.timeout_s}s")
        print(f"Timestamps: {result.format_attempts()}")  # e.g., "[+0.0s, +1.2s, +3.5s]"
```

### Failure Output Format

When failures occur, the batch runner prints detailed diagnostics:

```
Failure 1: request-123
    TimeoutError: status=408
    attempts=6, timeout=60.0s [+0.0s, +1.2s, +3.5s, +8.1s, +17.3s, +36.8s]
    body=URLError: timed out
```

The timestamps show when each retry attempt was **sent** (not scheduled), helping diagnose whether timeouts are due to slow server responses or queueing delays.

### Early Abort on Max Failures

```python
config = MinimaLlmConfig.from_env()
config.batch.max_failures = 10  # Abort after 10 failures

try:
    results = await backend.run_batched(requests)
except RuntimeError as e:
    print(f"Batch aborted: {e}")
    # Recent failures are included in exception message
```

### Cache Control via `force_refresh` and `cached` status value

The generate() method accepts a force_refresh parameter to bypass cache lookup while still writing new responses to cache:

Normal call - uses cache if available

    response = await backend.generate(req)

Force fresh LLM call - bypasses cache, writes new response
      response = await backend.generate(req, force_refresh=True)

This is useful when a cached response caused downstream errors and you need a fresh response.

The cached Response Attribute

MinimaLlmResponse includes a cached: bool attribute indicating whether the response came from cache:

    response = await backend.generate(req)
    if response.cached:
        print("Response served from cache")
    else:
        print("Fresh LLM call made")

The batch runner uses this to report accurate throughput statistics (distinguishing cache hits from actual LLM calls).

Context Variables for Adapters

When wrapping OpenAIMinimaLlm in adapters (e.g., for DSPy), you may not control the function signatures between your retry logic and the generate() call. Context variables provide an alternative way to propagate these values:

    from trec_auto_judge.llm import (
        set_force_refresh, reset_force_refresh, get_force_refresh,
        set_last_cached, get_last_cached,
    )

Example: Retry on parse errors in run_dspy_batch()

When DSPy fails to parse an LLM response (e.g., missing required fields), the cached response may be stale or malformed. The retry logic needs to signal cache bypass, but DSPy's predictor.acall() doesn't accept a force_refresh parameter:

    async def process_with_retry(item):
        for attempt in range(max_attempts):
            # On retry, set contextvar to bypass cache
            if attempt > 0:
                token = set_force_refresh(True)
            try:
                result = await predictor.acall(**kwargs)  # No force_refresh param
                return result
            except AdapterParseError:
                continue  # Retry with fresh response
            finally:
                if attempt > 0:
                    reset_force_refresh(token)  # Always reset

The DSPy adapter checks get_force_refresh() internally and passes it to generate():

**Inside MinimaLlmDSPyLM.acall():**

    force_refresh = force_refresh or get_force_refresh()
    resp = await self._minimallm.generate(req, force_refresh=force_refresh)
    set_last_cached(resp.cached)  # Preserve for heartbeat stats
    return [resp.text]

| Function                                | Description                                   |
|-----------------------------------------|-----------------------------------------------|
| set_force_refresh(force: bool) -> Token | Request cache bypass, returns reset token     |
| get_force_refresh() -> bool             | Check if cache bypass requested               |
| reset_force_refresh(token)              | Reset to previous state (call in finally)     |
| set_last_cached(cached: bool)           | Record cache status after unwrapping response |
| get_last_cached() -> bool               | Get cache status (used by heartbeat)          |

## Examples

### Complete Example: Question Answering

```python
import asyncio
from trec_auto_judge.llm import OpenAIMinimaLlm, MinimaLlmRequest

async def qa_batch():
    backend = OpenAIMinimaLlm.from_env()

    questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?"
    ]

    requests = [
        MinimaLlmRequest(
            request_id=f"qa-{i}",
            messages=[{"role": "user", "content": q}],
            temperature=0.7,
            max_tokens=50
        )
        for i, q in enumerate(questions)
    ]

    results = await backend.run_batched(requests)

    for i, (q, result) in enumerate(zip(questions, results)):
        if isinstance(result, MinimaLlmResponse):
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: {result.text}\n")
        else:
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: ERROR - {result.message}\n")

    await backend.aclose()

if __name__ == "__main__":
    asyncio.run(qa_batch())
```



### Complete Example: DSPy Integration

See `trec25/judges/umbrela/umbrela_baseline.py` for a real-world example of using `run_dspy_batch()` for batch DSPy execution.

## Architecture

### Core Components

- **`OpenAIMinimaLlm`**: Main LLM backend (implements `AsyncMinimaLlmBackend`)
- **`MinimaLlmConfig`**: Configuration with environment loading
- **`BatchConfig`**: Batch execution configuration
- **`MinimaLlmDSPyLM`**: DSPy adapter (implements DSPy `BaseLM`)
- **`run_batched_callable`**: Generic async batch executor
- **`run_dspy_batch`**: DSPy batch executor with field extraction

### Design Philosophy

1. **Minimal dependencies**: Uses stdlib only for HTTP (`urllib`)
2. **Async-first**: Built on `asyncio` for efficient I/O
3. **Fail-fast**: Explicit validation and error reporting
4. **Type-safe**: Uses dataclasses and type hints
5. **Protocol-based**: Duck typing via `Protocol` classes
6. **Framework-agnostic**: Works with DSPy, LangChain, or standalone

## Troubleshooting

### "Could not locate DSPy BaseLM"

DSPy may have reorganized its imports. The adapter tries multiple import paths but may need updating for newer DSPy versions.

### "MinimaLlmDSPyLM was called synchronously inside a running event loop"

You tried to call DSPy synchronously from within an async context. Use `await predictor.acall(...)` instead of `predictor(...)`.

### "Aborting batch: N failures"

The batch hit the max failures threshold. Increase `BATCH_MAX_FAILURES` or investigate the failures in the error message.

### Requests timing out

Increase `TIMEOUT_S` environment variable or reduce `BATCH_NUM_WORKERS` to limit concurrency.

## Performance Tips

1. **Tune worker count**: Adjust `BATCH_NUM_WORKERS` based on your endpoint's capacity
2. **Configure RPM**: Set `RPM` to match your rate limits
3. **Adjust backpressure**: Tune `MAX_OUTSTANDING` to control concurrent requests
4. **Monitor heartbeat**: Watch progress output to detect stalls or slow responses
5. **Use batching**: Always prefer `run_batched()` over individual `generate()` calls for multiple requests

## See Also

- **DSPy Documentation**: https://github.com/stanfordnlp/dspy
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
- **Example Judges**: `trec25/judges/` for real-world usage examples