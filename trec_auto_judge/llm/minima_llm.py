# minima_llm.py
from __future__ import annotations

"""Minimal LLM endpoint. OpenAI-compatible async adapter (stdlib-only).

This adapter is intended as a *beginner-friendly default* for the TREC Auto-Judge
starter kit. Advanced users can ignore it and use the same environment variables
to configure their own LiteLLM/DSPy/LangChain/etc.

Key properties:
  - OpenAI-compatible POST /v1/chat/completions
  - async with explicit backpressure (Semaphore) and optional pacing (RPM gate)
  - retries with exponential backoff + jitter
  - optional gzip request-body compression (off by default)
  - batch runner with heartbeat + early abort

Environment variables are parsed in MinimaLlmConfig.from_env().
"""

import asyncio
import gzip
import hashlib
import json
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from .llm_config import BatchConfig, MinimaLlmConfig
from .llm_protocol import AsyncMinimaLlmBackend, Json, MinimaLlmFailure, MinimaLlmRequest, MinimaLlmResponse

import contextvars
# Task-local flag for cache bypass (safe for parallel async execution)
_force_refresh_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar('force_refresh', default=False)
_last_cached_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar('last_cached', default=False)


# Public API for adapter authors
def set_last_cached(cached: bool) -> None:
    """Call after generate() to record cache status for heartbeat tracking.

    Adapter authors should call this when their adapter unwraps MinimaLlmResponse
    and loses the cached attribute.
    """
    _last_cached_ctx.set(cached)


def get_last_cached() -> bool:
    """Get cache status from most recent generate() in this async task."""
    return _last_cached_ctx.get()

def set_force_refresh(force_refresh: bool)->contextvars.Token[bool]:
    """Call in generate() to force re-issuing the prompt (e.g. when response parsing failed)

    Adapter authors should call this when `acall` does not pass the `force_refresh` flag to the LLM backend.
    """
    return _force_refresh_ctx.set(force_refresh)

def reset_force_refresh(token:contextvars.Token[bool]):
    _force_refresh_ctx.reset(token)
    
def get_force_refresh() -> bool:
    """Get force_refresh requests for recent generate() in this async task."""
    return _force_refresh_ctx.get()


T = TypeVar("T")
R = TypeVar("R")


# ----------------------------
# Helpers: sleep + backoff
# ----------------------------

def _sleep_s(seconds: float) -> asyncio.Future:
    return asyncio.sleep(seconds)


def _jittered(base: float, jitter: float) -> float:
    if jitter <= 0:
        return base
    lo = max(0.0, 1.0 - jitter)
    hi = 1.0 + jitter
    return base * random.uniform(lo, hi)


# ----------------------------
# Pacing gate: simple rpm limiter
# ----------------------------

class RpmGate:
    def __init__(self, rpm: int):
        self._rpm = rpm
        self._lock = asyncio.Lock()
        self._next_ok = 0.0

    async def wait_turn(self) -> None:
        if self._rpm <= 0:
            return
        spacing = 60.0 / float(self._rpm)
        async with self._lock:
            now = time.monotonic()
            if now < self._next_ok:
                await asyncio.sleep(self._next_ok - now)
                now = time.monotonic()
            self._next_ok = now + spacing


# ----------------------------
# Cooldown gate (global)
# ----------------------------

class Cooldown:
    def __init__(self, floor_s: float, cap_s: float, halflife_s: float):
        self._floor = max(0.0, floor_s)
        self._cap = max(self._floor, cap_s)
        self._halflife = max(1e-6, halflife_s)

        self._lock = asyncio.Lock()
        self._cooldown_s = 0.0
        self._last = time.monotonic()

    def _decay(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last)
        self._last = now
        if self._cooldown_s <= 0.0:
            return
        # exponential decay with half-life
        decay = 0.5 ** (dt / self._halflife)
        self._cooldown_s *= decay
        if self._cooldown_s < self._floor:
            self._cooldown_s = 0.0

    async def wait_if_needed(self) -> None:
        async with self._lock:
            self._decay()
            cd = self._cooldown_s
        if cd > 0.0:
            await asyncio.sleep(cd)

    async def bump(self, suggested_s: float) -> None:
        async with self._lock:
            self._decay()
            new_cd = max(self._floor, suggested_s)
            self._cooldown_s = min(self._cap, max(self._cooldown_s, new_cd))


# ----------------------------
# HTTP helpers
# ----------------------------

def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _is_retriable_status(status: int) -> bool:
    return status in (408, 409, 425, 429, 500, 502, 503, 504)


def _is_overload_status(status: int) -> bool:
    return status in (429, 503, 504)


# ----------------------------
# Prompt cache (SQLite-backed)
# ----------------------------

class PromptCache:
    """
    SQLite-backed prompt cache. Multi-process safe via WAL mode.

    This cache stores LLM responses keyed by a hash of the request parameters.
    Multiple processes can safely read/write concurrently.
    """

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response_text TEXT NOT NULL,
                response_raw TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> Optional[Tuple[str, Optional[Json]]]:
        """Retrieve cached response by key. Returns (text, raw_json) or None."""
        row = self._conn.execute(
            "SELECT response_text, response_raw FROM cache WHERE key = ?",
            (key,)
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row[1]) if row[1] else None
        return (row[0], raw)

    def put(self, key: str, text: str, raw: Optional[Json]) -> None:
        """Store response in cache."""
        raw_json = json.dumps(raw) if raw else None
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, response_text, response_raw, created_at) VALUES (?, ?, ?, ?)",
            (key, text, raw_json, time.time())
        )
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


# ----------------------------
# Batch runner helpers
# ----------------------------

class _FailureCollector:
    def __init__(self, *, print_first_n: int, keep_last_n: int) -> None:
        self._print_first = max(0, int(print_first_n))
        self._keep = max(0, int(keep_last_n))
        self._seen = 0
        self._summaries: List[str] = []

    def record(self, f: MinimaLlmFailure) -> None:
        self._seen += 1
        # Include attempts and timeout in summary
        timeout_info = f", timeout={f.timeout_s}s" if f.timeout_s else ""
        msg = f"{f.request_id}: {f.error_type}: {f.message} (attempts={f.attempts}{timeout_info})"
        if self._seen <= self._print_first:
            ts = f.format_attempts()
            print(f"Failure {self._seen}: {f.request_id}")
            print(f"    {f.error_type}: {f.message}")
            print(f"    attempts={f.attempts}, timeout={f.timeout_s}s {ts}")
            if f.body_snippet:
                print(f"    body={f.body_snippet[:100]}")
        if self._keep > 0:
            self._summaries.append(msg)
            if len(self._summaries) > self._keep:
                self._summaries = self._summaries[-self._keep :]

    @property
    def count(self) -> int:
        return self._seen

    def summary_lines(self) -> List[str]:
        return list(self._summaries)

class _Heartbeat:
    def __init__(self, *, interval_seconds: float, stall_timeout_seconds: float, num_workers: int = 0) -> None:
        self._every_s = float(interval_seconds)
        self._stall_s = float(stall_timeout_seconds)
        self._num_workers = num_workers
        self._start = time.monotonic()
        self._last_done = self._start
        self._last_print = self._start
        self._cached_count = 0
        self._llm_count = 0
        self.done = 0

        # Per-interval LLM counters (excludes cache hits
        self._interval_llm_sent = 0      # LLM requests sent this interval
        self._interval_llm_received = 0  # LLM responses received this interval
        # self._prev_llm_count = 0         # LLM count at last print (for interval calculation)

    def mark_start(self) -> None:
        """Mark that a request has been sent (only counts LLM calls, not cache hits)."""
        self._interval_llm_sent += 1  # Assume that this call is not cached (we don't know yet, but we pull back the count in `mark_done` if need to be)

    def mark_done(self, *, cached: bool = False) -> None:
        self._last_done = time.monotonic()
        if cached:
            self._cached_count += 1
            self.done += 1
            self._interval_llm_sent -= 1 # because it was cached, it wasn't really sent -- but we did not know that in `mark_start`.
        else:
            self._interval_llm_received += 1
            self._llm_count += 1
            self.done += 1

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        if seconds < 0 or not seconds < float("inf"):
            return "?"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h{m:02d}m"
        if m > 0:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    def maybe_print(self, *, total: int, failed: int, queued: int = 0) -> None:
        now = time.monotonic()

        if self._every_s > 0 and (now - self._last_print) >= self._every_s:
            interval_s = now - self._last_print
            elapsed = now - self._start

            # Per-interval rates
            sent_rate = self._interval_llm_sent / interval_s if interval_s > 0 else 0
            recv_rate = self._interval_llm_received / interval_s if interval_s > 0 else 0

            # Reset interval counters
            self._interval_llm_sent = 0
            self._interval_llm_received = 0

            # ETA estimation based on LLM calls only (not cache hits)
            llm_done = self._llm_count
            remaining = max(0, total - self.done)

            if llm_done > 0 and elapsed > 0:
                llm_rate = llm_done / elapsed  # Overall LLM rate
                eta_s = remaining / llm_rate if llm_rate > 0 else float("inf")
                eta_str = self._fmt_eta(eta_s)
            else:
                eta_str = "?"

            print(
                f"[{elapsed:7.1f}s] "
                f"done={self.done}/{total} "
                f"sent={sent_rate:.1f}/s recv={recv_rate:.1f}/s "
                f"cached={self._cached_count} "
                f"failed={failed} "
                f"eta={eta_str}"
            )
            self._last_print = now

        if self._stall_s > 0 and (now - self._last_done) >= self._stall_s:
            elapsed = now - self._start
            print(
                f"[{elapsed:7.1f}s] WARNING: "
                f"no completions for {now - self._last_done:.1f}s"
            )
            self._last_done = now  # avoid spamming



# ----------------------------
# Generic batch executor
# ----------------------------

async def run_batched_callable(
    items: List[T],
    async_callable: Callable[[T], Awaitable[R]],
    batch_config: Optional[BatchConfig]=None,
) -> List[Union[R, MinimaLlmFailure]]:
    """
    Execute a batch of async calls using the worker pool pattern.

    This is a generic async batch executor that works with any async callable.
    It maintains batching infrastructure: worker pool, queue, heartbeat, and
    failure tracking.

    Parameters
    ----------
    items : List[T]
        List of items to process
    async_callable : Callable[[T], Awaitable[R]]
        Async function to call for each item
    batch_config : BatchConfig
        Configuration for batch execution (num_workers, max_failures, etc.)

    Returns
    -------
    List[Union[R, MinimaLlmFailure]]
        Results in input order (success values or MinimaLlmFailure)
    """
    
    if batch_config is None:
        batch_config=BatchConfig.from_env()

    num_workers = max(1, int(batch_config.num_workers))
    hb = _Heartbeat(
        interval_seconds=batch_config.heartbeat_s,
        stall_timeout_seconds=batch_config.stall_s,
        num_workers=num_workers,
    )
    fc = _FailureCollector(
        print_first_n=batch_config.print_first_failures,
        keep_last_n=batch_config.keep_failure_summaries,
    )
    abort_event = asyncio.Event()  # Shared flag for early abort

    total = len(items)
    results: List[Optional[Union[R, MinimaLlmFailure]]] = [None] * total
    q: asyncio.Queue[Tuple[int, T]] = asyncio.Queue()

    for i, item in enumerate(items):
        q.put_nowait((i, item))

    async def worker() -> None:
        while True:
            # Check for early abort before taking next item
            if abort_event.is_set():
                return

            try:
                i, item = q.get_nowait()
            except asyncio.QueueEmpty:
                return

            # Reset contextvar to prevent stale state from prior task bleeding through
            set_last_cached(False)
            hb.mark_start()  # We don't know yet if this was cached, makes the count wrong.
            cached = False
            try:
                result = await async_callable(item)
                # Check if result is a failure (generate() returns Result, not raises)
                if isinstance(result, MinimaLlmFailure):
                    fc.record(result)
                # Check if result was from cache (MinimaLlmResponse has cached attr)
                else:
                    # Get cached status. Since the DSPy adapter unwraps the MinimaLlmRespone object, and drops the cached flag, we use a context var as a fall-back
                    cached = bool(getattr(result, "cached", False)) or get_last_cached()
                results[i] = result
            except Exception as e:
                # Code errors (NameError, TypeError, etc.) propagate immediately
                if isinstance(e, (NameError, TypeError, AttributeError, SyntaxError, ImportError)):
                    raise
                # LLM and transport errors are recorded as failures
                f = MinimaLlmFailure(
                    request_id=f"input {i}/{item.request_id}" if hasattr(item, "request_id") else  f"input {i}",
                    error_type=type(e).__name__,
                    message=str(e),
                    attempts=1,
                )
                results[i] = f
                fc.record(f)

            hb.mark_done(cached=cached)
            q.task_done()

            # Check if we should trigger early abort after recording failure
            if batch_config.max_failures is not None and fc.count > batch_config.max_failures:
                abort_event.set()
                return

    async def heartbeat_loop() -> None:
        while True:
            hb.maybe_print(total=total, failed=fc.count)
            if hb.done >= total or abort_event.is_set():
                return
            await asyncio.sleep(hb._every_s)

    workers = [asyncio.create_task(worker()) for _ in range(max(1, int(batch_config.num_workers)))]
    hb_task = asyncio.create_task(heartbeat_loop())

    try:
        await asyncio.gather(*workers)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted. Syncing cache and cleaning up...")
        for w in workers:
            w.cancel()
        raise
    finally:
        # Always cancel heartbeat cleanly after workers finish
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    # Early-abort policy (raised after cleanup)
    if abort_event.is_set():
        lines = fc.summary_lines()
        tail = "\n".join(f"  - {s}" for s in lines)
        raise RuntimeError(f"Aborting batch: {fc.count} failures\nRecent failures:\n{tail}")

    # All results are filled
    return [r for r in results if r is not None]


# ----------------------------
# Main backend
# ----------------------------

class OpenAIMinimaLlm(AsyncMinimaLlmBackend):
    """
    OpenAI-compatible backend using stdlib urllib.

    Dependency-light, provides caching, retries, error handling, etc.
    """

    def __init__(self, cfg: MinimaLlmConfig):
        self.cfg = cfg

        self._sem = asyncio.Semaphore(cfg.max_outstanding)
        self._rpm = RpmGate(cfg.rpm)
        self._cooldown = Cooldown(cfg.cooldown_floor_s, cfg.cooldown_cap_s, cfg.cooldown_halflife_s)
        self._closed = False

        b = cfg._normalize_base_url(cfg.base_url)
        self._has_v1 = b.endswith("/v1")
        self._base = b

        # Initialize cache if configured
        self._cache: Optional[PromptCache] = None
        if cfg.cache_dir:
            os.makedirs(cfg.cache_dir, exist_ok=True)
            db_path = os.path.join(cfg.cache_dir, "minima_llm.db")
            self._cache = PromptCache(db_path)

    @classmethod
    def from_env(cls) -> "OpenAIMinimaLlm":
        """Construct backend from environment variables via MinimaLlmConfig."""
        return cls(MinimaLlmConfig.from_env())

    async def aclose(self) -> None:
        # urllib has no session to close; keep for symmetry.
        self._closed = True
        if self._cache is not None:
            print(f"Synchronizing cache at {self.cfg.cache_dir}.")
            self._cache.close()
            print("Cache synched.")

    def _endpoint(self, path: str) -> str:
        # If base_url already ends in /v1, avoid duplicating /v1.
        if self._has_v1 and path.startswith("/v1/"):
            path = path[len("/v1") :]
        return self._base.rstrip("/") + path

    def _headers(self, *, body_is_gzip: bool) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "trec-auto-judge-minimalllm"
        }
        if self.cfg.api_key is not None:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        if body_is_gzip:
            h["Content-Encoding"] = "gzip"
        return h

    @staticmethod
    def _parse_retry_after(headers: Dict[str, str]) -> Optional[float]:
        """Extract Retry-After header as seconds (float)."""
        ra = headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None

    def _make_cache_key(self, req: MinimaLlmRequest) -> str:
        """Generate cache key from request parameters."""
        obj: Dict[str, Any] = {"model": self.cfg.model, "messages": req.messages}
        if req.temperature is not None:
            obj["temperature"] = req.temperature
        if req.max_tokens is not None:
            obj["max_tokens"] = req.max_tokens
        if req.extra:
            obj["extra"] = req.extra
        canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def _post_json(self, url: str, payload: Json) -> Tuple[int, Dict[str, str], bytes]:
        """
        Perform a blocking urllib POST in a thread, to keep async-friendly.

        Returns (status_code, headers_dict, body_bytes).
        Headers are lowercased for consistent lookup.
        """
        body = _json_dumps(payload)
        body_is_gzip = bool(self.cfg.compress_gzip)
        if body_is_gzip:
            body = gzip.compress(body)

        req = urllib.request.Request(
            url=url,
            data=body,
            headers=self._headers(body_is_gzip=body_is_gzip),
            method="POST",
        )

        def _do() -> Tuple[int, Dict[str, str], bytes]:
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    return int(resp.status), headers, resp.read()
            except urllib.error.HTTPError as e:
                headers = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
                data = e.read() if e.fp is not None else b""
                return int(e.code), headers, data
            except urllib.error.URLError as e:
                # Timeout or connection error - return synthetic 408 for retry
                return 408, {}, f"URLError: {e.reason}".encode()

        return await asyncio.to_thread(_do)

    async def generate(self, req: MinimaLlmRequest, *, force_refresh: bool = False) -> MinimaLlmResult:
        """
        Generate a response for the given request.

        Parameters
        ----------
        req : MinimaLlmRequest
            The request to process
        force_refresh : bool
            If True, bypass cache lookup and make a fresh LLM call.
            The new response will still be written to cache.

        Returns MinimaLlmResponse on success, MinimaLlmFailure on error.
        Failures include full retry context: attempt count, timestamps, timeout.
        """
        
        # Check cache first (unless force_refresh)
        cache_key: Optional[str] = None
        if self._cache is not None:
            cache_key = self._make_cache_key(req)
            if not force_refresh:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    set_last_cached(True)
                    return MinimaLlmResponse(request_id=req.request_id, text=cached[0], raw=cached[1], cached=True)

        payload: Json = {
            "model": self.cfg.model,
            "messages": req.messages,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.extra:
            payload.update(req.extra)

        url = self._endpoint("/v1/chat/completions")

        attempt = 0
        attempt_timestamps: List[float] = []
        last_body: Optional[str] = None

        while True:
            attempt += 1
            await self._cooldown.wait_if_needed()
            await self._rpm.wait_turn()

            async with self._sem:
                attempt_timestamps.append(time.monotonic())  # Record send time
                status, headers, raw = await self._post_json(url, payload)

            body_text = raw.decode("utf-8", errors="replace")
            last_body = body_text[:300]

            if 200 <= status < 300:
                try:
                    data = json.loads(body_text)
                except Exception as e:
                    return MinimaLlmFailure(
                        request_id=req.request_id,
                        error_type="JSONDecodeError",
                        message=f"non-JSON response: {e}",
                        attempts=attempt,
                        status=status,
                        body_snippet=last_body,
                        timeout_s=self.cfg.timeout_s,
                        attempt_timestamps=tuple(attempt_timestamps),
                    )

                try:
                    text = data["choices"][0]["message"]["content"]
                except Exception as e:
                    return MinimaLlmFailure(
                        request_id=req.request_id,
                        error_type="MalformedResponse",
                        message=f"missing expected fields: {e}",
                        attempts=attempt,
                        status=status,
                        body_snippet=last_body,
                        timeout_s=self.cfg.timeout_s,
                        attempt_timestamps=tuple(attempt_timestamps),
                    )

                # Store in cache on success
                if self._cache is not None and cache_key is not None:
                    self._cache.put(cache_key, str(text), data)

                set_last_cached(False)
                return MinimaLlmResponse(request_id=req.request_id, text=str(text), raw=data)

            # non-2xx: parse Retry-After if present
            retry_after = self._parse_retry_after(headers)

            if _is_overload_status(status) or status == 408:
                # Timeout (408) or overload suggests server might be struggling
                await self._cooldown.bump(retry_after or self.cfg.cooldown_floor_s or 1.0)

            if attempt >= self.cfg.max_attempts or not _is_retriable_status(status):
                error_type = "TimeoutError" if status == 408 else "HTTPError"
                return MinimaLlmFailure(
                    request_id=req.request_id,
                    error_type=error_type,
                    message=f"status={status}",
                    attempts=attempt,
                    status=status,
                    body_snippet=last_body,
                    timeout_s=self.cfg.timeout_s,
                    attempt_timestamps=tuple(attempt_timestamps),
                )

            # Honor Retry-After if provided; otherwise use exponential backoff
            if retry_after is not None:
                await asyncio.sleep(retry_after)
            else:
                backoff = min(self.cfg.max_backoff_s, self.cfg.base_backoff_s * (2 ** (attempt - 1)))
                await asyncio.sleep(_jittered(backoff, self.cfg.jitter))


    # ----------------------------
    # Batch runner
    # ----------------------------

    async def run_batched(self, requests: List[MinimaLlmRequest]) -> List[MinimaLlmResult]:
        """
        Execute a batch using the config's batch policy and return results.

        This runs requests concurrently (subject to cfg.max_outstanding) and
        prints a heartbeat for long-running jobs.
        """
        return await run_batched_callable(requests, self.generate, self.cfg.batch)

    async def run_batched_callable(
        self,
        items: List[T],
        async_callable: Callable[[T], Awaitable[R]],
    ) -> List[Union[R, MinimaLlmFailure]]:
        """
        Execute a batch of async calls using the worker pool pattern.

        Convenience method that delegates to the standalone run_batched_callable
        function with this backend's configuration.
        """
        return await run_batched_callable(items, async_callable, self.cfg.batch)
