# minimal_llm.py
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
from .llm_protocol import AsyncMinimaLlmBackend, Json, MinimaLlmRequest, MinimaLlmResponse

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class MinimaLlmFailure:
    request_id: str
    error_type: str
    message: str
    attempts: int
    status: Optional[int] = None
    body_snippet: Optional[str] = None
    timeout_s: Optional[float] = None
    attempt_timestamps: Tuple[float, ...] = ()

    def format_attempts(self) -> str:
        """Format attempt timestamps relative to first attempt."""
        if not self.attempt_timestamps:
            return ""
        t0 = self.attempt_timestamps[0]
        times = [f"+{t - t0:.1f}s" for t in self.attempt_timestamps]
        return f"[{', '.join(times)}]"


Result = Union[MinimaLlmResponse, MinimaLlmFailure]


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
    def __init__(self, *, interval_seconds: float, stall_timeout_seconds: float) -> None:
        self._every_s = float(interval_seconds)
        self._stall_s = float(stall_timeout_seconds)
        self._start = time.monotonic()
        self._last_done = self._start
        self._last_print = self._start

    def mark_done(self) -> None:
        self._last_done = time.monotonic()

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

    def maybe_print(self, *, done: int, total: int, failed: int) -> None:
        now = time.monotonic()

        if self._every_s > 0 and (now - self._last_print) >= self._every_s:
            elapsed = now - self._start

            # ETA estimation
            if done > 0 and elapsed > 0:
                rate = done / elapsed  # items per second
                remaining = max(0, total - done)
                eta_s = remaining / rate if rate > 0 else float("inf")
                eta_str = self._fmt_eta(eta_s)
            else:
                eta_str = "?"

            print(
                f"[{elapsed:7.1f}s] "
                f"completed={done}/{total} "
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


# class _Heartbeat:
#     def __init__(self, *, interval_seconds: float, stall_timeout_seconds: float) -> None:
#         self._every_s = float(interval_seconds)
#         self._stall_s = float(stall_timeout_seconds)
#         self._start = time.monotonic()
#         self._last_done = self._start
#         self._last_print = self._start

#     def mark_done(self) -> None:
#         self._last_done = time.monotonic()

#     def maybe_print(self, *, done: int, total: int, failed: int) -> None:
#         now = time.monotonic()
#         if self._every_s > 0 and (now - self._last_print) >= self._every_s:
#             elapsed = now - self._start           
#             print(f"[{elapsed:7.1f}s] completed={done}/{total} failed={failed}")
#             self._last_print = now

#         if self._stall_s > 0 and (now - self._last_done) >= self._stall_s:
#             elapsed = now - self._start
#             print(f"[{elapsed:7.1f}s] WARNING: no completions for {now - self._last_done:.1f}s")
#             self._last_done = now  # avoid spamming


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
    
    hb = _Heartbeat(interval_seconds=batch_config.heartbeat_s, stall_timeout_seconds=batch_config.stall_s)
    fc = _FailureCollector(
        print_first_n=batch_config.print_first_failures,
        keep_last_n=batch_config.keep_failure_summaries,
    )

    total = len(items)
    results: List[Optional[Union[R, MinimaLlmFailure]]] = [None] * total
    q: asyncio.Queue[Tuple[int, T]] = asyncio.Queue()

    for i, item in enumerate(items):
        q.put_nowait((i, item))

    async def worker() -> None:
        while True:
            try:
                i, item = q.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                result = await async_callable(item)
                # Check if result is a failure (generate() returns Result, not raises)
                if isinstance(result, MinimaLlmFailure):
                    fc.record(result)
                results[i] = result
            except Exception as e:
                # Fallback for non-LLM exceptions
                f = MinimaLlmFailure(
                    request_id=f"input-{i}",
                    error_type=type(e).__name__,
                    message=str(e),
                    attempts=1,
                )
                results[i] = f
                fc.record(f)

            hb.mark_done()
            q.task_done()

    async def heartbeat_loop() -> None:
        while True:
            done = sum(1 for x in results if x is not None)
            hb.maybe_print(done=done, total=total, failed=fc.count)
            if done >= total:
                return
            await asyncio.sleep(0.5)

    workers = [asyncio.create_task(worker()) for _ in range(max(1, int(batch_config.num_workers)))]
    hb_task = asyncio.create_task(heartbeat_loop())

    await asyncio.gather(*workers)
    await hb_task

    # Early-abort policy
    if batch_config.max_failures is not None and fc.count > batch_config.max_failures:
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

    This intentionally stays dependency-light; advanced users can swap it out.
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
            self._cache.close()

    def _endpoint(self, path: str) -> str:
        # If base_url already ends in /v1, avoid duplicating /v1.
        if self._has_v1 and path.startswith("/v1/"):
            path = path[len("/v1") :]
        return self._base.rstrip("/") + path

    def _headers(self, *, body_is_gzip: bool) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
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

    async def generate(self, req: MinimaLlmRequest) -> Result:
        """
        Generate a response for the given request.

        Returns MinimaLlmResponse on success, MinimaLlmFailure on error.
        Failures include full retry context: attempt count, timestamps, timeout.
        """
        # Check cache first
        cache_key: Optional[str] = None
        if self._cache is not None:
            cache_key = self._make_cache_key(req)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return MinimaLlmResponse(request_id=req.request_id, text=cached[0], raw=cached[1])

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

    async def prompt_one(self, req: MinimaLlmRequest) -> Result:
        """Backwards-compatible alias for generate()."""
        return await self.generate(req)

    # ----------------------------
    # Batch runner
    # ----------------------------

    async def run_batched(self, requests: List[MinimaLlmRequest]) -> List[Result]:
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
