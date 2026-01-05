# llm_config.py
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional


# ----------------------------
# Helper functions (shared)
# ----------------------------

def _env_str(name: str) -> Optional[str]:
    v = os.getenv(name)
    return None if v in (None, "") else v


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    return default if v is None else int(v)


def _env_float(name: str, default: float) -> float:
    v = _env_str(name)
    return default if v is None else float(v)


def _env_opt_int(name: str, default: Optional[int]) -> Optional[int]:
    v = _env_str(name)
    if v is None:
        return default
    if v.strip().lower() in ("none", "null"):
        return None
    return int(v)


def _first_non_none(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v is not None:
            return v
    return None


# ----------------------------
# Batch configuration
# ----------------------------

@dataclass(frozen=True)
class BatchConfig:
    """
    Configuration for async batch execution with worker pool pattern.

    This config is used by run_batched_callable() and is independent of
    LLM-specific settings. It can be used standalone for generic async batching.

    Parameters
    ----------
    num_workers : int
        Number of concurrent workers in the pool
    max_failures : Optional[int]
        Abort batch after this many failures (None = never abort)
    heartbeat_s : float
        Print progress every N seconds
    stall_s : float
        Warn if no completions for N seconds
    print_first_failures : int
        Print first N failures verbosely
    keep_failure_summaries : int
        Keep last N failure summaries for diagnostics
    """
    num_workers: int = 64
    max_failures: Optional[int] = 25
    heartbeat_s: float = 10.0
    stall_s: float = 300.0
    print_first_failures: int = 5
    keep_failure_summaries: int = 20

    @classmethod
    def from_env(cls) -> "BatchConfig":
        """Load batch configuration from BATCH_* environment variables."""
        return cls(
            num_workers=_env_int("BATCH_NUM_WORKERS", 64),
            max_failures=_env_opt_int("BATCH_MAX_FAILURES", 25),
            heartbeat_s=_env_float("BATCH_HEARTBEAT_S", 10.0),
            stall_s=_env_float("BATCH_STALL_S", 300.0),
            print_first_failures=_env_int("BATCH_PRINT_FIRST_FAILURES", 5),
            keep_failure_summaries=_env_int("BATCH_KEEP_FAILURE_SUMMARIES", 20),
        )


# ----------------------------
# LLM configuration
# ----------------------------

@dataclass(frozen=True)
class MinimaLlmConfig:
    """
    Configuration for the MinimaLlm backend.

    MinimaLlm is a minimal, framework-agnostic adapter for OpenAI-compatible
    LLM endpoints. The configuration is designed to support long-running,
    batch-style workloads (such as evaluation or offline scoring), while
    remaining simple enough for beginners and robust enough for shared
    infrastructure.

    This config intentionally avoids coupling to any specific client
    framework (e.g., LangChain, LiteLLM, DSPy). Advanced users may reuse
    the same environment variables to configure their own tooling.

    Endpoint
    --------
    base_url:
        Base URL of the OpenAI-compatible endpoint.
        Examples:
          - https://api.openai.com/v1
          - http://localhost:8000/v1

    model:
        Model identifier understood by the endpoint.

    api_key:
        Bearer token used for authentication.
        May be None for local or unsecured endpoints; in that case, no
        Authorization header is sent.

    Batch execution and monitoring
    ------------------------------
    num_workers:
        Number of concurrent batch workers used by MinimaLlm's batch runner.
        Workers > max_outstanding is fine; max_outstanding is the real
        concurrency limit at the HTTP layer.

    max_failures:
        Abort the batch after this many failures. Set to None to disable early
        abort. (Parsed from BATCH_MAX_FAILURES where "none"/"null" disables.)

    heartbeat_s:
        Interval in seconds at which batch progress is printed.

    stall_s:
        Emit a stall warning if no completions occur for this many seconds.

    print_first_failures:
        Number of initial failures printed verbosely.

    keep_failure_summaries:
        Number of recent failure summaries retained for abort diagnostics.

    Transport / pacing
    ------------------
    max_outstanding:
        Maximum number of in-flight HTTP requests at any time (hard limit).

    rpm:
        Maximum requests per minute. Implemented as a simple pacing mechanism.
        Set to 0 to disable.

    timeout_s:
        Per-request timeout in seconds.

    Retry and backoff
    -----------------
    max_attempts:
        Maximum number of attempts per request (including initial attempt).
        Set to 0 for infinite retries (will retry forever until success or Ctrl-C).

    base_backoff_s:
        Base delay for exponential backoff (seconds).

    max_backoff_s:
        Upper bound on backoff delay (seconds).

    jitter:
        Proportional random jitter applied to backoff delays (e.g., 0.2 = ±20%).

    Cooldown after overload
    -----------------------
    cooldown_floor_s, cooldown_cap_s, cooldown_halflife_s:
        Parameters controlling a global cooldown after overload signals such as
        HTTP 429/502/503/504. Cooldown decays with the given half-life.

    HTTP
    ----
    compress_gzip:
        If True, request bodies are sent gzip-compressed. Disabled by default,
        since many OpenAI-compatible servers do not support it.
    """

    # Endpoint
    base_url: str
    model: str
    api_key: Optional[str] = None  # optional for local endpoints

    # Batch execution (composed)
    batch: BatchConfig = field(default_factory=BatchConfig)

    # Transport / backpressure
    max_outstanding: int = 32
    rpm: int = 600  # 0 disables pacing
    timeout_s: float = 60.0

    # Retry / backoff
    max_attempts: int = 6
    base_backoff_s: float = 0.5
    max_backoff_s: float = 20.0
    jitter: float = 0.2

    # Cooldown after overload (429/502/503/504)
    cooldown_floor_s: float = 0.0
    cooldown_cap_s: float = 30.0
    cooldown_halflife_s: float = 20.0

    # HTTP
    compress_gzip: bool = False

    # Cache
    cache_dir: Optional[str] = None  # None = disabled

    # ----------------------------
    # Config modification
    # ----------------------------

    def with_model(self, model: str) -> "MinimaLlmConfig":
        """Return a new config with the model replaced."""
        from dataclasses import replace
        return replace(self, model=model)

    # ----------------------------
    # Backward compatibility properties
    # ----------------------------

    @property
    def num_workers(self) -> int:
        return self.batch.num_workers

    @property
    def max_failures(self) -> Optional[int]:
        return self.batch.max_failures

    @property
    def heartbeat_s(self) -> float:
        return self.batch.heartbeat_s

    @property
    def stall_s(self) -> float:
        return self.batch.stall_s

    @property
    def print_first_failures(self) -> int:
        return self.batch.print_first_failures

    @property
    def keep_failure_summaries(self) -> int:
        return self.batch.keep_failure_summaries

    # ----------------------------
    # Config parsing (private)
    # ----------------------------

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "MinimaLlmConfig":
        """
        Construct a MinimaLlmConfig from environment variables.

        Required:
          - OPENAI_BASE_URL
          - OPENAI_MODEL

        Optional:
          - OPENAI_API_KEY or OPENAI_TOKEN
          - BATCH_NUM_WORKERS, BATCH_MAX_FAILURES
          - BATCH_HEARTBEAT_S, BATCH_STALL_S
          - BATCH_PRINT_FIRST_FAILURES, BATCH_KEEP_FAILURE_SUMMARIES
          - MAX_OUTSTANDING, RPM, TIMEOUT_S
          - MAX_ATTEMPTS, BASE_BACKOFF_S, MAX_BACKOFF_S, JITTER
          - COOLDOWN_FLOOR_S, COOLDOWN_CAP_S, COOLDOWN_HALFLIFE_S
          - COMPRESS_GZIP
          - CACHE_DIR
        """
        base_url = _env_str("OPENAI_BASE_URL")
        model = _env_str("OPENAI_MODEL")
        api_key = _first_non_none(
            _env_str("OPENAI_API_KEY"),
            _env_str("OPENAI_TOKEN"),
        )

        missing = []
        if base_url is None:
            missing.append("OPENAI_BASE_URL")
        if model is None:
            missing.append("OPENAI_MODEL")
        if missing:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

        return cls(
            base_url=cls._normalize_base_url(base_url),
            model=model,
            api_key=api_key,
            batch=BatchConfig.from_env(),
            # transport
            max_outstanding=_env_int("MAX_OUTSTANDING", 32),
            rpm=_env_int("RPM", 600),
            timeout_s=_env_float("TIMEOUT_S", 60.0),
            # retry
            max_attempts=_env_int("MAX_ATTEMPTS", 50),
            base_backoff_s=_env_float("BASE_BACKOFF_S", 0.5),
            max_backoff_s=_env_float("MAX_BACKOFF_S", 20.0),
            jitter=_env_float("JITTER", 0.2),
            # cooldown
            cooldown_floor_s=_env_float("COOLDOWN_FLOOR_S", 0.0),
            cooldown_cap_s=_env_float("COOLDOWN_CAP_S", 60.0),
            cooldown_halflife_s=_env_float("COOLDOWN_HALFLIFE_S", 20.0),
            # http
            compress_gzip=(_env_int("COMPRESS_GZIP", 0) != 0),
            # cache
            cache_dir=_env_str("CACHE_DIR"),
        )

    @classmethod
    def from_yaml(cls, path: "Path") -> "MinimaLlmConfig":
        """
        Load MinimaLlmConfig from a YAML file.

        Supports direct config format with base_url and model:

            base_url: "http://localhost:8000/v1"
            model: "llama-3.3-70b-instruct"
            api_key: "optional-key"  # optional

        Args:
            path: Path to the YAML config file

        Returns:
            MinimaLlmConfig instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If required fields (base_url, model) are missing
        """
        import yaml
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        if "base_url" not in data:
            raise ValueError(f"Missing required field 'base_url' in {path}")
        if "model" not in data:
            raise ValueError(f"Missing required field 'model' in {path}")

        return cls(
            base_url=cls._normalize_base_url(data["base_url"]),
            model=data["model"],
            api_key=data.get("api_key", ""),
        )

    # ----------------------------
    # Pickle support (exclude api_key for security)
    # ----------------------------

    def __getstate__(self) -> dict:
        """Exclude api_key from pickled state for security."""
        state = {k: getattr(self, k) for k in self.__dataclass_fields__}
        state.pop("api_key", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state, re-fetching api_key from environment."""
        # Re-fetch api_key from environment variables
        api_key = _first_non_none(
            _env_str("OPENAI_API_KEY"),
            _env_str("OPENAI_TOKEN"),
        )
        # Frozen dataclass requires object.__setattr__
        for key, value in state.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "api_key", api_key)

    def describe(self) -> str:
        """
        Return a human-readable description of the active MinimaLlm configuration.

        Intended for logging at startup of long-running batch jobs so the
        execution parameters are recorded alongside results.
        """
        lines: list[str] = []

        def add(section: str) -> None:
            lines.append(section)

        def kv(k: str, v: object) -> None:
            lines.append(f"  {k}: {v}")

        add("MinimaLlmConfig")
        add("Endpoint")
        kv("base_url", self.base_url)
        kv("model", self.model)
        kv("api_key", "<set>" if self.api_key is not None else "<none>")

        add("Batch execution")
        kv("num_workers", self.num_workers)
        kv("max_failures", self.max_failures)
        kv("heartbeat_s", self.heartbeat_s)
        kv("stall_s", self.stall_s)
        kv("print_first_failures", self.print_first_failures)
        kv("keep_failure_summaries", self.keep_failure_summaries)

        add("Transport / pacing")
        kv("max_outstanding", self.max_outstanding)
        kv("rpm", self.rpm)
        kv("timeout_s", self.timeout_s)

        add("Retry / backoff")
        kv("max_attempts", self.max_attempts)
        kv("base_backoff_s", self.base_backoff_s)
        kv("max_backoff_s", self.max_backoff_s)
        kv("jitter", self.jitter)

        add("Cooldown")
        kv("cooldown_floor_s", self.cooldown_floor_s)
        kv("cooldown_cap_s", self.cooldown_cap_s)
        kv("cooldown_halflife_s", self.cooldown_halflife_s)

        add("HTTP")
        kv("compress_gzip", self.compress_gzip)

        add("Cache")
        kv("cache_dir", self.cache_dir if self.cache_dir else "<disabled>")

        return "\n".join(lines)
