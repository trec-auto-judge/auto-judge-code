# llm_protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

Json = Dict[str, Any]


@dataclass(frozen=True)
class MinimaLlmRequest:
    """
    Minimal request shape for a single LLM call.

    This stays intentionally stdlib-only so that participants can use their own
    frameworks (DSPy, LangChain, LiteLLM, raw HTTP, etc.) without dependency
    conflicts.

    Parameters
    ----------
    request_id:
        Stable identifier for this request (used for logging and error reporting).

    messages:
        OpenAI-compatible chat message list. Each message is a dict with keys:
        - role: "system" | "user" | "assistant" | ...
        - content: string

    temperature, max_tokens:
        Standard generation knobs. If None, the endpoint default is used.

    extra:
        Additional OpenAI-compatible parameters to forward verbatim (e.g., stop,
        top_p). Keep this policy-free and backend-agnostic.
    """

    request_id: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Optional[Json] = None


@dataclass(frozen=True)
class MinimaLlmResponse:
    """Result of a successful LLM call."""

    request_id: str
    text: str
    raw: Optional[Json] = None
    cached: bool = False  # True if returned from prompt cache



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


MinimaLlmResult = Union[MinimaLlmResponse, MinimaLlmFailure]




@runtime_checkable
class AsyncMinimaLlmBackend(Protocol):
    """
    Minimal async backend interface.

    The harness and DSPy adapter only rely on these methods. Backends can
    implement rate limiting, retries, and backpressure internally.
    """

    async def generate(self, req: MinimaLlmRequest) -> MinimaLlmResult:
        """Perform one LLM call and return the generated text."""
        ...


    async def aclose(self) -> None:
        """Release any backend resources (sessions, files)."""
        ...
