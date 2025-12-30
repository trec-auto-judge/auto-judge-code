# llm_protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

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


@runtime_checkable
class AsyncMinimaLlmBackend(Protocol):
    """
    Minimal async backend interface.

    The harness and DSPy adapter only rely on these methods. Backends can
    implement rate limiting, retries, and backpressure internally.
    """

    async def generate(self, req: MinimaLlmRequest) -> MinimaLlmResponse:
        """Perform one LLM call and return the generated text."""
        ...

    async def prompt_one(self, req: MinimaLlmRequest) -> MinimaLlmResponse:
        """
        Backwards-compatible alias for `generate`.

        New code should call `generate`. This method exists to avoid churn in
        older examples/tests.
        """
        ...

    async def aclose(self) -> None:
        """Release any backend resources (sessions, files)."""
        ...
