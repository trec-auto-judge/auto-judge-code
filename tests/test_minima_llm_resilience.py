# tests/test_minima_llm_resilience.py
"""
Tests for MinimaLLM 502 error resilience and infinite retry behavior.

These tests verify:
- 502 is treated as an overload status (triggers cooldown)
- max_attempts=0 allows infinite retries
- Overload warning printed once per request
- Recovery message printed after overload recovery
"""

import asyncio
from dataclasses import replace
from unittest.mock import patch
import pytest

from trec_auto_judge.llm.minima_llm import (
    OpenAIMinimaLlm,
    _is_overload_status,
    _is_retriable_status,
)
from trec_auto_judge.llm.llm_config import MinimaLlmConfig
from trec_auto_judge.llm.llm_protocol import MinimaLlmRequest, MinimaLlmResponse, MinimaLlmFailure


class TestOverloadStatusCodes:
    """Test that 502 is properly classified as overload status."""

    def test_502_is_overload_status(self):
        """502 Bad Gateway should be considered an overload status."""
        assert _is_overload_status(502) is True

    def test_502_is_retriable_status(self):
        """502 Bad Gateway should be retriable."""
        assert _is_retriable_status(502) is True

    def test_overload_status_codes(self):
        """All expected overload status codes should return True."""
        overload_codes = [429, 502, 503, 504]
        for code in overload_codes:
            assert _is_overload_status(code) is True, f"{code} should be overload status"

    def test_non_overload_status_codes(self):
        """Non-overload codes should return False."""
        non_overload = [200, 400, 401, 403, 404, 500]
        for code in non_overload:
            assert _is_overload_status(code) is False, f"{code} should not be overload status"


@pytest.fixture
def base_config():
    """Load config from environment, override timeouts for fast tests."""
    cfg = MinimaLlmConfig.from_env()
    return replace(
        cfg,
        max_outstanding=1,
        timeout_s=1.0,
        base_backoff_s=0.01,
        max_backoff_s=0.02,
        cooldown_floor_s=0.01,
        cooldown_cap_s=0.05,
        cooldown_halflife_s=1.0,
        cache_dir=None,  # Disable cache for tests
    )


class TestInfiniteRetries:
    """Test max_attempts=0 for infinite retries."""

    def test_infinite_retries_eventually_succeeds(self, base_config):
        """With max_attempts=0, retries should continue until success."""
        config = replace(base_config, max_attempts=0)
        backend = OpenAIMinimaLlm(config)

        call_count = 0

        async def mock_post_json(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return (502, {}, b"Bad Gateway")
            else:
                return (200, {}, b'{"choices":[{"message":{"content":"test response"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                result = await backend.generate(req)
                return result, call_count

        result, calls = asyncio.run(run_test())

        assert isinstance(result, MinimaLlmResponse)
        assert result.text == "test response"
        assert calls == 3  # Two failures, one success

    def test_limited_retries_fails_after_max(self, base_config):
        """With max_attempts=3, should fail after 3 attempts."""
        config = replace(base_config, max_attempts=3)
        backend = OpenAIMinimaLlm(config)

        async def mock_post_json(url, payload):
            return (502, {}, b"Bad Gateway")

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                result = await backend.generate(req)
                return result

        result = asyncio.run(run_test())

        assert isinstance(result, MinimaLlmFailure)
        assert result.attempts == 3
        assert "502" in result.message


class TestOverloadWarnings:
    """Test overload warning and recovery messages."""

    def test_overload_warning_printed_once(self, base_config, capsys):
        """Overload warning should be printed only once per request."""
        config = replace(base_config, max_attempts=5)
        backend = OpenAIMinimaLlm(config)

        call_count = 0

        async def mock_post_json(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return (502, {}, b"Bad Gateway")
            else:
                return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req)

        asyncio.run(run_test())

        captured = capsys.readouterr()
        assert captured.out.count("Server overload") == 1
        assert "HTTP 502" in captured.out
        assert "Ctrl-C" in captured.out

    def test_recovery_message_printed(self, base_config, capsys):
        """Recovery message should be printed after overload recovery."""
        config = replace(base_config, max_attempts=5)
        backend = OpenAIMinimaLlm(config)

        call_count = 0

        async def mock_post_json(url, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (502, {}, b"Bad Gateway")
            else:
                return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req)

        asyncio.run(run_test())

        captured = capsys.readouterr()
        assert "Server recovered" in captured.out

    def test_no_warning_on_immediate_success(self, base_config, capsys):
        """No warning should be printed if request succeeds immediately."""
        config = replace(base_config, max_attempts=5)
        backend = OpenAIMinimaLlm(config)

        async def mock_post_json(url, payload):
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req)

        asyncio.run(run_test())

        captured = capsys.readouterr()
        assert "Server overload" not in captured.out
        assert "Server recovered" not in captured.out


class TestCooldownOnOverload:
    """Test that 502 triggers cooldown bump."""

    def test_502_triggers_cooldown_bump(self, base_config):
        """502 should trigger cooldown.bump() call."""
        config = replace(
            base_config,
            max_attempts=2,
            cooldown_floor_s=0.1,
            cooldown_cap_s=1.0,
            cooldown_halflife_s=10.0,
        )
        backend = OpenAIMinimaLlm(config)

        bump_calls = []
        original_bump = backend._cooldown.bump

        async def tracking_bump(suggested_s):
            bump_calls.append(suggested_s)
            await original_bump(suggested_s)

        async def mock_post_json(url, payload):
            return (502, {}, b"Bad Gateway")

        async def run_test():
            backend._cooldown.bump = tracking_bump
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req)

        asyncio.run(run_test())

        # Should have bumped cooldown for each 502 response
        assert len(bump_calls) == 2  # max_attempts=2