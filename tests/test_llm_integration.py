"""
Integration tests for MinimaLLM core functionality.

These tests verify the code doesn't crash when used normally and all responses
are returned correctly. They make real LLM calls using environment-based config.

Prerequisites:
- OPENAI_BASE_URL must be set
- OPENAI_MODEL must be set
- OPENAI_API_KEY (or OPENAI_TOKEN) must be set
"""

import pytest
import os 
from trec_auto_judge.llm import (
    BatchConfig,
    MinimaLlmConfig,
    MinimaLlmRequest,
    MinimaLlmResponse,
    OpenAIMinimaLlm,
)

SKIP_LLM_ENDPOINT_TESTS = os.getenv("SKIP_LLM_ENDPOINT_TESTS", "false").lower() in {"1", "true", "yes"}

@pytest.mark.skipif(
    SKIP_LLM_ENDPOINT_TESTS,
    reason="Skipping LLM endpoint tests (SKIP_LLM_ENDPOINT_TESTS set)"
)
@pytest.fixture(autouse=True)
def _prep_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BATCH_NUM_WORKERS", "16")
    monkeypatch.setenv("MAX_OUTSTANDING", "8")
    monkeypatch.setenv("RPM", "0")
    monkeypatch.setenv("BATCH_MAX_FAILURES", "20")
    monkeypatch.setenv("BATCH_HEARTBEAT_S", "10.0")
    monkeypatch.setenv("TIMEOUT_S", "1200.0")
    
    cache_dir = tmp_path / "minimallm-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CACHE_DIR", str(cache_dir))
    print("Set environment, CACHE_DIR =", os.environ["CACHE_DIR"])

@pytest.mark.skipif(
    SKIP_LLM_ENDPOINT_TESTS,
    reason="Skipping LLM endpoint tests (SKIP_LLM_ENDPOINT_TESTS set)"
)
@pytest.mark.asyncio
async def test_single_request():
    """Test single LLM request doesn't crash"""
    llm = OpenAIMinimaLlm.from_env()
    print("lm config = ", llm.cfg)

    req = MinimaLlmRequest(
        request_id="test-1",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    response = await llm.generate(req)

    assert isinstance(response, MinimaLlmResponse)
    assert response.request_id == "test-1"
    assert len(response.text) > 0

    await llm.aclose()

@pytest.mark.skipif(
    SKIP_LLM_ENDPOINT_TESTS,
    reason="Skipping LLM endpoint tests (SKIP_LLM_ENDPOINT_TESTS set)"
)
@pytest.mark.asyncio
async def test_batch_execution():
    """Test parallel batch execution returns all responses"""
    llm = OpenAIMinimaLlm.from_env()

    requests = [
        MinimaLlmRequest(request_id="req-1", messages=[{"role": "user", "content": "Say hello"}]),
        MinimaLlmRequest(request_id="req-2", messages=[{"role": "user", "content": "Count to 3"}]),
        MinimaLlmRequest(request_id="req-3", messages=[{"role": "user", "content": "Name a color"}]),
    ]

    results = await llm.run_batched(requests)

    # All responses returned
    assert len(results) == 3

    # All are successes (not failures)
    for result in results:
        assert isinstance(result, MinimaLlmResponse), f"Got failure: {result}"

    # Request IDs preserved
    returned_ids = {r.request_id for r in results}
    expected_ids = {"req-1", "req-2", "req-3"}
    assert returned_ids == expected_ids

    await llm.aclose()


def test_config_loading():
    """Test configuration loads from environment"""
    llm_config = MinimaLlmConfig.from_env()
    batch_config = BatchConfig.from_env()

    # Basic sanity checks
    assert llm_config.base_url
    assert llm_config.model
    assert llm_config.batch is not None
    assert batch_config.num_workers > 0

@pytest.mark.skipif(
    SKIP_LLM_ENDPOINT_TESTS,
    reason="Skipping LLM endpoint tests (SKIP_LLM_ENDPOINT_TESTS set)"
)
@pytest.mark.asyncio
async def test_prompt_cache(tmp_path):
    """Test SQLite prompt cache works correctly"""
    import os

    cache_dir = str(tmp_path / "cache")

    # Create config with caching enabled
    base_config = MinimaLlmConfig.from_env()
    config = MinimaLlmConfig(
        base_url=base_config.base_url,
        model=base_config.model,
        api_key=base_config.api_key,
        cache_dir=cache_dir,
    )

    llm = OpenAIMinimaLlm(config)

    req = MinimaLlmRequest(
        request_id="cache-test-1",
        messages=[{"role": "user", "content": "Say the word 'cached'"}],
    )

    # First request - makes actual LLM call
    response1 = await llm.generate(req)
    assert isinstance(response1, MinimaLlmResponse)

    # Verify cache file was created
    db_path = os.path.join(cache_dir, "minima_llm.db")
    assert os.path.exists(db_path), "Cache DB should be created"

    # Second request with same content - should return from cache
    req2 = MinimaLlmRequest(
        request_id="cache-test-2",  # Different request_id, same content
        messages=[{"role": "user", "content": "Say the word 'cached'"}],
    )
    response2 = await llm.generate(req2)

    # Same text response (from cache), but request_id should be the new one
    assert response2.text == response1.text, "Cached response should match"
    assert response2.request_id == "cache-test-2", "Request ID should be from new request"

    await llm.aclose()


@pytest.mark.skipif(
    SKIP_LLM_ENDPOINT_TESTS,
    reason="Skipping LLM endpoint tests (SKIP_LLM_ENDPOINT_TESTS set)"
)
@pytest.mark.asyncio
async def test_dspy_adapter_parse_error_retry():
    """Test that AdapterParseError triggers retry with force_refresh=True.

    This tests the actual run_dspy_batch function with a mocked predictor that
    fails on first call, then verifies force_refresh is True on retry.
    """
    dspy = pytest.importorskip("dspy")
    # import dspy
    from pydantic import BaseModel
    from typing import Optional
    from unittest.mock import patch

    from trec_auto_judge.llm import get_force_refresh
    from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch, MinimaLlmDSPyLM

    # Simple signature for testing
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Annotation model
    class QAAnnotation(BaseModel):
        question: str
        answer: Optional[str] = None

    def convert_output(pred, obj: QAAnnotation):
        obj.answer = pred.answer

    # Track state across calls
    call_count = 0  # Our actual attempts (excludes DSPy internal JSON fallback)
    json_fallback_count = 0  # DSPy's internal JSON format retry
    force_refresh_on_retry = None
    call_log = []  # Collect debug info for assertion message

    # Create backend
    backend = OpenAIMinimaLlm.from_env()

    async def mock_lm_acall(self, *args, **kwargs):
        nonlocal call_count, json_fallback_count, force_refresh_on_retry
        current_force_refresh = get_force_refresh()

        # Check if this is DSPy's JSON format fallback (internal retry, not ours)
        messages = kwargs.get('messages', [])
        last_content = messages[-1].get('content', '') if messages else ''
        is_json_fallback = 'JSON object' in last_content

        if is_json_fallback:  # DSPy may automatically fall back onto JSON when parsing fails. Let's catch this and toss it.
            json_fallback_count += 1
            if json_fallback_count == 1:
                # First JSON fallback also fails - forces DSPy to raise AdapterParseError to us
                response = ['{"not_valid": "missing required fields"}']
            else:
                # Subsequent JSON fallbacks succeed
                response = ['{"reasoning": "Because 2+2=4", "answer": "42"}']
        else:
            call_count += 1
            if call_count == 1:
                response = ["This response has no DSPy field markers so parsing will fail"]
            else:
                force_refresh_on_retry = current_force_refresh
                response = ["[[ ## reasoning ## ]]\nBecause 2+2=4\n[[ ## answer ## ]]\n42"]

        call_log.append({
            "call": call_count,
            "json_fallback": json_fallback_count,
            "is_json": is_json_fallback,
            "force_refresh": current_force_refresh,
            "response": response[0][:80],
        })
        return response


    # Patch at class level since run_dspy_batch creates its own LM instance
    with patch.object(MinimaLlmDSPyLM, 'acall', mock_lm_acall):
        annotations = [QAAnnotation(question="What is 2+2?")]

        # This should: 1) call LM, 2) fail to parse, 3) retry with force_refresh=True
        results = await run_dspy_batch(
            QA,
            annotations,
            convert_output,
            backend=backend
        )

    # Verify retry happened and force_refresh was set
    assert call_count == 2, f"Expected 2 LM calls (1 fail + 1 retry), got {call_count}. Call log:\n{call_log}"
    assert force_refresh_on_retry is True, f"force_refresh should be True on retry call. Call log:\n{call_log}"

    # Verify result was processed
    assert results[0].answer == "42"

    # Verify force_refresh was reset after retry completed (finally block)
    assert get_force_refresh() is False, "force_refresh should be reset after call completes"

    await backend.aclose()
