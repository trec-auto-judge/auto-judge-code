# tests/test_minima_llm_multirun.py
"""
Tests for MinimaLLM behavior across multiple asyncio.run() calls.

These tests ensure that the lazy per-loop initialization of async primitives
(Semaphore, Lock) and the cache auto-reopen behavior work correctly when
the backend is reused across separate event loops.

NOTE: These tests do NOT require a real LLM endpoint. They test internal
async primitive behavior and cache handling only - no actual LLM calls are made.

Regression tests for:
- RuntimeError: Semaphore is bound to a different event loop
- ProgrammingError: Cannot operate on a closed database
"""

import asyncio
import pytest

from trec_auto_judge.llm.minima_llm import (
    OpenAIMinimaLlm,
    RpmGate,
    Cooldown,
    PromptCache,
)
from trec_auto_judge.llm.llm_config import MinimaLlmConfig


class TestRpmGateMultiLoop:
    """Test RpmGate lazy lock initialization across event loops."""

    def test_rpm_gate_works_across_asyncio_runs(self):
        """RpmGate should work when used in multiple asyncio.run() calls."""
        gate = RpmGate(rpm=60)

        async def use_gate():
            await gate.wait_turn()
            return True

        # First asyncio.run()
        result1 = asyncio.run(use_gate())
        assert result1 is True

        # Second asyncio.run() - should NOT raise "bound to different event loop"
        result2 = asyncio.run(use_gate())
        assert result2 is True

    def test_rpm_gate_lock_recreated_for_new_loop(self):
        """RpmGate should recreate its lock for a new event loop."""
        gate = RpmGate(rpm=60)

        async def get_lock_id():
            # Force lock creation by calling wait_turn
            await gate.wait_turn()
            return id(gate._lock)

        lock_id_1 = asyncio.run(get_lock_id())
        lock_id_2 = asyncio.run(get_lock_id())

        # Lock should be different objects (recreated for new loop)
        assert lock_id_1 != lock_id_2

    def test_rpm_gate_disabled_works_across_runs(self):
        """RpmGate with rpm=0 (disabled) should work across runs."""
        gate = RpmGate(rpm=0)  # Disabled

        async def use_gate():
            await gate.wait_turn()
            return True

        result1 = asyncio.run(use_gate())
        result2 = asyncio.run(use_gate())
        assert result1 is True
        assert result2 is True


class TestCooldownMultiLoop:
    """Test Cooldown lazy lock initialization across event loops."""

    def test_cooldown_works_across_asyncio_runs(self):
        """Cooldown should work when used in multiple asyncio.run() calls."""
        cooldown = Cooldown(floor_s=0.0, cap_s=10.0, halflife_s=5.0)

        async def use_cooldown():
            await cooldown.wait_if_needed()
            await cooldown.bump(0.1)
            return True

        # First asyncio.run()
        result1 = asyncio.run(use_cooldown())
        assert result1 is True

        # Second asyncio.run() - should NOT raise "bound to different event loop"
        result2 = asyncio.run(use_cooldown())
        assert result2 is True

    def test_cooldown_lock_recreated_for_new_loop(self):
        """Cooldown should recreate its lock for a new event loop."""
        cooldown = Cooldown(floor_s=0.0, cap_s=10.0, halflife_s=5.0)

        async def get_lock_id():
            await cooldown.wait_if_needed()
            return id(cooldown._lock)

        lock_id_1 = asyncio.run(get_lock_id())
        lock_id_2 = asyncio.run(get_lock_id())

        # Lock should be different objects (recreated for new loop)
        assert lock_id_1 != lock_id_2


class TestOpenAIMinimaLlmMultiLoop:
    """Test OpenAIMinimaLlm lazy initialization across event loops.

    NOTE: These tests do NOT make actual LLM calls. They only test
    the internal resource management behavior.
    """

    @pytest.fixture
    def backend_config(self, tmp_path):
        """Create a config with a temporary cache directory.

        Uses dummy URL/key since no actual LLM calls are made.
        """
        return MinimaLlmConfig(
            base_url="http://localhost:9999/v1",  # Dummy - not actually called
            api_key="test-key-not-used",
            model="test-model",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=10,
            rpm=0,  # Disable RPM limiting
        )

    def test_backend_async_resources_recreated_for_new_loop(self, backend_config):
        """Backend should recreate Semaphore for new loop, but RpmGate/Cooldown persist."""
        backend = OpenAIMinimaLlm(backend_config)

        async def get_resource_ids():
            backend._ensure_async_resources()
            # Also trigger RpmGate/Cooldown lock creation
            backend._rpm._ensure_lock()
            backend._cooldown._ensure_lock()
            return {
                "sem": id(backend._sem),
                "rpm": id(backend._rpm),
                "cooldown": id(backend._cooldown),
                "rpm_lock": id(backend._rpm._lock),
                "cooldown_lock": id(backend._cooldown._lock),
                "loop": id(backend._bound_loop),
            }

        ids_1 = asyncio.run(get_resource_ids())
        ids_2 = asyncio.run(get_resource_ids())

        # Semaphore should be recreated for new loop
        assert ids_1["sem"] != ids_2["sem"], "Semaphore should be recreated"
        # RpmGate and Cooldown persist (same object) but their locks are recreated
        assert ids_1["rpm"] == ids_2["rpm"], "RpmGate should persist across loops"
        assert ids_1["cooldown"] == ids_2["cooldown"], "Cooldown should persist across loops"
        assert ids_1["rpm_lock"] != ids_2["rpm_lock"], "RpmGate lock should be recreated"
        assert ids_1["cooldown_lock"] != ids_2["cooldown_lock"], "Cooldown lock should be recreated"
        assert ids_1["loop"] != ids_2["loop"], "Loop reference should change"

    def test_backend_cache_reopens_after_close(self, backend_config):
        """Backend cache should reopen automatically after being closed."""
        backend = OpenAIMinimaLlm(backend_config)

        async def use_close_and_verify():
            # Use cache and write data
            cache = backend._ensure_cache()
            assert cache is not None
            cache.put("test_key", "test_value", None)

            # Close - should set _cache to None
            await backend.aclose()
            assert backend._cache is None, "Cache should be cleared after aclose"

            # Reopen - should work and data should persist
            cache2 = backend._ensure_cache()
            assert cache2 is not None, "Cache should reopen"
            result = cache2.get("test_key")
            assert result is not None, "Data should persist across reopen"
            assert result[0] == "test_value"

            return True

        result = asyncio.run(use_close_and_verify())
        assert result is True

    def test_backend_works_across_asyncio_runs_without_close(self, backend_config):
        """Backend should work across asyncio.run() without explicit close."""
        backend = OpenAIMinimaLlm(backend_config)

        async def ensure_resources():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            # Just verify no exceptions are raised
            return cache is not None

        # Multiple asyncio.run() calls without closing
        result1 = asyncio.run(ensure_resources())
        assert result1 is True

        result2 = asyncio.run(ensure_resources())
        assert result2 is True

        result3 = asyncio.run(ensure_resources())
        assert result3 is True

    def test_semaphore_usable_after_loop_change(self, backend_config):
        """Semaphore should be usable in new event loop without deadlock."""
        backend = OpenAIMinimaLlm(backend_config)

        async def use_semaphore():
            backend._ensure_async_resources()
            # Actually acquire/release the semaphore
            async with backend._sem:
                return True

        # First run
        result1 = asyncio.run(use_semaphore())
        assert result1 is True

        # Second run - this would fail with old code:
        # "RuntimeError: Semaphore is bound to a different event loop"
        result2 = asyncio.run(use_semaphore())
        assert result2 is True


class TestPromptCacheReopen:
    """Test PromptCache behavior with close and reopen pattern."""

    def test_cache_operations_after_reopen(self, tmp_path):
        """Cache should work correctly after being closed and reopened."""
        db_path = str(tmp_path / "test_cache.db")

        # First cache instance - write data
        cache1 = PromptCache(db_path)
        cache1.put("key1", "value1", {"raw": "data1"})
        cache1.close()

        # Second cache instance - read data back
        cache2 = PromptCache(db_path)
        result = cache2.get("key1")
        cache2.close()

        assert result is not None
        assert result[0] == "value1"
        assert result[1] == {"raw": "data1"}

    def test_cache_data_persists_across_reopens(self, tmp_path):
        """Data written before close should be readable after reopen."""
        db_path = str(tmp_path / "persist_cache.db")

        # Write multiple entries
        cache1 = PromptCache(db_path)
        cache1.put("key_a", "value_a", None)
        cache1.put("key_b", "value_b", {"nested": "data"})
        cache1.close()

        # Reopen and verify all data
        cache2 = PromptCache(db_path)
        result_a = cache2.get("key_a")
        result_b = cache2.get("key_b")
        result_missing = cache2.get("nonexistent")
        cache2.close()

        assert result_a == ("value_a", None)
        assert result_b == ("value_b", {"nested": "data"})
        assert result_missing is None


class TestIntegrationMultipleAsyncioRuns:
    """Integration test simulating the rubric judge use case.

    This simulates the exact pattern that caused the original bugs:
    RubricJudge calls asyncio.run() twice - once for nugget generation,
    once for grading - reusing the same backend.

    NOTE: No actual LLM calls are made.
    """

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a backend with temporary cache."""
        cfg = MinimaLlmConfig(
            base_url="http://localhost:9999/v1",  # Dummy
            api_key="test-key",
            model="test-model",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=5,
            rpm=0,
        )
        return OpenAIMinimaLlm(cfg)

    def test_simulated_nuggify_then_judge_workflow(self, backend):
        """
        Simulate the rubric judge workflow that calls asyncio.run() twice:
        1. First for nugget generation
        2. Then for grading

        This is the exact pattern that caused the original bug.
        """

        async def simulate_nuggify():
            """Simulate nugget generation phase."""
            backend._ensure_async_resources()
            cache = backend._ensure_cache()

            # Simulate async work with the backend's resources
            await backend._cooldown.wait_if_needed()
            await backend._rpm.wait_turn()

            # Use semaphore (this is where the original bug manifested)
            async with backend._sem:
                pass

            # Simulate cache write
            if cache:
                cache.put("nuggify_key", "nuggify_result", None)

            return "nuggify_done"

        async def simulate_judge():
            """Simulate grading phase."""
            backend._ensure_async_resources()
            cache = backend._ensure_cache()

            # Simulate async work with the backend's resources
            await backend._cooldown.wait_if_needed()
            await backend._rpm.wait_turn()

            # Use semaphore
            async with backend._sem:
                pass

            # Simulate cache read/write
            if cache:
                cache.put("judge_key", "judge_result", None)

            return "judge_done"

        # Run nuggify phase
        result1 = asyncio.run(simulate_nuggify())
        assert result1 == "nuggify_done"

        # Run judge phase - this should NOT raise any errors
        # Old code would fail here with:
        # "RuntimeError: Semaphore is bound to a different event loop"
        result2 = asyncio.run(simulate_judge())
        assert result2 == "judge_done"

    def test_workflow_with_aclose_between_phases(self, backend):
        """
        Test workflow where aclose() is called between phases.
        This simulates the run_dspy_batch behavior that closes the backend.
        """

        async def phase1_with_close():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            if cache:
                cache.put("phase1", "data1", None)
            await backend.aclose()
            return "phase1_done"

        async def phase2_after_close():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            # Should be able to read data written in phase1
            # Old code would fail here with:
            # "ProgrammingError: Cannot operate on a closed database"
            if cache:
                result = cache.get("phase1")
                cache.put("phase2", "data2", None)
                return result
            return None

        # Phase 1: use and close
        result1 = asyncio.run(phase1_with_close())
        assert result1 == "phase1_done"

        # Phase 2: should work and have access to phase1 data
        result2 = asyncio.run(phase2_after_close())
        assert result2 is not None
        assert result2[0] == "data1"

    def test_three_phase_workflow(self, backend):
        """Test three consecutive asyncio.run() calls."""

        async def phase(n: int):
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            async with backend._sem:
                if cache:
                    cache.put(f"phase{n}", f"data{n}", None)
            return f"phase{n}_done"

        result1 = asyncio.run(phase(1))
        result2 = asyncio.run(phase(2))
        result3 = asyncio.run(phase(3))

        assert result1 == "phase1_done"
        assert result2 == "phase2_done"
        assert result3 == "phase3_done"

        # Verify all data was written
        async def verify():
            cache = backend._ensure_cache()
            return [cache.get(f"phase{n}") for n in [1, 2, 3]]

        results = asyncio.run(verify())
        assert all(r is not None for r in results)
        assert [r[0] for r in results] == ["data1", "data2", "data3"]


class TestBatchRunnerMultiLoop:
    """Test run_batched_callable across multiple asyncio.run() calls.

    This tests the batch execution infrastructure without needing an LLM.
    """

    def test_run_batched_callable_across_asyncio_runs(self, tmp_path):
        """run_batched_callable should work across multiple asyncio.run() calls."""
        from trec_auto_judge.llm.minima_llm import run_batched_callable
        from trec_auto_judge.llm.llm_config import BatchConfig

        batch_config = BatchConfig(
            num_workers=2,
            heartbeat_s=0,  # Disable heartbeat for tests
            stall_s=0,
            max_failures=None,
        )

        async def dummy_callable(item: int) -> int:
            await asyncio.sleep(0.001)  # Tiny delay to exercise async
            return item * 2

        async def run_batch(items):
            return await run_batched_callable(items, dummy_callable, batch_config)

        # First asyncio.run()
        results1 = asyncio.run(run_batch([1, 2, 3]))
        assert results1 == [2, 4, 6]

        # Second asyncio.run() - should work without errors
        results2 = asyncio.run(run_batch([10, 20, 30]))
        assert results2 == [20, 40, 60]

    def test_backend_run_batched_callable_across_runs(self, tmp_path):
        """Backend's run_batched_callable method should work across runs."""
        cfg = MinimaLlmConfig(
            base_url="http://localhost:9999/v1",
            api_key="test",
            model="test",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=5,
            rpm=0,
        )
        backend = OpenAIMinimaLlm(cfg)

        async def dummy_callable(item: str) -> str:
            # Use backend resources to exercise the lazy init
            backend._ensure_async_resources()
            async with backend._sem:
                await asyncio.sleep(0.001)
            return f"processed_{item}"

        async def run_batch(items):
            return await backend.run_batched_callable(items, dummy_callable)

        # First run
        results1 = asyncio.run(run_batch(["a", "b"]))
        assert set(results1) == {"processed_a", "processed_b"}

        # Second run - exercises the lazy reinit
        results2 = asyncio.run(run_batch(["x", "y", "z"]))
        assert set(results2) == {"processed_x", "processed_y", "processed_z"}


class TestDspyAdapterMultiLoop:
    """Test DSPy adapter (run_dspy_batch) across multiple asyncio.run() calls.

    This simulates the exact pattern from rubric_autojudge.py where
    run_dspy_batch is called twice via asyncio.run().

    Mocking strategy: We patch MinimaLlmDSPyLM.acall to intercept calls
    at the DSPy adapter layer. This exercises the full batch runner and
    DSPy parsing infrastructure without making actual HTTP calls.
    """

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a backend with temporary cache."""
        cfg = MinimaLlmConfig(
            base_url="http://localhost:9999/v1",
            api_key="test",
            model="test",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=5,
            rpm=0,
        )
        return OpenAIMinimaLlm(cfg)

    def test_mocked_dspy_batch_across_asyncio_runs(self, backend):
        """
        Test run_dspy_batch pattern across multiple asyncio.run() calls.

        This mocks the LLM call but exercises the full DSPy adapter path
        including MinimaLlmDSPyLM and run_dspy_batch.
        """
        dspy = pytest.importorskip("dspy")
        from pydantic import BaseModel
        from typing import Optional
        from unittest.mock import patch

        from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch, MinimaLlmDSPyLM

        # Simple signature for testing
        class SimpleSignature(dspy.Signature):
            input_text: str = dspy.InputField()
            output_text: str = dspy.OutputField()

        # Annotation model (combined input/output like rubric judge)
        class SimpleAnnotation(BaseModel):
            input_text: str
            output_text: Optional[str] = None

        def convert_output(pred, obj: SimpleAnnotation):
            obj.output_text = pred.output_text

        call_count = 0

        async def mock_acall(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return valid DSPy-formatted response (ChainOfThought requires reasoning)
            return ["[[ ## reasoning ## ]]\nThis is the reasoning.\n[[ ## output_text ## ]]\nmocked_response"]

        # Run twice with mocked LLM
        with patch.object(MinimaLlmDSPyLM, 'acall', mock_acall):
            # First asyncio.run() - simulates nugget generation
            annotations1 = [SimpleAnnotation(input_text="test1")]

            async def batch1():
                return await run_dspy_batch(
                    SimpleSignature,
                    annotations1,
                    convert_output,
                    backend=backend
                )

            results1 = asyncio.run(batch1())
            assert results1[0].output_text == "mocked_response"
            first_run_calls = call_count

            # Second asyncio.run() - simulates grading
            # This is where the original bug manifested
            annotations2 = [
                SimpleAnnotation(input_text="test2a"),
                SimpleAnnotation(input_text="test2b"),
            ]

            async def batch2():
                return await run_dspy_batch(
                    SimpleSignature,
                    annotations2,
                    convert_output,
                    backend=backend
                )

            results2 = asyncio.run(batch2())
            assert len(results2) == 2
            assert all(r.output_text == "mocked_response" for r in results2)

            # Verify both batches actually ran
            assert call_count > first_run_calls

    def test_simulated_rubric_judge_pattern(self, backend):
        """
        Simulate the exact rubric_autojudge.py pattern:
        1. asyncio.run(run_dspy_batch(...)) for GenerateNuggetQuestions
        2. asyncio.run(run_dspy_batch(...)) for GradeNuggetAnswer

        This is the pattern that triggered the original bugs.
        """
        dspy = pytest.importorskip("dspy")
        from pydantic import BaseModel
        from typing import Optional
        from unittest.mock import patch

        from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch, MinimaLlmDSPyLM

        # Signatures mimicking rubric judge
        class GenerateQuestions(dspy.Signature):
            """Generate questions from a query."""
            query_title: str = dspy.InputField()
            questions: str = dspy.OutputField()

        class GradeAnswer(dspy.Signature):
            """Grade how well a passage answers a question."""
            question: str = dspy.InputField()
            passage: str = dspy.InputField()
            grade: str = dspy.OutputField()

        # Annotation models (combined input/output)
        class QuestionGenData(BaseModel):
            query_title: str
            questions: Optional[str] = None

        class GradeData(BaseModel):
            question: str
            passage: str
            grade: Optional[int] = None

        def convert_questions(pred, obj: QuestionGenData):
            obj.questions = pred.questions

        def convert_grade(pred, obj: GradeData):
            obj.grade = int(pred.grade) if pred.grade.isdigit() else 0

        phase = {"current": ""}

        async def mock_acall(self, *args, **kwargs):
            # ChainOfThought requires reasoning field
            if phase["current"] == "nuggify":
                return ["[[ ## reasoning ## ]]\nGenerating questions.\n[[ ## questions ## ]]\nWhat is X?\nWhat is Y?"]
            else:
                return ["[[ ## reasoning ## ]]\nEvaluating answer.\n[[ ## grade ## ]]\n4"]

        with patch.object(MinimaLlmDSPyLM, 'acall', mock_acall):
            # Phase 1: Nugget generation (like create_nuggets)
            phase["current"] = "nuggify"
            gen_data = [
                QuestionGenData(query_title="Topic 1"),
                QuestionGenData(query_title="Topic 2"),
            ]

            async def nuggify():
                return await run_dspy_batch(
                    GenerateQuestions,
                    gen_data,
                    convert_questions,
                    backend=backend
                )

            # This is the first asyncio.run()
            gen_results = asyncio.run(nuggify())
            assert len(gen_results) == 2
            assert all(r.questions is not None for r in gen_results)

            # Phase 2: Grading (like judge)
            phase["current"] = "judge"
            grade_data = [
                GradeData(question="What is X?", passage="X is something."),
                GradeData(question="What is Y?", passage="Y is another thing."),
                GradeData(question="What is Z?", passage="Z is unknown."),
            ]

            async def judge():
                return await run_dspy_batch(
                    GradeAnswer,
                    grade_data,
                    convert_grade,
                    backend=backend
                )

            # This is the second asyncio.run() - where original bug occurred
            grade_results = asyncio.run(judge())
            assert len(grade_results) == 3
            assert all(r.grade == 4 for r in grade_results)

    def test_three_consecutive_dspy_batches(self, backend):
        """Test three consecutive asyncio.run(run_dspy_batch(...)) calls."""
        dspy = pytest.importorskip("dspy")
        from pydantic import BaseModel
        from typing import Optional
        from unittest.mock import patch

        from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch, MinimaLlmDSPyLM

        class Echo(dspy.Signature):
            input_val: str = dspy.InputField()
            output_val: str = dspy.OutputField()

        class EchoModel(BaseModel):
            input_val: str
            output_val: Optional[str] = None

        def convert(pred, obj):
            obj.output_val = pred.output_val

        batch_num = {"n": 0}

        async def mock_acall(self, *args, **kwargs):
            # ChainOfThought requires reasoning field
            return [f"[[ ## reasoning ## ]]\nProcessing batch {batch_num['n']}.\n[[ ## output_val ## ]]\nbatch_{batch_num['n']}"]

        with patch.object(MinimaLlmDSPyLM, 'acall', mock_acall):
            for i in range(3):
                batch_num["n"] = i
                data = [EchoModel(input_val=f"input_{i}")]

                async def run():
                    return await run_dspy_batch(Echo, data, convert, backend=backend)

                results = asyncio.run(run())
                assert results[0].output_val == f"batch_{i}"