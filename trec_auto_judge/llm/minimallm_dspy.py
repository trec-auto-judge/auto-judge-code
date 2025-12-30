# minimallm_dspy.py
from __future__ import annotations

import asyncio
import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast, get_args

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.exceptions import AdapterParseError
import re

from pydantic import BaseModel

from .llm_protocol import MinimaLlmRequest
from .minimal_llm import MinimaLlmFailure, OpenAIMinimaLlm


# ====== More tolerant chat adapter ========


class TolerantChatAdapter(ChatAdapter):
    # Matches a well-formed header anywhere in a line, e.g. [[ ## answerability ## ]]
    # Group captures the raw field name between the ## ... ## markers.
    _HEADER_RE = re.compile(
        r"\[\[\s*##\s*(?P<name>[^#\]\r\n]+?)\s*##\s*\]\]",
        flags=re.IGNORECASE,
    )

    @classmethod
    def normalize_field_name(cls, raw: str) -> str:
        # Mirror your old normalization: lower + spaces to underscores
        return raw.strip().lower().replace(" ", "_")

    @classmethod
    def is_optional_type(cls, tp):
        """General helper: returns True if annotation is Optional[...]"""
        return (
            getattr(tp, "__origin__", None) is typing.Union
            and type(None) in getattr(tp, "__args__", ())
        )

    @classmethod
    def is_optional_float(cls, ann):
        """Return True if annotation is Optional[float] or Union[float, NoneType]."""
        return cls.is_optional_type(ann) and float in typing.get_args(ann)

    @classmethod
    def try_parse_float(cls, val):
        """Safely parse float, or return None if invalid."""
        try:
            return float(str(val).strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _is_non_value(cls, s: str) -> bool:
        return s.strip().lower() in {"", "none", "null"}

    def parse(self, signature, completion: str):
        # 1) Stream-parse into sections, allowing headers anywhere in the line.
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        current_k, current_lines = sections[-1]

        def push_text(txt: str):
            txt = txt.strip()
            if txt:
                current_lines.append(txt)

        for raw_line in completion.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            last_end = 0
            for m in self._HEADER_RE.finditer(line):
                # Text before the header belongs to the current section
                before = line[last_end : m.start()]
                if before:
                    push_text(before)

                # Start a new section at this header
                header = self.normalize_field_name(m.group("name"))
                sections.append((header, []))
                current_k, current_lines = sections[-1]

                last_end = m.end()

            # Trailing text after the last header belongs to the current section
            after = line[last_end:]
            if after:
                push_text(after)

        # 2) Reduce sections into {field: value} for known output fields.
        parsed: dict[str, typing.Any] = {}
        for k, lines in sections:
            if k in signature.output_fields:
                val = "\n".join(lines).strip()
                if self._is_non_value(val):
                    continue
                parsed[k] = val  # last occurrence wins

        # 3) Fill missing fields + coerce Optional[float].
        for name, field in signature.output_fields.items():
            annotation = field.annotation

            if name in parsed:
                val = parsed[name]
                if self.is_optional_float(annotation):
                    parsed[name] = self.try_parse_float(val)
            else:
                if self.is_optional_type(annotation):
                    parsed[name] = None
                else:
                    raise AdapterParseError(
                        adapter_name="TolerantChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        parsed_result=parsed,
                        message=f"Missing required field: {name}",
                    )

        return parsed

    
dspy.settings.configure(adapter=TolerantChatAdapter())



# ==============

def _resolve_dspy_base_lm() -> Type[Any]:
    """
    Locate DSPy's BaseLM class across common DSPy layouts.

    DSPy moves internals occasionally; this helper keeps the adapter resilient.
    """
    if hasattr(dspy, "BaseLM"):
        return dspy.BaseLM  # type: ignore[attr-defined]

    for mod_name, attr in [
        ("dspy.clients", "BaseLM"),
        ("dspy.clients.base", "BaseLM"),
        ("dspy.clients.lm", "BaseLM"),
        ("dspy.models", "BaseLM"),
        ("dspy.lm", "BaseLM"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception:
            pass

    raise RuntimeError("Could not locate DSPy BaseLM")


_BaseLM = _resolve_dspy_base_lm()


class MinimaLlmDSPyLM(_BaseLM):  # type: ignore[misc]
    """
    DSPy BaseLM adapter that routes calls through OpenAIMinimaLlm.

    This adapter is intentionally minimal:
      - DSPy handles prompt construction and output parsing.
      - MinimaLlm handles HTTP transport, backpressure, retries, and pacing.
      - No LiteLLM dependency.
    """

    def __init__(self, minimallm: OpenAIMinimaLlm, **kwargs: Any):
        self._minimallm = minimallm
        model_value = minimallm.cfg.model

        # Initialize BaseLM in a version-tolerant way (DSPy 2.6.27 requires `model`).
        try:
            sig = inspect.signature(_BaseLM.__init__)  # type: ignore[arg-type]
            params = sig.parameters
            init_kwargs: Dict[str, Any] = {}

            if "model" in params:
                init_kwargs["model"] = model_value
            elif "model_name" in params:
                init_kwargs["model_name"] = model_value

            # Forward only kwargs that BaseLM actually accepts.
            for k, v in kwargs.items():
                if k in params:
                    init_kwargs[k] = v

            super().__init__(**init_kwargs)  # type: ignore[misc]
        except Exception:
            # Fallback chain
            try:
                super().__init__(model=model_value)  # type: ignore[misc]
            except Exception:
                try:
                    super().__init__(model_value)  # type: ignore[misc]
                except Exception:
                    super().__init__()  # type: ignore[misc]

        # Commonly expected attributes (harmless if unused)
        if not hasattr(self, "model"):
            self.model = model_value  # type: ignore[assignment]
        if not hasattr(self, "kwargs"):
            self.kwargs = {}  # type: ignore[assignment]
        if not hasattr(self, "history"):
            self.history = []  # type: ignore[assignment]

    async def acall(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        Async LM call used by DSPy.

        Parameters
        ----------
        prompt : str, optional
            Single prompt string (converted to user message)
        messages : list, optional
            OpenAI-format message list
        force_refresh : bool
            If True, bypass cache lookup and make a fresh LLM call.
            Useful for retrying when DSPy parsing fails on a cached response.

        Returns
        -------
        list[str]
            DSPy expects a list of completions. We return a singleton list.
        """
        if messages is None:
            if prompt is None:
                raise ValueError("DSPy LM requires either prompt or messages")
            messages = [{"role": "user", "content": prompt}]

        req = MinimaLlmRequest(
            request_id=str(kwargs.pop("request_id", "dspy")),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=kwargs.pop("temperature", None),
            max_tokens=kwargs.pop("max_tokens", None),
            extra=kwargs if kwargs else None,
        )

        resp = await self._minimallm.generate(req, force_refresh=force_refresh)
        if isinstance(resp, MinimaLlmFailure):
            raise RuntimeError(f"{resp.error_type}: {resp.message}")
        return [resp.text]

    # Some DSPy internals/adapters call forward/aforward.
    async def aforward(self, *args: Any, **kwargs: Any) -> List[str]:
        return await self.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> List[str]:
        return self.__call__(*args, **kwargs)

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Sync LM call fallback.

        If called inside a running event loop, raise a clear error rather than
        nesting event loops.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "MinimaLlmDSPyLM was called synchronously inside a running event loop. "
                "Use await pred.acall(...) or await lm.acall(...)."
            )

        return asyncio.run(self.acall(prompt=prompt, messages=messages, **kwargs))


# ----------------------------
# Batch execution helper
# ----------------------------

def _get_input_field_names(signature_class: Type[dspy.Signature]) -> List[str]:
    """
    Extract InputField names from a DSPy Signature class.

    Returns list of field names that are InputFields in the signature.
    """
    input_fields = []

    # Method 1: Check DSPy's signature-level field collections
    # DSPy stores fields in various places depending on version
    for attr_name in ['input_fields', '_input_fields', 'fields']:
        if hasattr(signature_class, attr_name):
            fields_obj = getattr(signature_class, attr_name)

            # Could be a dict
            if isinstance(fields_obj, dict):
                # Might be {name: field_obj} or nested structure
                for key, value in fields_obj.items():
                    if isinstance(key, str):
                        input_fields.append(key)
                if input_fields:
                    break
            # Could be a dict-like object
            elif hasattr(fields_obj, 'keys') and callable(fields_obj.keys):
                try:
                    input_fields = list(fields_obj.keys())
                    if input_fields:
                        break
                except Exception:
                    pass
            # Could be a list/sequence of field names
            elif hasattr(fields_obj, '__iter__'):
                try:
                    input_fields = [f for f in fields_obj if isinstance(f, str)]
                    if input_fields:
                        break
                except Exception:
                    pass

    if input_fields:
        return input_fields

    # Method 2: Check Pydantic v2 model_fields
    if hasattr(signature_class, 'model_fields'):
        fields_dict = signature_class.model_fields
        for name, field_info in fields_dict.items():
            # Check metadata list (Pydantic v2 standard)
            if hasattr(field_info, 'metadata') and field_info.metadata:
                for meta_item in field_info.metadata:
                    # DSPy might store InputField instance in metadata
                    meta_type = type(meta_item).__name__
                    meta_module = type(meta_item).__module__ if hasattr(type(meta_item), '__module__') else ''
                    if 'InputField' in meta_type or ('dspy' in meta_module and 'Input' in meta_type):
                        input_fields.append(name)
                        break

            # Check json_schema_extra for DSPy markers
            if name not in input_fields and hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)
                elif extra.get('prefix', '').lower().startswith('input'):
                    input_fields.append(name)

    if input_fields:
        return input_fields

    # Method 3: Check Pydantic v1 __fields__
    if hasattr(signature_class, '__fields__'):
        fields_dict = signature_class.__fields__
        for name, field_info in fields_dict.items():
            if hasattr(field_info, 'field_info'):
                field_info = field_info.field_info
            # Check extra for DSPy markers
            if hasattr(field_info, 'extra') and field_info.extra:
                extra = field_info.extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)
            # Check json_schema_extra
            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if extra.get('__dspy_field_type') == 'input':
                    input_fields.append(name)

    if input_fields:
        return input_fields

    # Method 4: Introspect class attributes for Field objects
    for name in signature_class.__annotations__:
        try:
            # Try class attribute
            field_obj = getattr(signature_class, name, None)
            if field_obj is None:
                # Try getting from __dict__
                field_obj = signature_class.__dict__.get(name, None)

            if field_obj is None:
                continue

            # Check type and class name
            field_type_str = str(type(field_obj))
            field_class_name = field_obj.__class__.__name__ if hasattr(field_obj, '__class__') else ''
            field_module = field_obj.__class__.__module__ if hasattr(field_obj, '__class__') else ''

            # Check multiple possible indicators of InputField
            is_input = any([
                'InputField' in field_class_name,
                'InputField' in field_type_str,
                'Input' in field_class_name and 'dspy' in field_module,
                hasattr(field_obj, 'json_schema_extra') and
                    isinstance(field_obj.json_schema_extra, dict) and
                    field_obj.json_schema_extra.get('__dspy_field_type') == 'input',
            ])

            if is_input:
                input_fields.append(name)

        except Exception:
            continue

    if input_fields:
        return input_fields

    # Method 5: Fallback heuristic - fields before first OutputField
    # DSPy signatures conventionally list inputs before outputs
    print(f"Warning: Could not detect InputFields via metadata, using annotation order heuristic")

    output_field_names = []

    # Try to find output fields
    for name in signature_class.__annotations__:
        try:
            field_obj = getattr(signature_class, name, None)
            if field_obj is None:
                field_obj = signature_class.__dict__.get(name, None)

            if field_obj is not None:
                field_type_str = str(type(field_obj))
                field_class_name = field_obj.__class__.__name__ if hasattr(field_obj, '__class__') else ''

                if 'OutputField' in field_class_name or 'OutputField' in field_type_str:
                    output_field_names.append(name)
        except Exception:
            continue

    # If we found output fields, everything before the first output is input
    if output_field_names:
        first_output = output_field_names[0]
        for name in signature_class.__annotations__:
            if name == first_output:
                break
            input_fields.append(name)
        return input_fields

    # Last resort: check if signature has exactly the expected fields from Umbrela
    # Return first 4 annotations as inputs (Umbrela has 4 inputs, 5 outputs)
    annotations = list(signature_class.__annotations__.keys())
    if len(annotations) >= 4:
        # Assume first 4 are inputs
        print(f"Warning: Using first 4 annotations as inputs: {annotations[:4]}")
        return annotations[:4]

    # Give up - return empty to trigger clear error
    print(f"Error: Could not detect any InputFields in signature {signature_class.__name__}")
    return []


async def run_dspy_batch(
    signature_class: Type[dspy.Signature],
    annotation_objs: List[BaseModel],
    output_converter: Callable[[Any, BaseModel], None],
    predictor_class: Type = dspy.ChainOfThought,
    backend: Optional[OpenAIMinimaLlm] = None
) -> List[BaseModel]:
    """
    Execute a DSPy batch with MinimaLLM backend.

    This helper automatically extracts input fields from annotation objects based on
    the DSPy signature's InputFields, executes predictions in parallel using batching,
    and updates annotation objects with results.

    Parameters
    ----------
    signature_class : Type[dspy.Signature]
        DSPy Signature class (e.g., Umbrela)
    annotation_objs : List[BaseModel]
        List of Pydantic models with fields matching signature InputFields
    output_converter : Callable[[Any, BaseModel], None]
        Function that updates annotation object with DSPy prediction result.
        Signature: (prediction: dspy.Prediction, obj: BaseModel) -> None
    predictor_class : Type
        DSPy predictor class (default: dspy.ChainOfThought)
    backend : Optional[OpenAIMinimaLlm]
        Pre-configured backend. If None, creates from environment.

    Returns
    -------
    List[BaseModel]
        Processed annotation objects with outputs filled in

    Example
    -------
    >>> class MyAnnotation(BaseModel):
    ...     title: str
    ...     text: str
    ...     score: Optional[float] = None
    >>>
    >>> class MySignature(dspy.Signature):
    ...     title: str = dspy.InputField()
    ...     text: str = dspy.InputField()
    ...     score: float = dspy.OutputField()
    ...
    ...     @classmethod
    ...     def convert_output(cls, pred, obj):
    ...         obj.score = float(pred.score)
    >>>
    >>> annotations = [MyAnnotation(title="...", text="..."), ...]
    >>> results = await run_dspy_batch(MySignature, annotations, MySignature.convert_output)
    """
    # Setup backend
    owns_backend = backend is None
    if backend is None:
        backend = OpenAIMinimaLlm.from_env()

    lm = MinimaLlmDSPyLM(backend)
    dspy.configure(lm=lm)

    predictor = predictor_class(signature_class)

    # Get input field names from signature
    input_fields = _get_input_field_names(signature_class)

    # Process each annotation
    async def process_one(obj: BaseModel) -> BaseModel:
        # Extract input kwargs from annotation object
        kwargs = obj.model_dump(include=set(input_fields))

        # Run prediction
        result = await predictor.acall(**kwargs)

        # Update annotation with results
        output_converter(result, obj)

        return obj

    # Execute batch - returns List[Union[BaseModel, MinimaLlmFailure]]
    results = await backend.run_batched_callable(annotation_objs, process_one)

    # Check for failures - fail fast, don't silently drop data
    failures = [r for r in results if isinstance(r, MinimaLlmFailure)]
    if failures:
        # Cleanup before raising
        if owns_backend:
            await backend.aclose()
        # Report failures
        msgs = [f"{f.request_id}: {f.error_type}: {f.message}" for f in failures[:5]]
        raise RuntimeError(
            f"{len(failures)} DSPy predictions failed:\n  " + "\n  ".join(msgs)
        )

    # Cleanup
    if owns_backend:
        await backend.aclose()

    # All results are BaseModel (failures already raised)
    return cast(List[BaseModel], results)
