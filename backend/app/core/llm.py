"""Unified async LLM and embedding clients.

``LLMClient`` wraps Groq (primary) with an Anthropic fallback and exposes a
``structured_complete`` helper that coerces model output into a Pydantic model.
``EmbeddingClient`` wraps a local ``sentence-transformers`` model.

Module-level singletons ``llm_client`` and ``embedding_client`` are the public
entry points used by the agents.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import anthropic
import json_repair
import structlog
from groq import AsyncGroq
from groq import RateLimitError as GroqRateLimitError
from pydantic import BaseModel, ValidationError

from app.core.config import settings

if TYPE_CHECKING:
    import numpy as np

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_INPUT_COST_PER_1K = 0.00025
ANTHROPIC_OUTPUT_COST_PER_1K = 0.00125
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass
class LLMResponse:
    """Container for a single completion plus its usage accounting."""

    content: str
    tokens: int
    cost_usd: float
    model: str


class LLMClient:
    """Async LLM client: Groq primary, Anthropic fallback, with usage tracking."""

    def __init__(self) -> None:
        self._groq = AsyncGroq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        self._anthropic = (
            anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            if settings.ANTHROPIC_API_KEY
            else None
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        use_reasoning: bool = True,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Return a completion, trying Groq first and falling back to Anthropic.

        ``use_reasoning`` selects the larger reasoning model over the fast
        extraction model. Any Groq-side failure triggers the Anthropic fallback;
        if neither provider is configured a ``RuntimeError`` is raised.
        """
        groq_model = (
            settings.GROQ_REASONING_MODEL if use_reasoning else settings.GROQ_EXTRACTION_MODEL
        )
        if self._groq is not None:
            try:
                return await self._complete_groq(messages, groq_model, temperature)
            except Exception as exc:  # noqa: BLE001 - intentional provider fallback
                logger.warning("groq_failed_falling_back", model=groq_model, error=str(exc))

        if self._anthropic is not None:
            return await self._complete_anthropic(messages, temperature)

        raise RuntimeError("No LLM provider configured: set GROQ_API_KEY or ANTHROPIC_API_KEY")

    async def _complete_groq(
        self, messages: list[dict[str, str]], model: str, temperature: float
    ) -> LLMResponse:
        """Call the Groq chat completions API, retrying on 429 with backoff."""
        delay = 5.0
        for attempt in range(5):
            try:
                resp = await self._groq.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content or ""
                tokens = resp.usage.total_tokens if resp.usage else 0
                return LLMResponse(content=content, tokens=tokens, cost_usd=0.0, model=model)
            except GroqRateLimitError:
                if attempt == 4:
                    raise
                logger.warning("groq_rate_limited", attempt=attempt, retry_in=delay)
                await asyncio.sleep(delay)
                delay *= 2

    async def _complete_anthropic(
        self, messages: list[dict[str, str]], temperature: float
    ) -> LLMResponse:
        """Call the Anthropic Messages API, splitting out system prompts."""
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        resp = await self._anthropic.messages.create(
            model=settings.ANTHROPIC_FALLBACK_MODEL,
            max_tokens=4096,
            temperature=temperature,
            system="\n\n".join(system_parts) if system_parts else anthropic.NOT_GIVEN,
            messages=chat,
        )
        content = "".join(block.text for block in resp.content if block.type == "text")
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens
        cost = (
            input_tokens / 1000 * ANTHROPIC_INPUT_COST_PER_1K
            + output_tokens / 1000 * ANTHROPIC_OUTPUT_COST_PER_1K
        )
        return LLMResponse(
            content=content,
            tokens=input_tokens + output_tokens,
            cost_usd=round(cost, 6),
            model=settings.ANTHROPIC_FALLBACK_MODEL,
        )

    async def structured_complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        use_reasoning: bool = True,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> tuple[T, LLMResponse]:
        """Complete and parse the output into ``response_model``.

        Prepends a system instruction describing the JSON schema, strips any
        markdown code fences, and retries up to ``max_retries`` times with an
        error-correction message when parsing fails. Returns the parsed model
        together with the final ``LLMResponse`` so callers can account usage.
        """
        schema = json.dumps(response_model.model_json_schema())
        convo: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You output only valid JSON that conforms to the following JSON Schema. "
                    "Do not include any prose, explanation, or markdown fences.\n"
                    f"JSON Schema:\n{schema}"
                ),
            },
            *messages,
        ]

        total_tokens = 0
        total_cost = 0.0
        last_model = ""
        last_error = ""

        for attempt in range(max_retries + 1):
            response = await self.complete(
                convo, use_reasoning=use_reasoning, temperature=temperature
            )
            total_tokens += response.tokens
            total_cost += response.cost_usd
            last_model = response.model
            cleaned = self._strip_fences(response.content)
            try:
                parsed = self._parse_model(response_model, cleaned)
                aggregate = LLMResponse(
                    content=response.content,
                    tokens=total_tokens,
                    cost_usd=round(total_cost, 6),
                    model=last_model,
                )
                return parsed, aggregate
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                logger.warning(
                    "structured_parse_failed",
                    attempt=attempt,
                    model=last_model,
                    error=last_error,
                )
                convo.append({"role": "assistant", "content": response.content})
                convo.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed JSON validation with this error:\n"
                            f"{last_error}\n"
                            "Return corrected JSON only, matching the schema exactly."
                        ),
                    }
                )

        raise ValueError(
            f"Failed to parse {response_model.__name__} after "
            f"{max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def _parse_model(response_model: type[T], cleaned: str) -> T:
        """Validate JSON into the model, repairing malformed JSON if needed.

        Tries a strict parse first; on failure it runs ``json_repair`` (fixing
        trailing commas, unquoted keys, single quotes, etc.) and re-validates,
        which often salvages a Llama response without spending a retry round.
        """
        try:
            return response_model.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError, ValueError):
            repaired = json_repair.repair_json(cleaned)
            return response_model.model_validate_json(repaired)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove surrounding markdown code fences and isolate the JSON body."""
        stripped = text.strip()
        stripped = _FENCE_RE.sub("", stripped).strip()
        start = stripped.find("{")
        start_arr = stripped.find("[")
        if start_arr != -1 and (start == -1 or start_arr < start):
            start = start_arr
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if start != -1 and end != -1 and end >= start:
            return stripped[start : end + 1]
        return stripped


class EmbeddingClient:
    """Lazy embedding wrapper.

    Prefers sentence-transformers (full stack, local/Docker). Falls back to
    fastembed (ONNX Runtime, no PyTorch) when sentence-transformers is not
    installed — used on the Render free tier to stay within 512 MB RAM.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._backend: str = ""

    @property
    def backend(self) -> str:
        """Name of the loaded backend ("fastembed"/"sentence-transformers"), or ""."""
        return self._backend

    def load(self) -> None:
        """Load the embedding model once. Safe to call repeatedly."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "loading_embedding_model",
                model=settings.EMBEDDING_MODEL,
                backend="sentence-transformers",
            )
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._backend = "sentence-transformers"
        except ImportError:
            from fastembed import TextEmbedding

            cache_dir = settings.FASTEMBED_CACHE_DIR or None
            logger.info(
                "loading_embedding_model", model=settings.EMBEDDING_MODEL, backend="fastembed"
            )
            # threads=1 keeps onnxruntime to a single memory arena — critical on
            # the 512 MB Render free tier where the default (one arena per core)
            # blows the memory limit.
            self._model = TextEmbedding(
                model_name=settings.EMBEDDING_MODEL, cache_dir=cache_dir, threads=1
            )
            self._backend = "fastembed"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a ``(len(texts), dim)`` float array.

        Synchronous and CPU-bound; call via ``aembed`` from async code so the
        event loop is not blocked.
        """
        import numpy as np

        self.load()
        if self._backend == "fastembed":
            # parallel=1 forces a single process — the default forks one worker
            # per core for large batches, each copying the ~180 MB ONNX model and
            # instantly OOM-ing the 512 MB tier. Small batch_size bounds the peak
            # ONNX activation working set.
            return np.array(list(self._model.embed(texts, batch_size=16, parallel=1)))
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

    async def aembed(self, texts: list[str]) -> np.ndarray:
        """Async wrapper running :meth:`embed` in a worker thread."""
        import asyncio

        return await asyncio.to_thread(self.embed, texts)


llm_client = LLMClient()
embedding_client = EmbeddingClient()
