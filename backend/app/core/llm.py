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
from typing import Any, TypeVar

import anthropic
import httpx
import json_repair
import numpy as np
import structlog
from groq import AsyncGroq
from groq import RateLimitError as GroqRateLimitError
from pydantic import BaseModel, ValidationError

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Jina embeddings tuning. Batch so a large candidate set never exceeds the API's
# per-request token/item limit; retry transient failures (the local backends
# never hit the network, so this failure class is new to the hosted path).
_JINA_BATCH_SIZE = 64
_JINA_MAX_RETRIES = 4
_JINA_RETRY_STATUS = {429, 500, 502, 503, 504}
# Task labels for the asymmetric jina-embeddings-v3 model.
EMBED_TASK_PASSAGE = "retrieval.passage"
EMBED_TASK_QUERY = "retrieval.query"

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_INPUT_COST_PER_1K = 0.00025
ANTHROPIC_OUTPUT_COST_PER_1K = 0.00125
# Few, short Groq retries: a free-tier daily-quota 429 won't clear mid-request,
# so a long backoff only stalls the run. Fail fast and let the caller fall back.
_GROQ_MAX_ATTEMPTS = 2
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
        """Call the Groq chat completions API, retrying on 429 with backoff.

        Retries are deliberately few and short: a 429 from the free tier's *daily*
        quota will not clear within a request, so a long backoff just stalls the
        run (and, across many papers, can outlast the worker). Fail fast so the
        caller can fall back to another model.
        """
        delay = 3.0
        for attempt in range(_GROQ_MAX_ATTEMPTS):
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
                if attempt == _GROQ_MAX_ATTEMPTS - 1:
                    raise
                logger.warning("groq_rate_limited", model=model, attempt=attempt, retry_in=delay)
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
        self._http: httpx.AsyncClient | None = None

    @property
    def backend(self) -> str:
        """Name of the loaded backend (jina/sentence-transformers/fastembed), or ""."""
        return self._backend

    @property
    def ready(self) -> bool:
        """True once a usable embedding backend has been resolved."""
        return bool(self._backend)

    def load(self) -> None:
        """Resolve the embedding backend once. Safe to call repeatedly.

        When ``JINA_API_KEY`` is set, embeddings are served by the hosted Jina
        API and no local model is loaded — this keeps the container within the
        512 MB Render free tier. Otherwise a local model is used. Raises
        ``RuntimeError`` if no backend can be configured, so the failure is loud
        instead of surfacing later as an opaque error mid-request.
        """
        if self._backend:
            return
        if settings.JINA_API_KEY:
            logger.info("loading_embedding_model", model=settings.JINA_MODEL, backend="jina")
            self._backend = "jina"
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
            return
        except ImportError:
            pass
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "No embedding backend available: set JINA_API_KEY, or install "
                "sentence-transformers or fastembed."
            ) from exc
        cache_dir = settings.FASTEMBED_CACHE_DIR or None
        logger.info(
            "loading_embedding_model", model=settings.EMBEDDING_MODEL, backend="fastembed"
        )
        # threads=1 keeps onnxruntime to a single memory arena — critical on the
        # 512 MB free tier where the default (one arena per core) blows the limit.
        self._model = TextEmbedding(
            model_name=settings.EMBEDDING_MODEL, cache_dir=cache_dir, threads=1
        )
        self._backend = "fastembed"

    async def aclose(self) -> None:
        """Close the shared HTTP client (called on app shutdown)."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _jina_payload(self, texts: list[str], task: str) -> dict:
        """Build the Jina embeddings request body (Matryoshka-truncated to dim)."""
        return {
            "model": settings.JINA_MODEL,
            "task": task,
            "dimensions": settings.EMBEDDING_DIM,
            # Guard against the occasional oversized input (Jina caps per-item tokens).
            "input": [t[:8000] for t in texts],
        }

    @staticmethod
    def _jina_vectors(payload: dict) -> np.ndarray:
        """Extract an index-ordered ``(n, dim)`` array from a Jina API response."""
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Jina response shape: {str(payload)[:200]}")
        rows = sorted(data, key=lambda d: d["index"])
        return np.array([row["embedding"] for row in rows], dtype="float32")

    async def _jina_request(self, texts: list[str], task: str) -> np.ndarray:
        """POST one batch to Jina with retries on transient failures."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=60.0, headers={"Authorization": f"Bearer {settings.JINA_API_KEY}"}
            )
        delay = 1.0
        for attempt in range(_JINA_MAX_RETRIES):
            try:
                resp = await self._http.post(
                    settings.JINA_API_URL, json=self._jina_payload(texts, task)
                )
                resp.raise_for_status()
                return self._jina_vectors(resp.json())
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _JINA_RETRY_STATUS or attempt == _JINA_MAX_RETRIES - 1:
                    raise
                logger.warning("jina_retrying", status=status, attempt=attempt, retry_in=delay)
            except httpx.TransportError as exc:
                if attempt == _JINA_MAX_RETRIES - 1:
                    raise
                logger.warning("jina_retrying", error=str(exc), attempt=attempt, retry_in=delay)
            await asyncio.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _aembed_jina(self, texts: list[str], task: str) -> np.ndarray:
        """Embed via Jina, chunked into batches to bound per-request size."""
        chunks = [
            texts[i : i + _JINA_BATCH_SIZE] for i in range(0, len(texts), _JINA_BATCH_SIZE)
        ]
        parts = [await self._jina_request(chunk, task) for chunk in chunks]
        return np.vstack(parts) if parts else np.empty((0, settings.EMBEDDING_DIM), dtype="float32")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts with a local CPU-bound backend into a ``(n, dim)`` array.

        Jina is served exclusively through :meth:`aembed`; this path covers the
        sentence-transformers and fastembed backends and is run in a worker
        thread by ``aembed`` so the event loop is not blocked.
        """
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

    async def aembed(self, texts: list[str], task: str = EMBED_TASK_PASSAGE) -> np.ndarray:
        """Embed texts asynchronously into a ``(n, dim)`` array.

        ``task`` selects the jina-embeddings-v3 adapter (passage vs query); it is
        ignored by the local backends, which are symmetric. Use
        ``EMBED_TASK_QUERY`` when embedding a search query.
        """
        self.load()
        if self._backend == "jina":
            return await self._aembed_jina(texts, task)
        return await asyncio.to_thread(self.embed, texts)


llm_client = LLMClient()
embedding_client = EmbeddingClient()
