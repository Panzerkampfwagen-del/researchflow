"""Application configuration loaded from the environment.

Exposes a single ``settings`` singleton. Every other module reads configuration
through this object rather than touching ``os.environ`` directly.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    DATABASE_URL: str = (
        "postgresql+asyncpg://researchflow:password@localhost:5432/researchflow"
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _fix_asyncpg_scheme(cls, v: str) -> str:
        # Render supplies postgresql://, asyncpg requires postgresql+asyncpg://
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    GROQ_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FASTEMBED_CACHE_DIR: str = ""  # set to pre-baked path when using fastembed backend
    GROQ_REASONING_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_EXTRACTION_MODEL: str = "llama-3.1-8b-instant"
    ANTHROPIC_FALLBACK_MODEL: str = "claude-3-5-haiku-20241022"

    ARXIV_RATE_LIMIT_SECONDS: float = 3.0
    SEMANTIC_SCHOLAR_RATE_LIMIT_SECONDS: float = 1.0
    MAX_PAPERS_PER_QUERY: int = 30

    EMBEDDING_DIM: int = 384
    LOAD_EMBEDDINGS_ON_STARTUP: bool = True
    DB_USE_NULLPOOL: bool = False

    SEMANTIC_SCHOLAR_API_KEY: str = ""

    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 20

    # Hybrid fusion: "rrf" (reciprocal rank fusion, scale-free) or "weighted"
    # (the 0.5/0.3/0.2 linear blend). RRF is the default; it needs no score
    # calibration across the dense / lexical / citation signals.
    FUSION_METHOD: str = "rrf"
    RRF_K: int = 60

    # Dense ANN backend: "hnsw" (the from-scratch index in app/retrieval/hnsw.py)
    # or "bruteforce" (exact cosine). HNSW only takes over once the candidate
    # pool is large enough to matter; smaller pools stay exact.
    ANN_BACKEND: str = "hnsw"
    HNSW_MIN_CANDIDATES: int = 256
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 64

    # NLI-based grounding: entail each cited claim from its source abstract.
    # RoBERTa NLI uses BPE (no extra sentencepiece dep); deberta-v3 variants are
    # stronger but require `sentencepiece`. Both expose labels in the order
    # [contradiction, entailment, neutral], which app/agents/grounding.py assumes.
    GROUNDING_ENABLED: bool = True
    NLI_MODEL: str = "cross-encoder/nli-roberta-base"
    GROUNDING_ENTAILMENT_THRESHOLD: float = 0.5


settings = Settings()
