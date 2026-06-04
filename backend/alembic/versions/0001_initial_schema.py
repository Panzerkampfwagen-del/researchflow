"""Initial ResearchFlow schema with pgvector support.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the pgvector extension, all tables and supporting indexes."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.execute(
        """
        CREATE TABLE research_sessions (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            query        TEXT NOT NULL,
            status       VARCHAR(20) NOT NULL DEFAULT 'pending',
            plan         JSONB,
            created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE
        );
        """
    )

    op.execute(
        """
        CREATE TABLE papers (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            arxiv_id            VARCHAR(50),
            semantic_scholar_id VARCHAR(100),
            title               TEXT NOT NULL,
            authors             JSONB NOT NULL DEFAULT '[]',
            abstract            TEXT,
            year                INTEGER,
            venue               VARCHAR(255),
            citation_count      INTEGER DEFAULT 0,
            url                 TEXT,
            embedding           vector(384),
            created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            CONSTRAINT uq_papers_arxiv_id UNIQUE (arxiv_id),
            CONSTRAINT uq_papers_semantic_scholar_id UNIQUE (semantic_scholar_id)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE session_papers (
            session_id      UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
            paper_id        UUID REFERENCES papers(id) ON DELETE CASCADE,
            relevance_score FLOAT,
            rank            INTEGER,
            PRIMARY KEY (session_id, paper_id)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE paper_analyses (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id  UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
            paper_id    UUID REFERENCES papers(id) ON DELETE CASCADE,
            problem     TEXT,
            methodology TEXT,
            datasets    JSONB DEFAULT '[]',
            metrics     JSONB DEFAULT '[]',
            key_results TEXT,
            limitations TEXT,
            confidence  FLOAT DEFAULT 0.0,
            created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            CONSTRAINT uq_paper_analyses_session_paper UNIQUE (session_id, paper_id)
        );
        """
    )

    op.execute(
        """
        CREATE TABLE reports (
            id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id             UUID REFERENCES research_sessions(id) ON DELETE CASCADE UNIQUE,
            executive_summary      TEXT,
            methodology_comparison JSONB DEFAULT '[]',
            research_gaps          JSONB DEFAULT '[]',
            trends                 JSONB DEFAULT '[]',
            future_directions      JSONB DEFAULT '[]',
            citations              JSONB DEFAULT '[]',
            markdown_content       TEXT,
            created_at             TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
    )

    op.execute(
        """
        CREATE TABLE agent_runs (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id    UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
            agent_name    VARCHAR(50) NOT NULL,
            status        VARCHAR(20) NOT NULL DEFAULT 'running',
            tokens_used   INTEGER DEFAULT 0,
            latency_ms    INTEGER,
            cost_usd      FLOAT DEFAULT 0.0,
            error_message TEXT,
            started_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at  TIMESTAMP WITH TIME ZONE
        );
        """
    )

    op.execute(
        "CREATE INDEX ix_papers_embedding ON papers "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
    )
    op.execute("CREATE INDEX ix_session_papers_session ON session_papers (session_id);")
    op.execute("CREATE INDEX ix_paper_analyses_session ON paper_analyses (session_id);")
    op.execute("CREATE INDEX ix_agent_runs_session ON agent_runs (session_id);")


def downgrade() -> None:
    """Drop every ResearchFlow table in reverse dependency order."""
    op.execute("DROP TABLE IF EXISTS agent_runs CASCADE;")
    op.execute("DROP TABLE IF EXISTS reports CASCADE;")
    op.execute("DROP TABLE IF EXISTS paper_analyses CASCADE;")
    op.execute("DROP TABLE IF EXISTS session_papers CASCADE;")
    op.execute("DROP TABLE IF EXISTS papers CASCADE;")
    op.execute("DROP TABLE IF EXISTS research_sessions CASCADE;")
