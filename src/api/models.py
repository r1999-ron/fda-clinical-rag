from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    session_id: str = Field(
        min_length=1
    )

    question: str = Field(
        min_length=1
    )

    filters: Optional[
        Dict[str, Any]
    ] = None


class CitationResponse(BaseModel):
    source: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None


class TraceResponse(BaseModel):
    request_id: str

    session_id: str | None = None

    rewrite_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    input_tokens: int
    output_tokens: int
    total_tokens: int

    citation_count: int
    citation_valid: bool

    abstained: bool

    error: str | None = None


class AskResponse(BaseModel):
    session_id: str

    original_question: str

    retrieval_query: str

    query_rewritten: bool

    answer: str

    citations: Dict[
        int,
        CitationResponse
    ]

    citation_validation: Dict[
        str,
        Any
    ]
    trace: Dict[
        str,
        Any,
    ]


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    indexed_at: str | None = None


class HealthResponse(BaseModel):
    status: str