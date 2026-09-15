from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunkTrace:
    source: str
    page: Optional[int]
    chunk_id: Optional[str]
    score: float


@dataclass
class RAGTrace:
    request_id: str
    session_id: Optional[str] = None

    original_question: Optional[str] = None
    retrieval_query: Optional[str] = None
    query_rewritten: bool = False

    rewrite_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    setup_latency_ms: float = 0.0
    context_build_latency_ms: float = 0.0
    overhead_latency_ms: float = 0.0

    retrieved_chunks: List[
        RetrievedChunkTrace
    ] = field(
        default_factory=list
    )

    citation_count: int = 0
    citation_valid: bool = True

    abstained: bool = False

    error: Optional[str] = None

    extra: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    retrieval_strategy: str | None = None


    def to_dict(self):
        return asdict(self)