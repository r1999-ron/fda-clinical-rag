from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.observability.trace import (
    RAGTrace,
    RetrievedChunkTrace,
)
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import (
    get_routed_retriever,
)


# ============================================================
# CONTEXT FORMATTING
# ============================================================

def format_context(
    results: List[Tuple[Document, float]]
) -> str:
    """
    Convert reranked chunks into numbered evidence blocks.

    We intentionally do NOT expose dense/reranker scores to the LLM.
    Scores are useful for debugging, not for generation.
    """

    context_blocks = []

    for index, (document, _) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        source = Path(
            metadata.get("source", "unknown")
        ).name

        page = metadata.get(
            "page_number",
            "unknown",
        )

        section = metadata.get("section")

        metadata_lines = [
            f"[SOURCE {index}]",
            f"Document: {source}",
            f"Page: {page}",
        ]

        if section:
            metadata_lines.append(
                f"Section: {section}"
            )

        metadata_lines.append(
            f"\nContent:\n{document.page_content}"
        )

        context_blocks.append(
            "\n".join(metadata_lines)
        )

    return "\n\n".join(context_blocks)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: str,
) -> str:
    return f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:

1. Do not use outside knowledge.

2. If the evidence is insufficient, say exactly:
   "I could not find sufficient evidence in the provided documents."

3. Every factual sentence must end with at least one citation
   such as [1] or [2].

4. Do NOT place citations only at the end of a multi-sentence
   paragraph. Cite each factual sentence individually.

5. Only cite source numbers that appear in the provided evidence.

6. Do not invent citations.

7. Do not invent document names, page numbers, or section names.

8. Answer only what is necessary to directly answer the question.

9. Do not include related information unless it is necessary
   for a complete answer.

10. Prefer concise answers.

11. Prefer the smallest sufficient set of supporting sources.

12. If one source fully supports a claim, do not cite additional
    sources merely because they discuss related information.

13. If multiple sources are genuinely required to support a claim,
    cite all relevant sources.

14. If two sources conflict, explicitly mention the conflict rather
    than deciding which one is correct.
    
15. When evidence contains numbered footnotes or reference markers,
    you may connect a statement with its corresponding numbered
    footnote when the relationship is clearly shown in the provided
    evidence.

16. When a factual claim depends on information from multiple evidence
    blocks, cite all evidence blocks required to establish the claim.
    For example, if one source contains the substantive statement and
    another source maps its footnote to a regulation, cite both.

Example of correct citation style:

Participation is voluntary [1].
Participants may withdraw without penalty [2].

Incorrect citation style:

Participation is voluntary.
Participants may withdraw without penalty [1][2].

QUESTION:

{question}

EVIDENCE:

{context}

ANSWER:
""".strip()


# ============================================================
# CITATION MAP
# ============================================================

def build_citation_map(
    results: List[Tuple[Document, float]]
) -> dict:
    """
    Map source numbers shown to the LLM back to trusted metadata.

    The LLM only generates [1], [2], etc.
    The backend owns source/page/chunk provenance.
    """

    citations = {}

    for index, (document, _) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        source = Path(
            metadata.get("source", "unknown")
        ).name

        citations[index] = {
            "source": source,
            "page": metadata.get("page_number"),
            "section": metadata.get("section"),
            "chunk_id": metadata.get("chunk_id"),
            "dense_score": metadata.get("dense_score"),
            "reranker_score": metadata.get("reranker_score"),
        }

    return citations


# ============================================================
# CITATION EXTRACTION
# ============================================================

def extract_citation_ids(
    answer: str,
) -> List[int]:
    """
    Example:

    "Participants may withdraw at any time. [1]"

    returns:

    [1]
    """

    citation_ids = {
        int(match)
        for match in re.findall(
            r"\[(\d+)\]",
            answer,
        )
    }

    return sorted(citation_ids)


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(
    answer: str,
    citation_map: dict,
) -> dict:
    """
    Verify that every citation produced by the model
    corresponds to a real retrieved source.

    Example:

    Available:
        [1], [2], [3]

    Generated:
        [1], [7]

    Then [7] is invalid.
    """

    cited_ids = set(
        extract_citation_ids(answer)
    )

    valid_ids = set(
        citation_map.keys()
    )

    invalid_ids = (
        cited_ids - valid_ids
    )

    return {
        "valid": len(invalid_ids) == 0,
        "cited_ids": sorted(cited_ids),
        "invalid_ids": sorted(invalid_ids),
    }


# ============================================================
# KEEP ONLY USED CITATIONS
# ============================================================

def get_used_citations(
    answer: str,
    citation_map: dict,
) -> dict:
    """
    Return only sources actually cited in the final answer.
    """

    cited_ids = extract_citation_ids(
        answer
    )

    used_citations = {}

    for citation_id in cited_ids:

        if citation_id in citation_map:

            used_citations[citation_id] = (
                citation_map[citation_id]
            )

    return used_citations


# ============================================================
# RAG PIPELINE
# ============================================================

def insufficient_evidence_response():
    return {
        "answer": (
            "I could not find sufficient "
            "evidence in the provided documents."
        ),
        "citations": {},
        "citation_map": {},
        "citation_validation": {
            "valid": True,
            "cited_ids": [],
            "invalid_ids": [],
        },
        "retrieved_context": "",
        "retrieved_chunks": [],
    }

def get_retrieval_strategy(
    self,
    query: str,
) -> str:

    if self.has_regulatory_identifier(
        query
    ):
        return "bm25"

    return "dense"

def generate_answer(
    question: str,
    retrieval_k: int = 10,
    rerank_k: int = 3,
    use_reranker: bool = False,
    filters: dict | None = None,
    trace: RAGTrace | None = None,
) -> dict:

    # ====================================================
    # 0. RETRIEVER SETUP
    # ====================================================

    setup_start = (
        time.perf_counter()
    )

    retriever = (
        get_routed_retriever()
    )

    setup_end = (
        time.perf_counter()
    )

    if trace:

        trace.setup_latency_ms = (
            (
                setup_end
                - setup_start
            )
            * 1000
        )

    # ====================================================
    # 1. ROUTED RETRIEVAL
    # ====================================================

    retrieval_strategy = (
        retriever.get_retrieval_strategy(
            question
        )
    )

    if trace:
        trace.retrieval_strategy = (
            retrieval_strategy
        )

    retrieval_start = (
        time.perf_counter()
    )

    candidate_results = (
        retriever.routed_search(
            query=question,
            filters=filters,
        )
    )

    candidate_results = (
        candidate_results[
        :retrieval_k
        ]
    )

    retrieval_end = (
        time.perf_counter()
    )

    if trace:
        trace.retrieval_latency_ms = (
                (
                        retrieval_end
                        - retrieval_start
                )
                * 1000
        )

    if not candidate_results:

        if trace:
            trace.abstained = True

        return (
            insufficient_evidence_response()
        )

    # ====================================================
    # 2. OPTIONAL RERANKING
    # ====================================================

    if use_reranker:

        reranker = (
            CrossEncoderReranker()
        )

        results = (
            reranker.rerank(
                query=question,
                results=candidate_results,
                top_n=rerank_k,
            )
        )

    else:

        results = (
            candidate_results[
                :rerank_k
            ]
        )

    # Everything after this stays the same.

    if not results:

        if trace:
            trace.abstained = True

        return (
            insufficient_evidence_response()
        )

    # ================================================
    # TRACE RETRIEVED CHUNKS
    # ================================================

    if trace:
        trace.retrieved_chunks = [
            RetrievedChunkTrace(
                source=Path(
                    document.metadata.get(
                        "source",
                        "unknown",
                    )
                ).name,

                page=document.metadata.get(
                    "page_number"
                ),

                chunk_id=document.metadata.get(
                    "chunk_id"
                ),

                score=float(
                    score
                ),
            )
            for document, score
            in results
        ]

    # ================================================
    # CONTEXT + PROMPT
    # ================================================

    context_start = (
        time.perf_counter()
    )

    context = format_context(
        results
    )

    prompt = build_prompt(
        question=question,
        context=context,
    )

    context_end = (
        time.perf_counter()
    )

    if trace:
        trace.context_build_latency_ms = (
                (
                        context_end
                        - context_start
                )
                * 1000
        )

    # ================================================
    # GENERATION
    # ================================================

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    generation_start = (
        time.perf_counter()
    )

    response = llm.invoke(
        prompt
    )

    generation_end = (
        time.perf_counter()
    )

    if trace:
        trace.generation_latency_ms = (
                (
                        generation_end
                        - generation_start
                )
                * 1000
        )

    answer = response.content

    # ================================================
    # TOKEN USAGE
    # ================================================

    if trace:

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage:
            trace.input_tokens = int(
                usage.get(
                    "input_tokens",
                    0,
                )
            )

            trace.output_tokens = int(
                usage.get(
                    "output_tokens",
                    0,
                )
            )

            trace.total_tokens = int(
                usage.get(
                    "total_tokens",
                    (
                            trace.input_tokens
                            + trace.output_tokens
                    ),
                )
            )

    # ================================================
    # CITATIONS
    # ================================================

    citation_map = (
        build_citation_map(
            results
        )
    )

    citation_validation = (
        validate_citations(
            answer=answer,
            citation_map=citation_map,
        )
    )

    used_citations = (
        get_used_citations(
            answer=answer,
            citation_map=citation_map,
        )
    )

    if trace:
        trace.citation_count = len(
            used_citations
        )

        trace.citation_valid = (
            citation_validation.get(
                "valid",
                False,
            )
        )

        trace.abstained = (
                "I could not find sufficient evidence "
                "in the provided documents."
                in answer
        )

    # ================================================
    # RESPONSE
    # ================================================

    return {
        "answer": answer,

        "citations": (
            used_citations
        ),

        "citation_map": (
            citation_map
        ),

        "citation_validation": (
            citation_validation
        ),

        "retrieved_context": (
            context
        ),

        "retrieved_chunks": [
            {
                "source_id": index,

                "source": Path(
                    document.metadata.get(
                        "source",
                        "unknown",
                    )
                ).name,

                "page": (
                    document.metadata.get(
                        "page_number"
                    )
                ),

                "chunk_id": (
                    document.metadata.get(
                        "chunk_id"
                    )
                ),

                "section": (
                    document.metadata.get(
                        "section"
                    )
                ),

                "text": (
                    document.page_content
                ),

                "score": float(
                    score
                ),
            }
            for index, (
                document,
                score,
            )
            in enumerate(
                results,
                start=1,
            )
        ],
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What risks must be explained "
        "during informed consent?"
    )

    result = generate_answer(
        question=question,
        retrieval_k=5,
        rerank_k=5,
        use_reranker=False,
        filters={
            "document_id":
                "05_informed_consent_guidance"
        },
    )

    print("\nQUESTION")
    print("=" * 100)
    print(question)

    print("\nANSWER")
    print("=" * 100)
    print(result["answer"])

    print("\nRETRIEVED CHUNKS")
    print("=" * 100)

    for chunk in result[
        "retrieved_chunks"
    ]:
        print(
            f"{chunk['source']} | "
            f"Page {chunk['page']} | "
            f"{chunk['chunk_id']}"
        )

    print("\nCITATIONS")
    print("=" * 100)

    for (
        citation_id,
        citation
    ) in result["citations"].items():

        print(
            f"[{citation_id}] "
            f"{citation['source']} | "
            f"Page {citation['page']} | "
            f"{citation['chunk_id']}"
        )

    print("\nCITATION VALIDATION")
    print("=" * 100)

    print(
        result[
            "citation_validation"
        ]
    )