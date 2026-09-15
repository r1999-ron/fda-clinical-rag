from __future__ import annotations

import time
import uuid

from src.conversation.models import (
    ConversationMessage,
)
from src.conversation.query_rewriter import (
    rewrite_query,
)
from src.generation.generator import (
    generate_answer,
)
from src.observability.trace import (
    RAGTrace,
)


def generate_conversational_answer(
    question: str,
    history=None,
    filters=None,
    retrieval_k: int = 5,
    context_k: int = 5,
    session_id: str | None = None,
) -> dict:

    total_start = (time.perf_counter())

    trace = RAGTrace(request_id=str(uuid.uuid4()),
        session_id=session_id,
        original_question=question,
    )

    history = history or []

    # ========================================================
    # QUERY REWRITING
    # ========================================================

    rewrite_start = (time.perf_counter())

    rewrite_result = (
        rewrite_query(
            question=question,
            history=history,
        )
    )

    rewrite_end = (time.perf_counter())

    trace.rewrite_latency_ms = (
        (
            rewrite_end
            - rewrite_start
        )
        * 1000
    )

    standalone_query = (
        rewrite_result
        .standalone_query
    )

    trace.retrieval_query = (
        standalone_query
    )

    trace.query_rewritten = (
        rewrite_result.rewritten
    )

    # ========================================================
    # RAG
    # ========================================================

    try:
        rag_result = (
            generate_answer(
                question=standalone_query,
                retrieval_k=retrieval_k,
                rerank_k=context_k,
                use_reranker=False,
                filters=filters,
                trace=trace,
            )
        )

    except Exception as exc:
        trace.error = str(exc)
        trace.total_latency_ms = (
                (
                        time.perf_counter()
                        - total_start
                )
                * 1000
        )

        measured_latency = (
                trace.rewrite_latency_ms
                + trace.setup_latency_ms
                + trace.retrieval_latency_ms
                + trace.context_build_latency_ms
                + trace.generation_latency_ms
        )

        trace.overhead_latency_ms = max(
            0.0,
            trace.total_latency_ms
            - measured_latency,
        )

        raise

    # ========================================================
    # SUCCESS-PATH TOTAL LATENCY
    # ========================================================

    trace.total_latency_ms = (
            (
                    time.perf_counter()
                    - total_start
            )
            * 1000
    )

    # ========================================================
    # UNACCOUNTED OVERHEAD
    # ========================================================

    measured_latency = (
            trace.rewrite_latency_ms
            + trace.setup_latency_ms
            + trace.retrieval_latency_ms
            + trace.context_build_latency_ms
            + trace.generation_latency_ms
    )

    trace.overhead_latency_ms = max(
        0.0,
        trace.total_latency_ms
        - measured_latency,
    )

    return {
        "original_question": question,

        "retrieval_query": (
            standalone_query
        ),

        "query_rewritten": (
            rewrite_result.rewritten
        ),

        **rag_result,

        "trace": (
            trace.to_dict()
        ),
    }

if __name__ == "__main__":

    # ========================================================
    # TURN 1
    # ========================================================

    first_question = (
        "What risks must be explained "
        "during informed consent?"
    )

    filters = {
        "document_id":
            "05_informed_consent_guidance"
    }

    first_result = (
        generate_conversational_answer(
            question=first_question,
            history=[],
            filters=filters,
        )
    )

    print("\n"+ "=" * 100)

    print("TURN 1")

    print("=" * 100)

    print(
        f"User: "
        f"{first_question}"
    )

    print(
        "\nRetrieval query:"
    )

    print(
        first_result[
            "retrieval_query"
        ]
    )

    print(
        "\nAssistant:"
    )

    print(
        first_result[
            "answer"
        ]
    )

    # ========================================================
    # BUILD HISTORY
    # ========================================================

    history = [
        ConversationMessage(
            role="user",
            content=first_question,
        ),
        ConversationMessage(
            role="assistant",
            content=first_result[
                "answer"
            ],
        ),
    ]

    # ========================================================
    # TURN 2
    # ========================================================

    follow_up = (
        "What about alternatives?"
    )

    second_result = (
        generate_conversational_answer(
            question=follow_up,
            history=history,
            filters=filters,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TURN 2"
    )

    print(
        "=" * 100
    )

    print(
        f"User: "
        f"{follow_up}"
    )

    print(
        "\nRewritten:"
    )

    print(
        second_result[
            "query_rewritten"
        ]
    )

    print(
        "\nRetrieval query:"
    )

    print(
        second_result[
            "retrieval_query"
        ]
    )

    print(
        "\nAssistant:"
    )

    print(
        second_result[
            "answer"
        ]
    )

    print(
        "\nRetrieved chunks:"
    )

    for chunk in second_result[
        "retrieved_chunks"
    ]:

        print(
            f"- "
            f"{chunk['source']} | "
            f"Page {chunk['page']} | "
            f"{chunk['chunk_id']}"
        )