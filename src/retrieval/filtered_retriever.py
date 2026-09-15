from pathlib import Path
from typing import Any, Dict, Optional

from src.indexing.incremental_indexer import (
    load_incremental_vector_store,
)
from src.vectorstore.qdrant_store import (
    build_metadata_filter,
)


def search(
    query: str,
    k: int = 5,
    filters: Optional[
        Dict[str, Any]
    ] = None,
):
    """
    Dense retrieval with optional Qdrant metadata filters.

    Example:

    filters={
        "document_id":
            "05_informed_consent_guidance"
    }
    """

    vector_store = (
        load_incremental_vector_store()
    )

    try:

        query_filter = (
            build_metadata_filter(
                filters
            )
        )

        return (
            vector_store
            .similarity_search_with_score(
                query=query,
                k=k,
                filter=query_filter,
            )
        )

    finally:

        vector_store.client.close()


if __name__ == "__main__":

    query = (
        "What risks must be explained "
        "during informed consent?"
    )

    document_id = (
        "05_informed_consent_guidance"
    )

    filters = (
        {
            "document_id": document_id
        }
        if document_id
        else None
    )

    print(
        f"Query: {query}"
    )

    print(
        f"Filters: {filters}"
    )

    results = search(
        query=query,
        k=5,
        filters=filters,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FILTERED RETRIEVAL RESULTS"
    )

    print(
        "=" * 100
    )

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = (
            document.metadata
        )

        print(
            f"\nRESULT #{rank}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            f"Document ID: "
            f"{metadata.get('document_id')}"
        )

        print(
            f"Source: "
            f"{Path(metadata.get('source', '')).name}"
        )

        print(
            f"Page: "
            f"{metadata.get('page_number')}"
        )

        print(
            f"Chunk: "
            f"{metadata.get('chunk_id')}"
        )

        print("\nText:")
        print(
            document.page_content
        )