from src.indexing.incremental_indexer import load_or_create_vector_store


def search(vector_store, query: str, k: int = 5):
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    print("\n" + "#" * 100)
    print(f"QUERY: {query}")
    print("#" * 100)

    for index, (document, score) in enumerate(results, start=1):
        print("\n" + "=" * 80)

        print(f"RESULT #{index}")
        print(f"Score: {score:.4f}")

        print(
            "Document ID:",
            document.metadata.get("document_id")
        )

        print(
            "Source:",
            document.metadata.get("source")
        )

        print(
            "Page:",
            document.metadata.get("page_number")
        )

        print(
            "Chunk ID:",
            document.metadata.get("chunk_id")
        )

        print(
            "Section:",
            document.metadata.get("section")
        )

        print("\nText:")
        print(document.page_content)

    return results


if __name__ == "__main__":

    vector_store = (
        load_or_create_vector_store()
    )

    queries = [

        # Existing corpus
        (
            "What risks must be explained "
            "during informed consent?"
        ),

        (
            "Can a participant withdraw "
            "from a clinical trial?"
        ),

        (
            "What does "
            "21 CFR 50.25(a)(2) require?"
        ),

        # New corpus
        (
            "What is a master protocol?"
        ),

        (
            "What are the responsibilities "
            "of a clinical investigator?"
        ),

        (
            "What safety reporting responsibilities "
            "does a sponsor have?"
        ),

        (
            "What are the responsibilities "
            "of an IRB?"
        ),

        (
            "How should clinical trial "
            "safety be monitored?"
        ),

        (
            "What controls are required "
            "for electronic clinical records?"
        ),

        (
            "What are the requirements "
            "for postmarketing safety reporting?"
        ),
    ]

    try:

        for query in queries:

            search(
                vector_store=vector_store,
                query=query,
                k=5,
            )

    finally:

        # IMPORTANT:
        # Keep this only if load_or_create_vector_store()
        # is NOT your cached application-level store.
        vector_store.client.close()