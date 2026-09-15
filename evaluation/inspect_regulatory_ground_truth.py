import json
from pathlib import Path

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "expanded_retrieval_questions.json"
)


def load_regulatory_questions():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        questions = json.load(
            file
        )

    return [
        item
        for item in questions
        if item.get(
            "query_type"
        )
        == "regulatory_identifier"
    ]


def print_results(
    name,
    results,
    limit=10,
):

    print(
        "\n"
        + "-" * 100
    )

    print(
        name
    )

    print(
        "-" * 100
    )

    for rank, (
        document,
        score,
    ) in enumerate(
        results[:limit],
        start=1,
    ):

        metadata = (
            document.metadata
        )

        print(
            f"\nRESULT #{rank}"
        )

        print(
            "Document:",
            metadata.get(
                "document_id"
            ),
        )

        print(
            "Page:",
            metadata.get(
                "page_number"
            ),
        )

        print(
            "Chunk ID:",
            metadata.get(
                "chunk_id"
            ),
        )

        print(
            "Score:",
            score,
        )

        print(
            "\nText:"
        )

        print(
            document.page_content
        )


def main():

    questions = (
        load_regulatory_questions()
    )

    retriever = HybridRetriever(
        dense_k=10,
        bm25_k=10,
        final_k=10,
    )

    try:

        for item in questions:

            question = (
                item[
                    "question"
                ]
            )

            print(
                "\n\n"
                + "=" * 120
            )

            print(
                f"{item['id']} | "
                f"{question}"
            )

            print(
                "=" * 120
            )

            print(
                "Expected sources:",
                item.get(
                    "expected_sources",
                    [],
                ),
            )

            print(
                "Current expected chunk IDs:",
                item.get(
                    "expected_chunk_ids",
                    [],
                ),
            )

            dense_results = (
                retriever.dense_search(
                    question
                )
            )

            bm25_results = (
                retriever.bm25_search(
                    question
                )
            )

            hybrid_results = (
                retriever.search(
                    question
                )
            )

            print_results(
                name="DENSE",
                results=dense_results,
            )

            print_results(
                name="BM25",
                results=bm25_results,
            )

            print_results(
                name="HYBRID",
                results=hybrid_results,
            )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()