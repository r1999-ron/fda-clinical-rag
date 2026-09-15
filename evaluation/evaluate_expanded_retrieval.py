import json
from collections import defaultdict
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


def load_questions():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def source_name(
    document,
) -> str:

    return Path(
        document.metadata.get(
            "source",
            ""
        )
    ).name


def find_first_relevant_rank(
    results,
    expected_sources,
    expected_chunk_ids=None,
):

    expected_sources = set(
        expected_sources or []
    )

    expected_chunk_ids = set(
        expected_chunk_ids or []
    )

    for rank, (
        document,
        _
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = (
            document.metadata
        )

        chunk_id = (
            metadata.get(
                "chunk_id"
            )
        )

        source = Path(
            metadata.get(
                "source",
                ""
            )
        ).name

        # ==========================================
        # STRICT CHUNK-LEVEL EVALUATION
        # ==========================================

        if expected_chunk_ids:

            if (
                chunk_id
                in expected_chunk_ids
            ):
                return rank

        # ==========================================
        # SOURCE-LEVEL FALLBACK
        # ==========================================

        elif (
            source
            in expected_sources
        ):
            return rank

    return None


def evaluate_results(
    results,
    expected_sources,
    expected_chunk_ids=None,
):

    rank = (
        find_first_relevant_rank(
            results=results,
            expected_sources=expected_sources,
            expected_chunk_ids=expected_chunk_ids,
        )
    )

    return {
        "rank": rank,

        "hit_1": (
            rank is not None
            and rank <= 1
        ),

        "hit_3": (
            rank is not None
            and rank <= 3
        ),

        "hit_5": (
            rank is not None
            and rank <= 5
        ),

        "reciprocal_rank": (
            1 / rank
            if rank
            else 0.0
        ),
    }


def calculate_summary(
    results,
):

    total = len(
        results
    )

    if total == 0:
        return {
            "total": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
        }

    return {
        "total":
            total,

        "hit_at_1":
            sum(
                item[
                    "hit_1"
                ]
                for item in results
            )
            / total,

        "hit_at_3":
            sum(
                item[
                    "hit_3"
                ]
                for item in results
            )
            / total,

        "hit_at_5":
            sum(
                item[
                    "hit_5"
                ]
                for item in results
            )
            / total,

        "mrr":
            sum(
                item[
                    "reciprocal_rank"
                ]
                for item in results
            )
            / total,
    }


def print_summary(
    name,
    summary,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        name
    )

    print(
        "=" * 70
    )

    print(
        f"Questions : "
        f"{summary['total']}"
    )

    print(
        f"Hit@1     : "
        f"{summary['hit_at_1'] * 100:.2f}%"
    )

    print(
        f"Hit@3     : "
        f"{summary['hit_at_3'] * 100:.2f}%"
    )

    print(
        f"Hit@5     : "
        f"{summary['hit_at_5'] * 100:.2f}%"
    )

    print(
        f"MRR       : "
        f"{summary['mrr']:.4f}"
    )


def main():

    questions = (
        load_questions()
    )

    answerable_questions = [
        item
        for item in questions
        if item.get(
            "answerable"
        )
    ]

    print(
        f"Total dataset: "
        f"{len(questions)}"
    )

    print(
        f"Retrieval questions: "
        f"{len(answerable_questions)}"
    )

    retriever = HybridRetriever(
        dense_k=10,
        bm25_k=10,
        final_k=10,
    )

    results_by_retriever = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
        "routed": [],
    }

    category_results = (
        defaultdict(
            lambda: {
                "dense": [],
                "bm25": [],
                "hybrid": [],
                "routed": [],
            }
        )
    )

    try:

        for index, item in enumerate(
            answerable_questions,
            start=1,
        ):

            question = (
                item[
                    "question"
                ]
            )

            expected_sources = (
                item[
                    "expected_sources"
                ]
            )

            expected_chunk_ids = (
                item.get(
                    "expected_chunk_ids",
                    [],
                )
            )

            query_type = (
                item.get(
                    "query_type",
                    "unknown",
                )
            )

            print(
                "\n"
                + "#" * 100
            )

            print(
                f"{index}/"
                f"{len(answerable_questions)}"
            )

            print(
                f"ID: "
                f"{item['id']}"
            )

            print(
                f"Type: "
                f"{query_type}"
            )

            print(
                f"Question: "
                f"{question}"
            )

            print(
                "Expected sources:",
                expected_sources,
            )

            print(
                "Expected chunks:",
                expected_chunk_ids,
            )

            # ==========================================
            # DENSE
            # ==========================================

            dense_results = (
                retriever.dense_search(
                    question
                )
            )

            dense_eval = (
                evaluate_results(
                    results=dense_results,
                    expected_sources=expected_sources,
                    expected_chunk_ids=expected_chunk_ids,
                )
            )

            # ==========================================
            # BM25
            # ==========================================

            bm25_results = (
                retriever.bm25_search(
                    question
                )
            )

            bm25_eval = (
                evaluate_results(
                    results=bm25_results,
                    expected_sources=expected_sources,
                    expected_chunk_ids=expected_chunk_ids,
                )
            )

            # ==========================================
            # HYBRID
            # ==========================================

            hybrid_results = (
                retriever.search(
                    question
                )
            )

            hybrid_eval = (
                evaluate_results(
                    results=hybrid_results,
                    expected_sources=expected_sources,
                    expected_chunk_ids=expected_chunk_ids,
                )
            )

            # ==========================================
            # ROUTED
            # ==========================================

            routed_results = (
                retriever.routed_search(
                    question
                )
            )

            routed_eval = (
                evaluate_results(
                    results=routed_results,
                    expected_sources=expected_sources,
                    expected_chunk_ids=expected_chunk_ids,
                )
            )

            evaluations = {
                "dense":
                    dense_eval,

                "bm25":
                    bm25_eval,

                "hybrid":
                    hybrid_eval,

                "routed":
                    routed_eval,
            }

            for (
                retriever_name,
                evaluation,
            ) in evaluations.items():

                results_by_retriever[
                    retriever_name
                ].append(
                    evaluation
                )

                category_results[
                    query_type
                ][
                    retriever_name
                ].append(
                    evaluation
                )

            print(
                "Ranks:",
                {
                    name:
                        value["rank"]
                    for name, value
                    in evaluations.items()
                },
            )

        # ==============================================
        # OVERALL RESULTS
        # ==============================================

        print(
            "\n\n"
            + "#" * 100
        )

        print(
            "OVERALL RETRIEVAL RESULTS"
        )

        print(
            "#" * 100
        )

        for name in [
            "dense",
            "bm25",
            "hybrid",
            "routed",
        ]:

            summary = (
                calculate_summary(
                    results_by_retriever[
                        name
                    ]
                )
            )

            print_summary(
                name.upper(),
                summary,
            )

        # ==============================================
        # RESULTS BY QUERY TYPE
        # ==============================================

        print(
            "\n\n"
            + "#" * 100
        )

        print(
            "RESULTS BY QUERY TYPE"
        )

        print(
            "#" * 100
        )

        for (
            query_type,
            retriever_results,
        ) in category_results.items():

            print(
                "\n"
                + "*" * 100
            )

            print(
                f"QUERY TYPE: "
                f"{query_type}"
            )

            print(
                "*" * 100
            )

            for name in [
                "dense",
                "bm25",
                "hybrid",
                "routed",
            ]:

                summary = (
                    calculate_summary(
                        retriever_results[
                            name
                        ]
                    )
                )

                print_summary(
                    name.upper(),
                    summary,
                )

    finally:

        retriever.close()


if __name__ == "__main__":
    main()