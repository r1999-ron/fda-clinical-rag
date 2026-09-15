import json
from pathlib import Path
from typing import List, Dict, Any

from src.retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"

DENSE_K = 5
BM25_K = 5
HYBRID_K = 5


def load_questions() -> List[Dict[str, Any]]:
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_filename(source: str) -> str:
    return Path(source).name


def is_relevant(
    document,
    evaluation_item: Dict[str, Any],
) -> bool:
    metadata = document.metadata

    source = get_filename(metadata.get("source", ""))

    page_number = metadata.get("page_number")

    expected_source = (evaluation_item["expected_source"])

    expected_pages = (evaluation_item["expected_pages"])

    expected_terms = (evaluation_item.get("expected_terms",[],))

    source_matches = (source == expected_source)

    page_matches = (page_number in expected_pages)

    text = document.page_content.lower()

    if expected_terms:
        term_matches = any(
            term.lower() in text
            for term in expected_terms
        )
    else:
        term_matches = True

    return (
        source_matches
        and page_matches
        and term_matches
    )


def evaluate_ranked_results(
    results,
    evaluation_item,
):
    relevant_rank = None

    retrieved_results = []

    for rank, (document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        relevant = is_relevant(
            document=document,
            evaluation_item=evaluation_item,
        )

        if (
            relevant
            and relevant_rank is None
        ):
            relevant_rank = rank

        retrieved_results.append(
            {
                "rank": rank,
                "score": float(score),
                "source": get_filename(
                    document.metadata.get(
                        "source",
                        "",
                    )
                ),
                "page": document.metadata.get(
                    "page_number"
                ),
                "chunk_id": document.metadata.get(
                    "chunk_id"
                ),
                "relevant": relevant,
            }
        )

    return {
        "relevant_rank": relevant_rank,

        "hit_at_1": (
            relevant_rank is not None
            and relevant_rank <= 1
        ),

        "hit_at_3": (
            relevant_rank is not None
            and relevant_rank <= 3
        ),

        "hit_at_5": (
            relevant_rank is not None
            and relevant_rank <= 5
        ),

        "reciprocal_rank": (
            1 / relevant_rank
            if relevant_rank is not None
            else 0
        ),

        "retrieved_results": (
            retrieved_results
        ),
    }


# ============================================================
# DENSE
# ============================================================

def evaluate_dense(
    retriever: HybridRetriever,
    evaluation_item,
):
    question = evaluation_item["question"]

    original_k = retriever.dense_k

    retriever.dense_k = DENSE_K

    try:
        results = retriever.dense_search(
            question
        )
    finally:
        retriever.dense_k = original_k

    evaluation = evaluate_ranked_results(
        results=results,
        evaluation_item=evaluation_item,
    )

    evaluation["question"] = question

    return evaluation


# ============================================================
# BM25
# ============================================================

def evaluate_bm25(
    retriever: HybridRetriever,
    evaluation_item,
):
    question = evaluation_item["question"]

    original_k = retriever.bm25_k

    retriever.bm25_k = BM25_K

    try:
        results = retriever.bm25_search(
            question
        )
    finally:
        retriever.bm25_k = original_k

    evaluation = evaluate_ranked_results(
        results=results,
        evaluation_item=evaluation_item,
    )

    evaluation["question"] = question

    return evaluation


# ============================================================
# HYBRID
# ============================================================

def evaluate_hybrid(
    retriever: HybridRetriever,
    evaluation_item,
):
    question = evaluation_item["question"]

    original_final_k = retriever.final_k

    retriever.final_k = HYBRID_K

    try:
        results = retriever.search(
            question
        )
    finally:
        retriever.final_k = original_final_k

    evaluation = evaluate_ranked_results(
        results=results,
        evaluation_item=evaluation_item,
    )

    evaluation["question"] = question

    return evaluation


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:

    total = len(results)

    if total == 0:
        return {
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "mrr": 0,
        }

    return {
        "hit_at_1": sum(
            1
            for result in results
            if result["hit_at_1"]
        ) / total,

        "hit_at_3": sum(
            1
            for result in results
            if result["hit_at_3"]
        ) / total,

        "hit_at_5": sum(
            1
            for result in results
            if result["hit_at_5"]
        ) / total,

        "mrr": sum(
            result["reciprocal_rank"]
            for result in results
        ) / total,
    }


def print_metrics(
    title: str,
    metrics: Dict[str, float],
):
    print("\n")
    print("#" * 100)
    print(title)
    print("#" * 100)

    print(
        f"Hit@1 : "
        f"{metrics['hit_at_1']:.2%}"
    )

    print(
        f"Hit@3 : "
        f"{metrics['hit_at_3']:.2%}"
    )

    print(
        f"Hit@5 : "
        f"{metrics['hit_at_5']:.2%}"
    )

    print(
        f"MRR   : "
        f"{metrics['mrr']:.4f}"
    )


# ============================================================
# PER QUESTION COMPARISON
# ============================================================

def print_query_comparison(
    question,
    dense_result,
    bm25_result,
    hybrid_result,
):
    print("\n" + "=" * 100)

    print(
        f"QUESTION: {question}"
    )

    print("=" * 100)

    print(
        f"Dense relevant rank : "
        f"{dense_result['relevant_rank']}"
    )

    print(
        f"BM25 relevant rank  : "
        f"{bm25_result['relevant_rank']}"
    )

    print(
        f"Hybrid relevant rank: "
        f"{hybrid_result['relevant_rank']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    questions = load_questions()

    print(
        f"Loaded "
        f"{len(questions)} "
        f"evaluation questions."
    )

    retriever = HybridRetriever(
        dense_k=10,
        bm25_k=10,
        final_k=5,
        rrf_k=60,
    )

    dense_results = []
    bm25_results = []
    hybrid_results = []

    try:

        for evaluation_item in questions:

            dense_result = (
                evaluate_dense(
                    retriever=retriever,
                    evaluation_item=(
                        evaluation_item
                    ),
                )
            )

            bm25_result = (
                evaluate_bm25(
                    retriever=retriever,
                    evaluation_item=(
                        evaluation_item
                    ),
                )
            )

            hybrid_result = (
                evaluate_hybrid(
                    retriever=retriever,
                    evaluation_item=(
                        evaluation_item
                    ),
                )
            )

            dense_results.append(
                dense_result
            )

            bm25_results.append(
                bm25_result
            )

            hybrid_results.append(
                hybrid_result
            )

            print_query_comparison(
                question=(
                    evaluation_item[
                        "question"
                    ]
                ),
                dense_result=(
                    dense_result
                ),
                bm25_result=(
                    bm25_result
                ),
                hybrid_result=(
                    hybrid_result
                ),
            )

        # ====================================================
        # FINAL METRICS
        # ====================================================

        dense_metrics = (
            calculate_metrics(
                dense_results
            )
        )

        bm25_metrics = (
            calculate_metrics(
                bm25_results
            )
        )

        hybrid_metrics = (
            calculate_metrics(
                hybrid_results
            )
        )

        print_metrics(
            title="DENSE-ONLY RESULTS",
            metrics=dense_metrics,
        )

        print_metrics(
            title="BM25-ONLY RESULTS",
            metrics=bm25_metrics,
        )

        print_metrics(
            title="HYBRID RRF RESULTS",
            metrics=hybrid_metrics,
        )

        # ====================================================
        # HYBRID VS DENSE
        # ====================================================

        print("\n")
        print("#" * 100)
        print("HYBRID VS DENSE")
        print("#" * 100)

        print(
            f"Hit@1 change : "
            f"{hybrid_metrics['hit_at_1'] - dense_metrics['hit_at_1']:+.2%}"
        )

        print(
            f"Hit@3 change : "
            f"{hybrid_metrics['hit_at_3'] - dense_metrics['hit_at_3']:+.2%}"
        )

        print(
            f"Hit@5 change : "
            f"{hybrid_metrics['hit_at_5'] - dense_metrics['hit_at_5']:+.2%}"
        )

        print(
            f"MRR change   : "
            f"{hybrid_metrics['mrr'] - dense_metrics['mrr']:+.4f}"
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()