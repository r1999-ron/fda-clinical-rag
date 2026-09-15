from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# MODEL_NAME = (
#     "cross-encoder/"
#     "ms-marco-MiniLM-L-6-v2"
# )

MODEL_NAME = "BAAI/bge-reranker-large"


class CrossEncoderReranker:

    def __init__(self):

        print(
            f"Loading reranker: "
            f"{MODEL_NAME}"
        )

        self.model = CrossEncoder(
            MODEL_NAME
        )

    def rerank(
        self,
        query: str,
        results: List[
            Tuple[Document, float]
        ],
        top_n: int = 3,
    ) -> List[
        Tuple[Document, float]
    ]:

        if not results:
            return []

        pairs = [
            (
                query,
                document.page_content,
            )
            for document, _ in results
        ]

        scores = self.model.predict(
            pairs
        )

        reranked_results = []

        for (
            document_with_score,
            reranker_score,
        ) in zip(
            results,
            scores,
        ):

            document = (
                document_with_score[0]
            )

            dense_score = (
                document_with_score[1]
            )

            document.metadata[
                "dense_score"
            ] = float(
                dense_score
            )

            document.metadata[
                "reranker_score"
            ] = float(
                reranker_score
            )

            reranked_results.append(
                (
                    document,
                    float(reranker_score),
                )
            )

        reranked_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return reranked_results[
            :top_n
        ]