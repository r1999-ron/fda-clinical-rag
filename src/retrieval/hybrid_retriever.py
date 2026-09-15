from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.indexing.incremental_indexer import (
    load_incremental_vector_store,
    process_pdf,
    DATA_DIR,
)
from src.vectorstore.qdrant_store import build_metadata_filter

REGULATORY_ID_PATTERN = re.compile(
    r"""
    \b\d+\s+CFR\s+
    \d+\.\d+
    (?:\([a-z0-9]+\))*
    """,
    re.IGNORECASE | re.VERBOSE,
)

@lru_cache(maxsize=1)
def get_routed_retriever():
    """
    Build Dense + BM25 retrieval resources once
    and reuse them across API requests.
    """

    return HybridRetriever(
        dense_k=10,
        bm25_k=10,
        final_k=10,
    )

class HybridRetriever:
    """
    Dense retrieval:
        Qdrant embeddings

    Sparse retrieval:
        BM25 lexical search

    Fusion:
        Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        dense_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 10,
        rrf_k: int = 60,
    ):
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rrf_k = rrf_k

        print("Loading dense vector store...")

        self.vector_store = (
            load_incremental_vector_store()
        )

        print("Preparing BM25 corpus...")

        self.documents = self._load_chunks()

        self.tokenized_corpus = [
            self._tokenize(
                document.page_content
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print(
            f"BM25 ready with "
            f"{len(self.documents)} chunks."
        )

    def _load_chunks(
            self,
    ) -> List[Document]:
        """
        Build the BM25 corpus using the exact same
        document-processing pipeline as incremental indexing.

        This guarantees that BM25 and Qdrant share:

            document_id
            page_number
            chunk_id
            section

        and therefore refer to the same physical chunks.
        """

        chunks = []

        pdf_files = sorted(
            DATA_DIR.glob(
                "*.pdf"
            )
        )

        print(
            f"Preparing BM25 from "
            f"{len(pdf_files)} PDFs..."
        )

        for pdf_path in pdf_files:
            print(
                f"BM25 processing: "
                f"{pdf_path.name}"
            )

            document_chunks = (
                process_pdf(
                    pdf_path
                )
            )

            chunks.extend(
                document_chunks
            )

        print(
            f"BM25 corpus contains "
            f"{len(chunks)} chunks."
        )

        return chunks

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()

        return re.findall(
            r"""
            \d+\.\d+(?:\([a-z0-9]+\))* |  # 50.25(a)(2)
            fda-\d+-[a-z0-9]+-\d+      |  # FDA-2011-D-0469
            [a-z]+(?:-[a-z]+)*          |  # research-related
            \d+                             # 21, 45
            """,
            text,
            re.VERBOSE,
        )

    @staticmethod
    def _matches_filters(
            document: Document,
            filters: dict | None,
    ) -> bool:

        if not filters:
            return True

        metadata = (
            document.metadata
        )

        for key, expected_value in (
                filters.items()
        ):

            if expected_value is None:
                continue

            actual_value = (
                metadata.get(
                    key
                )
            )

            if actual_value != expected_value:
                return False

        return True

    @staticmethod
    def has_regulatory_identifier(
            query: str,
    ) -> bool:
        """
        Detect exact regulatory references such as:

            21 CFR 50.25(a)(2)
            45 CFR 46.116(b)(2)
            21 CFR 312.32
        """

        return bool(
            REGULATORY_ID_PATTERN.search(
                query
            )
        )

    @staticmethod
    def _chunk_location(
            document: Document,
    ):
        metadata = document.metadata

        document_id = metadata.get(
            "document_id"
        )

        page_number = metadata.get(
            "page_number"
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if not (
                document_id
                and page_number is not None
                and chunk_id
        ):
            return None

        match = re.search(
            r"_page_(\d+)_chunk_(\d+)$",
            chunk_id,
        )

        if not match:
            return None

        return {
            "document_id": document_id,
            "page_number": int(
                match.group(1)
            ),
            "chunk_number": int(
                match.group(2)
            ),
        }

    def expand_with_neighbors(
            self,
            results,
            neighbors_before: int = 3,
            neighbors_after: int = 1,
            max_results: int = 10,
    ):

        if not results:
            return []

        documents_by_id = {
            document.metadata.get(
                "chunk_id"
            ): document
            for document in self.documents
            if document.metadata.get(
                "chunk_id"
            )
        }

        expanded = []
        seen_chunk_ids = set()

        for document, score in results:

            location = (
                self._chunk_location(
                    document
                )
            )

            if not location:
                continue

            document_id = (
                location[
                    "document_id"
                ]
            )

            page_number = (
                location[
                    "page_number"
                ]
            )

            chunk_number = (
                location[
                    "chunk_number"
                ]
            )

            # ====================================================
            # PRESERVE ORIGINAL DOCUMENT ORDER
            # ====================================================

            start_chunk = max(
                1,
                chunk_number
                - neighbors_before,
            )

            end_chunk = (
                    chunk_number
                    + neighbors_after
            )

            neighbor_numbers = list(
                range(
                    start_chunk,
                    end_chunk + 1,
                )
            )

            # ====================================================
            # RESOLVE NEIGHBOR CHUNKS
            # ====================================================

            for neighbor_number in (
                    neighbor_numbers
            ):

                neighbor_chunk_id = (
                    f"{document_id}"
                    f"_page_{page_number}"
                    f"_chunk_{neighbor_number}"
                )

                neighbor_document = (
                    documents_by_id.get(
                        neighbor_chunk_id
                    )
                )

                if not neighbor_document:
                    continue

                if (
                        neighbor_chunk_id
                        in seen_chunk_ids
                ):
                    continue

                seen_chunk_ids.add(
                    neighbor_chunk_id
                )

                distance = abs(
                    neighbor_number
                    - chunk_number
                )

                neighbor_score = (
                    float(score)
                    if distance == 0
                    else float(score)
                         - (
                                 distance
                                 * 0.001
                         )
                )

                expanded.append(
                    (
                        neighbor_document,
                        neighbor_score,
                    )
                )

                if (
                        len(expanded)
                        >= max_results
                ):
                    return expanded

        return expanded

    # ========================================================
    # DENSE RETRIEVAL
    # ========================================================

    def dense_search(
            self,
            query: str,
            filters: dict | None = None,
    ) -> List[Tuple[Document, float]]:

        query_filter = (
            build_metadata_filter(
                filters
            )
        )

        return (
            self.vector_store
            .similarity_search_with_score(
                query=query,
                k=self.dense_k,
                filter=query_filter,
            )
        )

    # ========================================================
    # BM25 RETRIEVAL
    # ========================================================

    def bm25_search(
            self,
            query: str,
            filters: dict | None = None,
    ) -> List[Tuple[Document, float]]:

        tokenized_query = (
            self._tokenize(
                query
            )
        )

        scores = (
            self.bm25.get_scores(
                tokenized_query
            )
        )

        ranked_indexes = sorted(
            range(
                len(scores)
            ),
            key=lambda index: scores[
                index
            ],
            reverse=True,
        )

        results = []

        for index in ranked_indexes:

            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            document = (
                self.documents[
                    index
                ]
            )

            # Respect document/page/etc filters.
            if not self._matches_filters(
                    document=document,
                    filters=filters,
            ):
                continue

            results.append(
                (
                    document,
                    score,
                )
            )

            if (
                    len(results)
                    >= self.bm25_k
            ):
                break

        return results

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    def get_retrieval_strategy(
            self,
            query: str,
    ) -> str:
        """
        Decide which retrieval strategy should handle the query.

        Exact CFR section/subsection:
            BM25

        Semantic / broad queries:
            Dense
        """

        if self.has_regulatory_identifier(
                query
        ):
            return "bm25"

        return "dense"

    def routed_search(
            self,
            query: str,
            filters: dict | None = None,
    ):
        """
        Retrieval policy:

        Exact CFR section/subsection:
            BM25 + neighboring context expansion

        Everything else:
            Dense semantic retrieval
        """

        strategy = (
            self.get_retrieval_strategy(
                query
            )
        )

        # ========================================================
        # EXACT REGULATORY IDENTIFIER
        # ========================================================

        if strategy == "bm25":

            print(
                "Retrieval route: BM25 + CONTEXT EXPANSION "
                "(regulatory identifier detected)"
            )

            bm25_results = (
                self.bm25_search(
                    query=query,
                    filters=filters,
                )
            )

            if not bm25_results:
                return self.dense_search(
                    query=query,
                    filters=filters,
                )

            # Only the strongest exact-match anchor.
            anchor_results = (
                bm25_results[:1]
            )

            return self.expand_with_neighbors(
                results=anchor_results,
                neighbors_before=3,
                neighbors_after=1,
                max_results=5,
            )

        # ========================================================
        # SEMANTIC QUERY
        # ========================================================

        print(
            "Retrieval route: DENSE "
            "(semantic query)"
        )

        return self.dense_search(
            query=query,
            filters=filters,
        )

    @staticmethod
    def _document_key(
            document: Document,
    ) -> str:

        metadata = (
            document.metadata
        )

        document_id = (
            metadata.get(
                "document_id"
            )
        )

        chunk_id = (
            metadata.get(
                "chunk_id"
            )
        )

        if (
                document_id
                and chunk_id
        ):
            return (
                f"{document_id}|"
                f"{chunk_id}"
            )

        # Defensive fallback.
        source = Path(
            metadata.get(
                "source",
                "unknown",
            )
        ).name

        page = metadata.get(
            "page_number"
        )

        return (
            f"{source}|"
            f"{page}|"
            f"{chunk_id}"
        )

    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    def reciprocal_rank_fusion(
            self,
            dense_results,
            bm25_results,
    ):
        fused_scores = defaultdict(float)

        documents_by_key = {}

        dense_rank_by_key = {}
        bm25_rank_by_key = {}

        dense_score_by_key = {}
        bm25_score_by_key = {}

        # ========================================================
        # DENSE
        # ========================================================

        for rank, (
                document,
                dense_score,
        ) in enumerate(
            dense_results,
            start=1,
        ):
            key = self._document_key(
                document
            )

            fused_scores[key] += (
                    1 / (
                    self.rrf_k
                    + rank
            )
            )

            # Prefer the dense/Qdrant document as canonical.
            documents_by_key[key] = document

            dense_rank_by_key[key] = rank

            dense_score_by_key[key] = float(
                dense_score
            )

        # ========================================================
        # BM25
        # ========================================================

        for rank, (
                document,
                bm25_score,
        ) in enumerate(
            bm25_results,
            start=1,
        ):
            key = self._document_key(
                document
            )

            fused_scores[key] += (
                    1 / (
                    self.rrf_k
                    + rank
            )
            )

            # Only use the BM25 document if the chunk
            # was not already returned by dense search.
            if key not in documents_by_key:
                documents_by_key[key] = document

            bm25_rank_by_key[key] = rank

            bm25_score_by_key[key] = float(
                bm25_score
            )

        # ========================================================
        # SORT
        # ========================================================

        ranked_keys = sorted(
            fused_scores.keys(),
            key=lambda key: fused_scores[key],
            reverse=True,
        )

        fused_results = []

        for key in ranked_keys[
                   :self.final_k
                   ]:
            document = (
                documents_by_key[key]
            )

            rrf_score = float(
                fused_scores[key]
            )

            document.metadata[
                "dense_rank"
            ] = dense_rank_by_key.get(
                key
            )

            document.metadata[
                "bm25_rank"
            ] = bm25_rank_by_key.get(
                key
            )

            document.metadata[
                "dense_score"
            ] = dense_score_by_key.get(
                key
            )

            document.metadata[
                "bm25_score"
            ] = bm25_score_by_key.get(
                key
            )

            document.metadata[
                "rrf_score"
            ] = rrf_score

            fused_results.append(
                (
                    document,
                    rrf_score,
                )
            )

        return fused_results

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def search(
        self,
        query: str,
    ):

        dense_results = (
            self.dense_search(
                query
            )
        )

        bm25_results = (
            self.bm25_search(
                query
            )
        )

        fused_results = (
            self.reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
            )
        )

        return fused_results

    def close(self):
        self.vector_store.client.close()

    def validate_chunk_identity(
            self,
            query: str,
    ):
        """
        Debug helper.

        Shows whether chunks returned by both retrievers
        resolve to identical deterministic keys.
        """

        dense_results = (
            self.dense_search(
                query
            )
        )

        bm25_results = (
            self.bm25_search(
                query
            )
        )

        dense_keys = {
            self._document_key(
                document
            )
            for document, _
            in dense_results
        }

        bm25_keys = {
            self._document_key(
                document
            )
            for document, _
            in bm25_results
        }

        overlap = (
                dense_keys
                & bm25_keys
        )

        print(
            "\n"
            + "=" * 100
        )

        print(
            "CHUNK IDENTITY CHECK"
        )

        print(
            "=" * 100
        )

        print(
            f"Dense results: "
            f"{len(dense_keys)}"
        )

        print(
            f"BM25 results: "
            f"{len(bm25_keys)}"
        )

        print(
            f"Shared chunks: "
            f"{len(overlap)}"
        )

        for key in sorted(
                overlap
        ):
            print(
                f"MATCH: {key}"
            )


if __name__ == "__main__":

    queries = [

        (
            "What risks must be explained "
            "during informed consent?"
        ),

        (
            "What is a master protocol?"
        ),

        (
            "What does "
            "21 CFR 50.25(a)(2) require?"
        ),
    ]

    retriever = HybridRetriever(
        dense_k=10,
        bm25_k=10,
        final_k=5,
    )

    try:

        for query in queries:

            print(
                "\n"
                + "=" * 100
            )

            print(
                f"QUERY: {query}"
            )

            print(
                "=" * 100
            )

            results = (
                retriever.routed_search(
                    query
                )
            )

            for rank, (
                document,
                score,
            ) in enumerate(
                results[:5],
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
                    "Chunk:",
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

    finally:

        retriever.close()