from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.embeddings.embedder import (
    get_embedding_model,
)
from src.ingestion.cleaner import (
    clean_documents,
)
from src.ingestion.loader import (
    DATA_DIR,
    load_pdfs,
)
from src.ingestion.chunker import (
    chunk_documents,
    chunk_documents_structure_aware,
    chunk_documents_token_aware,
)

CHUNKING_STRATEGY = "structure"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

QDRANT_PATH = "./qdrant_data"

COLLECTION_NAME = (
    f"enterprise_{CHUNKING_STRATEGY}_"
    f"{CHUNK_SIZE}_{CHUNK_OVERLAP}"
)


def create_vector_store(
    chunks: List[Document],
):

    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )


def load_vector_store():
    embeddings = get_embedding_model()

    print(
        f"Loading collection: "
        f"{COLLECTION_NAME}"
    )

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )


def build_chunks(documents):

    if CHUNKING_STRATEGY == "character":

        return chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    if CHUNKING_STRATEGY == "token":

        return chunk_documents_token_aware(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    if CHUNKING_STRATEGY == "structure":

        return chunk_documents_structure_aware(
            documents,
            max_chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    raise ValueError(
        f"Unknown chunking strategy: "
        f"{CHUNKING_STRATEGY}"
    )

def build_metadata_filter(
    filters: Optional[
        Dict[str, Any]
    ] = None,
):
    """
    Build an AND-based Qdrant metadata filter.

    Example:

        {
            "document_id":
                "05_informed_consent_guidance",

            "page_number": 11
        }

    becomes:

        document_id == ...
        AND
        page_number == 11
    """

    if not filters:
        return None

    conditions = []

    for key, value in filters.items():

        if value is None:
            continue

        conditions.append(
            FieldCondition(
                key=f"metadata.{key}",
                match=MatchValue(
                    value=value
                ),
            )
        )

    if not conditions:
        return None

    return Filter(
        must=conditions
    )


if __name__ == "__main__":

    # 1. Load
    documents = load_pdfs(DATA_DIR)

    # 2. Clean
    documents = clean_documents(
        documents
    )

    # 3. Chunk
    chunks = build_chunks(
        documents
    )

    print("\n" + "=" * 80)
    print("INDEX CONFIGURATION")
    print("=" * 80)

    print(
        f"Strategy       : "
        f"{CHUNKING_STRATEGY}"
    )

    print(
        f"Chunk size     : "
        f"{CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap  : "
        f"{CHUNK_OVERLAP}"
    )

    print(
        f"Total chunks   : "
        f"{len(chunks)}"
    )

    print(
        f"Collection     : "
        f"{COLLECTION_NAME}"
    )

    # 4. Embed + persist to Qdrant
    vector_store = create_vector_store(
        chunks
    )

    print("\nIndexing complete.")

    vector_store.client.close()