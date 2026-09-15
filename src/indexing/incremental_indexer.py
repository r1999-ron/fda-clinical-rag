import hashlib
from pathlib import Path
from typing import List

from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from qdrant_client import QdrantClient

from src.embeddings.embedder import get_embedding_model
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_pdfs, load_pdf
from src.ingestion.chunker import (
    chunk_documents_structure_aware,
)
from src.indexing.manifest import (
    load_manifest,
    save_manifest,
)

from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

QDRANT_PATH = "./qdrant_data"

COLLECTION_NAME = "enterprise_incremental_v1"


@lru_cache(maxsize=1)
def load_incremental_vector_store():

    print(
        f"Loading collection: "
        f"{COLLECTION_NAME}"
    )

    return (
        QdrantVectorStore
        .from_existing_collection(
            embedding=get_embedding_model(),
            path=QDRANT_PATH,
            collection_name=COLLECTION_NAME,
        )
    )


def remove_deleted_documents(
    vector_store,
    manifest,
    current_pdf_files,
):
    """
    Remove documents that still exist in the manifest/Qdrant
    but no longer exist in data/raw.
    """

    manifest_documents = manifest.get(
        "documents",
        {}
    )

    current_document_ids = {
        build_document_id(pdf_path)
        for pdf_path in current_pdf_files
    }

    manifest_document_ids = set(
        manifest_documents.keys()
    )

    removed_document_ids = (
        manifest_document_ids
        - current_document_ids
    )

    for document_id in removed_document_ids:

        record = manifest_documents.get(
            document_id,
            {}
        )

        filename = record.get(
            "filename",
            document_id,
        )

        print(
            f"REMOVED: {filename}"
        )

        delete_document_chunks(
            vector_store=vector_store,
            document_id=document_id,
        )

        del manifest_documents[
            document_id
        ]

        save_manifest(
            manifest
        )

        print(
            f"Removed from manifest: "
            f"{filename}"
        )

def load_or_create_vector_store():

    embeddings = get_embedding_model()

    client = QdrantClient(
        path=QDRANT_PATH
    )

    existing_collections = {
        collection.name
        for collection
        in client.get_collections().collections
    }

    if COLLECTION_NAME not in existing_collections:

        print(
            f"Collection "
            f"{COLLECTION_NAME} "
            f"does not exist."
        )

        print(
            "Creating collection..."
        )

        embedding_dimension = len(
            embeddings.embed_query(
                "dimension probe"
            )
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    else:

        print(
            f"Loading existing collection: "
            f"{COLLECTION_NAME}"
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    SHA-256 hash of the PDF.

    If file content changes,
    the hash changes.
    """

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(
                block
            )

    return sha256.hexdigest()


def build_document_id(
    pdf_path: Path,
) -> str:
    """
    Stable document ID.

    Filename is sufficient for this learning project.

    Later this could be:
    database ID / UUID / source-system ID.
    """

    return pdf_path.stem


def add_deterministic_chunk_ids(
    chunks: List[Document],
    document_id: str,
) -> List[Document]:
    """
    Avoid IDs such as:

        chunk_703

    because those depend on processing order.

    Instead:

        informed_consent_guidance_page_10_chunk_2
    """

    page_chunk_counter = {}

    for chunk in chunks:

        page_number = (
            chunk.metadata.get(
                "page_number"
            )
        )

        key = page_number

        page_chunk_counter[key] = (
            page_chunk_counter.get(
                key,
                0,
            )
            + 1
        )

        chunk_number = (
            page_chunk_counter[key]
        )

        chunk_id = (
            f"{document_id}"
            f"_page_{page_number}"
            f"_chunk_{chunk_number}"
        )

        chunk.metadata[
            "document_id"
        ] = document_id

        chunk.metadata[
            "chunk_id"
        ] = chunk_id

    return chunks


def process_pdf(
    pdf_path: Path,
) -> List[Document]:

    documents = load_pdf(
        pdf_path
    )

    documents = clean_documents(
        documents
    )

    chunks = (
        chunk_documents_structure_aware(
            documents,
            max_chunk_size=500,
            chunk_overlap=50,
        )
    )

    document_id = (
        build_document_id(
            pdf_path
        )
    )

    return add_deterministic_chunk_ids(
        chunks=chunks,
        document_id=document_id,
    )


def index_document(
    vector_store,
    chunks: List[Document],
):
    """
    Add document chunks to existing Qdrant collection.
    """

    vector_store.add_documents(
        documents=chunks
    )


def delete_document_chunks(
        vector_store,
        document_id: str,
):
    """
    Delete every existing Qdrant point belonging
    to one logical document.

    This prevents stale + new versions from existing
    together after a PDF changes.
    """

    print(
        f"Deleting old chunks for "
        f"document_id={document_id}"
    )

    vector_store.client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )
        ),
        wait=True,
    )

    print(
        f"Deleted old chunks for "
        f"document_id={document_id}"
    )


def backfill_manifest_metadata(
    manifest,
):
    """
    Backfill fields added after older manifest
    entries were already created.

    This does NOT re-index documents.
    """

    manifest_documents = manifest.get(
        "documents",
        {}
    )

    changed = False

    for (
        document_id,
        record,
    ) in manifest_documents.items():

        if not record.get(
            "indexed_at"
        ):

            record[
                "indexed_at"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            changed = True

            print(
                f"Backfilled indexed_at: "
                f"{document_id}"
            )

    if changed:
        save_manifest(
            manifest
        )


def main():

    manifest = load_manifest()

    backfill_manifest_metadata(
        manifest
    )

    manifest_documents = (
        manifest.setdefault(
            "documents",
            {}
        )
    )

    pdf_files = sorted(
        DATA_DIR.glob(
            "*.pdf"
        )
    )

    print(
        f"Found "
        f"{len(pdf_files)} PDFs."
    )

    vector_store = (
        load_or_create_vector_store()
    )

    try:

        remove_deleted_documents(
            vector_store=vector_store,
            manifest=manifest,
            current_pdf_files=pdf_files,
        )

        for pdf_path in pdf_files:

            document_id = (
                build_document_id(
                    pdf_path
                )
            )

            current_hash = (
                calculate_file_hash(
                    pdf_path
                )
            )

            existing_record = (
                manifest_documents.get(
                    document_id
                )
            )

            # ================================================
            # UNCHANGED
            # ================================================

            if (
                existing_record
                and existing_record.get(
                    "hash"
                )
                == current_hash
            ):

                print(
                    f"SKIP: "
                    f"{pdf_path.name} "
                    f"is unchanged."
                )

                continue

            # ================================================
            # NEW OR UPDATED
            # ================================================

            if existing_record:

                print(
                    f"UPDATED: "
                    f"{pdf_path.name}"
                )

                # IMPORTANT:
                # Remove the previous indexed version before
                # inserting the changed document.
                delete_document_chunks(
                    vector_store=vector_store,
                    document_id=document_id,
                )

            else:

                print(
                    f"NEW: "
                    f"{pdf_path.name}"
                )

            chunks = process_pdf(
                pdf_path
            )

            print(
                f"Chunks generated: "
                f"{len(chunks)}"
            )

            index_document(
                vector_store=vector_store,
                chunks=chunks,
            )

            manifest_documents[
                document_id
            ] = {
                "filename": pdf_path.name,
                "hash": current_hash,
                "chunk_count": len(chunks),
                "indexed_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

            save_manifest(
                manifest
            )

            print(
                f"Indexed: "
                f"{pdf_path.name}"
            )

    finally:

        vector_store.client.close()

    print(
        "\nIncremental indexing complete."
    )


if __name__ == "__main__":
    main()