import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_pdfs, DATA_DIR


def add_chunk_metadata(chunks):
    for index, chunk in enumerate(chunks):
        page = chunk.metadata.get("page")

        chunk.metadata["chunk_id"] = f"chunk_{index}"

        if page is not None:
            chunk.metadata["page_number"] = page + 1

    return chunks


# ============================================================
# CHARACTER-BASED CHUNKING
# ============================================================

def chunk_documents(
    documents,
    chunk_size=800,
    chunk_overlap=100,
):
    print(
        f"Character chunking: "
        f"size={chunk_size}, overlap={chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    chunks = add_chunk_metadata(chunks)

    print(f"Generated chunks: {len(chunks)}")

    return chunks


# ============================================================
# TOKEN-AWARE CHUNKING
# ============================================================

def chunk_documents_token_aware(
    documents,
    chunk_size=300,
    chunk_overlap=50,
):
    print(
        f"Token-aware chunking: "
        f"size={chunk_size}, overlap={chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    chunks = add_chunk_metadata(chunks)

    print(f"Generated chunks: {len(chunks)}")

    return chunks


# ============================================================
# STRUCTURE-AWARE HELPERS
# ============================================================

def is_heading(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    if len(line) > 160:
        return False

    word_count = len(line.split())

    # Avoid treating long prose as headings.
    if word_count > 18:
        return False

    # TOC dotted leaders.
    if re.search(r"\.{4,}", line):
        return False

    # Reject things like:
    # 1740. Identify all comments...
    if re.match(r"^\d{3,}\.", line):
        return False

    # 3. Reasonably Foreseeable Risks and Discomforts20 214
    numbered_heading = re.match(
        r"^[1-9]\d?\.\s+[A-Z].*",
        line,
    )

    # 8.1.2 Tasks and Use Scenarios
    decimal_heading = re.match(
        r"^[1-9]\d?(?:\.\d+)+\s+[A-Z].*",
        line,
    )

    # B. Identifying Key Information
    letter_heading = re.match(
        r"^[A-Z]\.\s+[A-Z].*",
        line,
    )

    # IV. Risk Management
    roman_heading = re.match(
        r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z].*",
        line,
    )

    uppercase_heading = (
        line.isupper()
        and 2 <= word_count <= 10
        and "CONTAINS NONBINDING" not in line
        and "DRAFT" not in line
    )

    return bool(
        numbered_heading
        or decimal_heading
        or letter_heading
        or roman_heading
        or uppercase_heading
    )


def is_table_of_contents_page(document: Document) -> bool:
    """
    Prevent Table-of-Contents entries from becoming real section metadata.
    """

    text = document.page_content.lower()

    return (
        "table of contents" in text
        or "contents" == text.strip()
    )


def clean_heading(line: str) -> str:
    """
    Normalize headings extracted from PDFs.

    Example:
    '3. Reasonably Foreseeable Risks and Discomforts20 214'

    becomes:
    '3. Reasonably Foreseeable Risks and Discomforts'
    """

    heading = line.strip()

    # Remove trailing PDF line-number artifacts such as:
    # "Discomforts20 214"
    heading = re.sub(
        r"(?<=[A-Za-z\)])\d+\s+\d+\s*$",
        "",
        heading,
    )

    # Remove an isolated trailing page/line number.
    heading = re.sub(
        r"\s+\d{2,4}\s*$",
        "",
        heading,
    )

    return heading.strip()


def split_documents_into_sections(
    documents: List[Document],
) -> List[Document]:
    """
    Convert page-level LangChain Documents into section-aware Documents.

    Important behavior:

    If a section starts on page 10 and continues onto page 11,
    page 11 inherits the last known section heading.

    Section state is tracked independently for each PDF.
    """

    section_documents = []

    last_section_by_source = {}

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown",
        )

        current_heading = last_section_by_source.get(
            source
        )

        current_lines = []

        toc_page = is_table_of_contents_page(
            document
        )

        def flush_section():
            if not current_lines:
                return

            text = "\n".join(
                current_lines
            ).strip()

            if not text:
                return

            metadata = document.metadata.copy()

            # Important:
            # Do NOT invent section metadata.
            #
            # If no trustworthy heading exists, store None.
            metadata["section"] = current_heading

            section_documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        for line in document.page_content.splitlines():

            stripped = line.strip()

            # Never learn section headings from TOC pages.
            heading_detected = (
                not toc_page
                and is_heading(stripped)
            )

            if heading_detected:

                # Save text belonging to previous section first.
                flush_section()

                current_lines = []

                current_heading = clean_heading(stripped)

                last_section_by_source[source] = current_heading

                current_lines.append(current_heading)

            else:

                current_lines.append(line)

        flush_section()

    return section_documents


# ============================================================
# STRUCTURE-AWARE CHUNKING
# ============================================================

def chunk_documents_structure_aware(
    documents,
    max_chunk_size=500,
    chunk_overlap=50,
):
    """
    Strategy:

    Page
      ↓
    Identify structural sections
      ↓
    Preserve section metadata
      ↓
    If section <= max_chunk_size:
        keep it
      ↓
    Otherwise:
        recursively split inside that section
    """

    print(
        f"Structure-aware chunking: "
        f"max_size={max_chunk_size}, "
        f"overlap={chunk_overlap}"
    )

    section_documents = split_documents_into_sections(
        documents
    )

    detected_sections = sum(
        1
        for section in section_documents
        if section.metadata.get("section")
    )

    print(
        f"Section documents created: "
        f"{len(section_documents)}"
    )

    print(
        f"Documents with detected section: "
        f"{detected_sections}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for section in section_documents:

        if len(section.page_content) <= max_chunk_size:

            chunks.append(section)

        else:

            smaller_chunks = splitter.split_documents(
                [section]
            )

            chunks.extend(smaller_chunks)

    chunks = add_chunk_metadata(chunks)

    print(f"Generated chunks: {len(chunks)}")

    return chunks


# ============================================================
# MANUAL INSPECTION
# ============================================================

if __name__ == "__main__":

    documents = load_pdfs(DATA_DIR)

    documents = clean_documents(
        documents
    )

    for document in documents:
        source = document.metadata.get("source", "")

        if source.endswith("05_informed_consent_guidance.pdf"):

            page_number = document.metadata.get("page", 0) + 1

            if 9 <= page_number <= 13:

                print("\n" + "=" * 100)
                print(f"PAGE {page_number}")
                print("=" * 100)

                for line in document.page_content.splitlines():

                    stripped = line.strip()

                    if not stripped:
                        continue

                    print(
                        f"{'✅ HEADING' if is_heading(stripped) else '❌'}"
                        f" | {repr(stripped)}"
                    )

    chunks = chunk_documents_structure_aware(
        documents,
        max_chunk_size=500,
        chunk_overlap=50,
    )

    print(f"\nTotal chunks: {len(chunks)}")

    # Print more than just the first few so we can evaluate
    # whether section extraction is actually trustworthy.
    for chunk in chunks[:30]:

        print("\n" + "=" * 100)

        print("Source:")
        print(
            chunk.metadata.get("source")
        )

        print("Page:")
        print(
            chunk.metadata.get("page_number")
        )

        print("Section:")
        print(
            chunk.metadata.get("section")
        )

        print("Chunk ID:")
        print(
            chunk.metadata.get("chunk_id")
        )

        print("\nContent:")

        print(
            chunk.page_content[:800]
        )