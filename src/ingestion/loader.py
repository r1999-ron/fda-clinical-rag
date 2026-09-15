from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_pdf(
    pdf_file: Path,
):
    print(
        f"Loading: "
        f"{pdf_file.name}"
    )

    loader = PyPDFLoader(
        str(pdf_file)
    )

    pages = loader.load()

    print(
        f"Pages loaded: "
        f"{len(pages)}"
    )

    return pages

def load_pdfs(directory: Path):
    documents = []

    pdf_files = list(directory.glob("*.pdf"))

    print(f"Looking in: {directory}")
    print(f"PDF files found: {len(pdf_files)}")

    for pdf_file in pdf_files:
        print(f"\nLoading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        print(f"Pages loaded: {len(pages)}")

        documents.extend(pages)

    return documents


if __name__ == "__main__":
    docs = load_pdfs(DATA_DIR)

    print(f"\nTotal pages loaded: {len(docs)}")

    if docs:
        print("\n--- FIRST DOCUMENT ---")
        print(docs[0])

        print("\n--- METADATA ---")
        print(docs[0].metadata)

        print("\n--- CONTENT ---")
        print(docs[0].page_content[:1000])