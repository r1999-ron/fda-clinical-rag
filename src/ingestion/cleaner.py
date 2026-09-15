import re


def clean_text(text: str) -> str:
    # Remove isolated line numbers
    text = re.sub(r"\b\d{2,4}\s*\n", "", text)

    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents):
    for document in documents:
        document.page_content = clean_text(document.page_content)

    return documents