import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def get_embedding_model():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to the project's .env file."
        )

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )


if __name__ == "__main__":
    embeddings = get_embedding_model()

    vector = embeddings.embed_query(
        "What are the requirements for informed consent?"
    )

    print(f"Embedding dimension: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")