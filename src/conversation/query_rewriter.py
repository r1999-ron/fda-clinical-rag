from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.conversation.models import (
    ConversationMessage,
)


load_dotenv()


class QueryRewriteResult(BaseModel):
    standalone_query: str
    rewritten: bool


def format_history(
    history: List[ConversationMessage],
    max_messages: int = 6,
) -> str:
    """
    Keep only recent conversation history.

    We do not want to send an unlimited conversation
    to the query rewriting model.
    """

    if not history:
        return "No previous conversation."

    recent_history = history[
        -max_messages:
    ]

    lines = []

    for message in recent_history:

        role = (
            "User"
            if message.role == "user"
            else "Assistant"
        )

        lines.append(
            f"{role}: "
            f"{message.content}"
        )

    return "\n".join(lines)


def rewrite_query(
    question: str,
    history: List[
        ConversationMessage
    ] | None = None,
) -> QueryRewriteResult:
    """
    Convert a conversational follow-up into a
    standalone retrieval query.

    Example:

        History:
        User: What risks must be explained during informed consent?

        Current:
        What about alternatives?

    Output:
        What alternative procedures or treatments should be
        explained during informed consent?
    """

    history = history or []

    # No history means there is nothing to resolve.
    if not history:
        return QueryRewriteResult(
            standalone_query=question,
            rewritten=False,
        )

    formatted_history = (
        format_history(
            history
        )
    )

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Add it to the project's .env file."
        )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=api_key,
    )

    structured_llm = (
        llm.with_structured_output(
            QueryRewriteResult
        )
    )

    prompt = f"""
You rewrite conversational questions into standalone
retrieval queries for an enterprise RAG system.

Your task is NOT to answer the question.

Use the conversation history only to resolve references,
missing subjects, pronouns, and follow-up context.

Rules:

1. Return a standalone search query.

2. Preserve the user's original intent.

3. Do NOT add facts that were not present in the conversation.

4. Do NOT answer the question.

5. If the current question already makes sense independently,
   return it unchanged.

6. Resolve references such as:
   - "what about alternatives?"
   - "what about the risks?"
   - "does it mention costs?"
   - "what happens after that?"
   using the conversation history.

7. Keep the rewritten query concise.

CONVERSATION HISTORY:

{formatted_history}

CURRENT QUESTION:

{question}
""".strip()

    result = structured_llm.invoke(
        prompt
    )

    return result

if __name__ == "__main__":

    history = [
        ConversationMessage(
            role="user",
            content=(
                "What risks must be explained "
                "during informed consent?"
            ),
        ),
        ConversationMessage(
            role="assistant",
            content=(
                "Reasonably foreseeable risks "
                "and discomforts should be explained."
            ),
        ),
    ]

    question = (
        "What about alternatives?"
    )

    result = rewrite_query(
        question=question,
        history=history,
    )

    print(
        "Original:"
    )
    print(
        question
    )

    print(
        "\nStandalone query:"
    )
    print(
        result.standalone_query
    )

    print(
        "\nRewritten:"
    )
    print(
        result.rewritten
    )