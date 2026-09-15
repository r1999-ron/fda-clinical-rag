from collections import defaultdict
from threading import Lock
from typing import Dict, List

from src.conversation.models import (
    ConversationMessage,
)


class InMemorySessionStore:
    """
    Simple development-only session store.

    Production:
        replace with Redis / external state store.
    """

    def __init__(self):
        self._sessions: Dict[
            str,
            List[ConversationMessage]
        ] = defaultdict(list)

        self._lock = Lock()

    def get_history(
        self,
        session_id: str,
    ) -> List[ConversationMessage]:

        with self._lock:

            # Return a copy so callers cannot mutate
            # internal state accidentally.
            return list(
                self._sessions.get(
                    session_id,
                    [],
                )
            )

    def add_message(
        self,
        session_id: str,
        message: ConversationMessage,
    ):
        with self._lock:
            self._sessions[
                session_id
            ].append(
                message
            )

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ):
        with self._lock:

            self._sessions[
                session_id
            ].append(
                ConversationMessage(
                    role="user",
                    content=user_message,
                )
            )

            self._sessions[
                session_id
            ].append(
                ConversationMessage(
                    role="assistant",
                    content=assistant_message,
                )
            )

    def clear(
        self,
        session_id: str,
    ):
        with self._lock:
            self._sessions.pop(
                session_id,
                None,
            )


session_store = (
    InMemorySessionStore()
)