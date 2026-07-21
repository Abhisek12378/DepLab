from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationNotFoundError(KeyError):
    pass


class ConversationLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredMessage:
    id: str
    role: str
    content: str
    created_at: datetime
    client_message_id: str | None = None
    result: dict[str, Any] | None = None


@dataclass
class Conversation:
    id: str
    requirements_text: str
    python_version: str
    platform: str
    created_at: datetime
    expires_at: datetime
    messages: list[StoredMessage] = field(default_factory=list)


class InMemoryConversationStore:
    """Thread-safe, TTL-bound store; replace with Redis without changing the API."""

    def __init__(
        self,
        ttl: timedelta = timedelta(hours=2),
        maximum_conversations: int = 500,
        maximum_messages: int = 40,
    ) -> None:
        self.ttl = ttl
        self.maximum_conversations = maximum_conversations
        self.maximum_messages = maximum_messages
        self._items: dict[str, Conversation] = {}
        self._lock = threading.RLock()

    def create(self, requirements_text: str, python_version: str, platform: str) -> Conversation:
        with self._lock:
            self._remove_expired()
            self._evict_oldest_if_needed()
            now = utc_now()
            conversation = Conversation(
                id=str(uuid.uuid4()),
                requirements_text=requirements_text,
                python_version=python_version,
                platform=platform,
                created_at=now,
                expires_at=now + self.ttl,
            )
            self._items[conversation.id] = conversation
            return copy.deepcopy(conversation)

    def get(self, conversation_id: str) -> Conversation:
        with self._lock:
            self._remove_expired()
            conversation = self._items.get(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            return copy.deepcopy(conversation)

    def append_exchange(
        self,
        conversation_id: str,
        user_message: StoredMessage,
        assistant_message: StoredMessage,
    ) -> Conversation:
        with self._lock:
            conversation = self._required(conversation_id)
            if len(conversation.messages) + 2 > self.maximum_messages:
                raise ConversationLimitError("This conversation reached its message limit.")
            conversation.messages.extend((user_message, assistant_message))
            conversation.expires_at = utc_now() + self.ttl
            return copy.deepcopy(conversation)

    def ensure_exchange_capacity(self, conversation_id: str) -> None:
        with self._lock:
            conversation = self._required(conversation_id)
            if len(conversation.messages) + 2 > self.maximum_messages:
                raise ConversationLimitError("This conversation reached its message limit.")

    def find_exchange(self, conversation_id: str, client_message_id: str) -> tuple[StoredMessage, StoredMessage] | None:
        conversation = self.get(conversation_id)
        for index, message in enumerate(conversation.messages):
            if message.role == "user" and message.client_message_id == client_message_id:
                if index + 1 < len(conversation.messages):
                    return message, conversation.messages[index + 1]
        return None

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            if self._items.pop(conversation_id, None) is None:
                raise ConversationNotFoundError(conversation_id)

    def _required(self, conversation_id: str) -> Conversation:
        self._remove_expired()
        conversation = self._items.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    def _remove_expired(self) -> None:
        now = utc_now()
        expired = [key for key, value in self._items.items() if value.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)

    def _evict_oldest_if_needed(self) -> None:
        while len(self._items) >= self.maximum_conversations:
            oldest = min(self._items.values(), key=lambda item: item.created_at)
            self._items.pop(oldest.id, None)
