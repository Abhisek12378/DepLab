from __future__ import annotations

import uuid

from deplab.advisor.contracts import AnalysisRequest, ConversationTurn
from deplab.advisor.service import AdvisorService

from .conversations import (
    Conversation,
    InMemoryConversationStore,
    StoredMessage,
    utc_now,
)


class ConversationApplication:
    def __init__(
        self,
        advisor: AdvisorService,
        store: InMemoryConversationStore,
    ) -> None:
        self.advisor = advisor
        self.store = store

    def create_conversation(
        self, requirements_text: str, python_version: str, platform: str
    ) -> Conversation:
        return self.store.create(requirements_text, python_version, platform)

    def ask(
        self,
        conversation_id: str,
        content: str,
        client_message_id: str,
    ) -> tuple[Conversation, StoredMessage, StoredMessage]:
        existing = self.store.find_exchange(conversation_id, client_message_id)
        if existing:
            return self.store.get(conversation_id), existing[0], existing[1]

        self.store.ensure_exchange_capacity(conversation_id)
        conversation = self.store.get(conversation_id)
        result = self.advisor.analyze(self._analysis_request(conversation, content))
        now = utc_now()
        user_message = StoredMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=content,
            created_at=now,
            client_message_id=client_message_id,
        )
        assistant_message = StoredMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=result.answer or result.summary,
            created_at=utc_now(),
            result=result.to_dict(),
        )
        updated = self.store.append_exchange(
            conversation_id, user_message, assistant_message
        )
        return updated, user_message, assistant_message

    @staticmethod
    def _analysis_request(conversation: Conversation, content: str) -> AnalysisRequest:
        context = tuple(
            ConversationTurn(role=message.role, content=message.content)
            for message in conversation.messages[-8:]
        )
        return AnalysisRequest(
            requirements_text=conversation.requirements_text,
            question=content,
            python_version=conversation.python_version,
            platform=conversation.platform,
            conversation_context=context,
        )
