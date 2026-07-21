from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .conversations import Conversation, StoredMessage


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateConversationRequest(StrictModel):
    requirements_text: str = Field(min_length=1, max_length=100_000)
    python_version: Literal["3.10", "3.11", "3.12"]
    platform: Literal["linux-x86_64"] = "linux-x86_64"


class SendMessageRequest(StrictModel):
    content: str = Field(min_length=1, max_length=2_000)
    client_message_id: str = Field(min_length=8, max_length=100)


class MessageResponse(StrictModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    result: dict[str, Any] | None = None

    @classmethod
    def from_domain(cls, message: StoredMessage) -> "MessageResponse":
        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            result=message.result,
        )


class ConversationResponse(StrictModel):
    id: str
    requirements_text: str
    python_version: str
    platform: str
    created_at: datetime
    expires_at: datetime
    messages: list[MessageResponse]

    @classmethod
    def from_domain(cls, conversation: Conversation) -> "ConversationResponse":
        return cls(
            id=conversation.id,
            requirements_text=conversation.requirements_text,
            python_version=conversation.python_version,
            platform=conversation.platform,
            created_at=conversation.created_at,
            expires_at=conversation.expires_at,
            messages=[MessageResponse.from_domain(item) for item in conversation.messages],
        )


class ExchangeResponse(StrictModel):
    conversation: ConversationResponse
    user_message: MessageResponse
    assistant_message: MessageResponse


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: Literal["deplab-api"] = "deplab-api"


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorBody
