from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import APIRouter, Response, status
from fastapi.concurrency import run_in_threadpool

from .application import ConversationApplication
from .schemas import (
    ConversationResponse,
    CreateConversationRequest,
    ExchangeResponse,
    HealthResponse,
    MessageResponse,
    SendMessageRequest,
)


class ConversationLocks:
    def __init__(self) -> None:
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get(self, conversation_id: str) -> asyncio.Lock:
        return self._locks[conversation_id]

    def discard(self, conversation_id: str) -> None:
        self._locks.pop(conversation_id, None)


def create_router(application: ConversationApplication) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    locks = ConversationLocks()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @router.post(
        "/conversations",
        response_model=ConversationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_conversation(
        payload: CreateConversationRequest,
    ) -> ConversationResponse:
        conversation = await run_in_threadpool(
            application.create_conversation,
            payload.requirements_text,
            payload.python_version,
            payload.platform,
        )
        return ConversationResponse.from_domain(conversation)

    @router.get(
        "/conversations/{conversation_id}",
        response_model=ConversationResponse,
    )
    async def get_conversation(conversation_id: str) -> ConversationResponse:
        conversation = await run_in_threadpool(application.store.get, conversation_id)
        return ConversationResponse.from_domain(conversation)

    @router.post(
        "/conversations/{conversation_id}/messages",
        response_model=ExchangeResponse,
    )
    async def send_message(
        conversation_id: str,
        payload: SendMessageRequest,
    ) -> ExchangeResponse:
        async with locks.get(conversation_id):
            conversation, user, assistant = await run_in_threadpool(
                application.ask,
                conversation_id,
                payload.content,
                payload.client_message_id,
            )
        return ExchangeResponse(
            conversation=ConversationResponse.from_domain(conversation),
            user_message=MessageResponse.from_domain(user),
            assistant_message=MessageResponse.from_domain(assistant),
        )

    @router.delete(
        "/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_conversation(conversation_id: str) -> Response:
        await run_in_threadpool(application.store.delete, conversation_id)
        locks.discard(conversation_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
