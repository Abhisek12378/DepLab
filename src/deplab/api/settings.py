from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",)
    conversation_ttl: timedelta = timedelta(hours=2)
    maximum_request_bytes: int = 120_000

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        origins = tuple(
            value.strip()
            for value in os.getenv(
                "DEPLAB_ALLOWED_ORIGINS", "http://localhost:5173"
            ).split(",")
            if value.strip()
        )
        return cls(
            host=os.getenv("DEPLAB_API_HOST", "127.0.0.1"),
            port=int(os.getenv("DEPLAB_API_PORT", "8000")),
            allowed_origins=origins,
            conversation_ttl=timedelta(
                minutes=int(os.getenv("DEPLAB_CONVERSATION_TTL_MINUTES", "120"))
            ),
            maximum_request_bytes=int(
                os.getenv("DEPLAB_MAXIMUM_REQUEST_BYTES", "120000")
            ),
        )
