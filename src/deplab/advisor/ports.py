from __future__ import annotations

from typing import Protocol

from .contracts import AnalysisRequest, ParsedIntent


class IntentParser(Protocol):
    def parse(self, request: AnalysisRequest) -> ParsedIntent: ...


class AnswerComposer(Protocol):
    def compose(self, result_payload: dict[str, object]) -> str: ...
