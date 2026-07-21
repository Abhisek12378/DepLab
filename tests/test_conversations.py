from __future__ import annotations

import unittest

from deplab.advisor.contracts import AdvisoryResult, AnalysisRequest
from deplab.api.application import ConversationApplication
from deplab.api.conversations import (
    ConversationLimitError,
    InMemoryConversationStore,
)


class RecordingAdvisor:
    def __init__(self) -> None:
        self.requests: list[AnalysisRequest] = []

    def analyze(self, request: AnalysisRequest) -> AdvisoryResult:
        self.requests.append(request)
        return AdvisoryResult(
            status="no_risk_predicted",
            summary="No covered risk was predicted.",
            answer=f"Answer for: {request.question}",
        )


class ConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.advisor = RecordingAdvisor()
        self.store = InMemoryConversationStore(maximum_messages=4)
        self.application = ConversationApplication(self.advisor, self.store)  # type: ignore[arg-type]
        self.conversation = self.application.create_conversation(
            "numpy==1.26.4\npandas==2.1.4", "3.11", "linux-x86_64"
        )

    def test_follow_up_receives_bounded_prior_conversation(self) -> None:
        self.application.ask(self.conversation.id, "Can I use NumPy 2.0.2?", "client-001")
        self.application.ask(self.conversation.id, "Why does pandas block that?", "client-002")
        follow_up = self.advisor.requests[-1]
        self.assertEqual(follow_up.question, "Why does pandas block that?")
        self.assertEqual([turn.role for turn in follow_up.conversation_context], ["user", "assistant"])
        self.assertIn("NumPy 2.0.2", follow_up.conversation_context[0].content)

    def test_client_message_id_makes_retry_idempotent(self) -> None:
        first = self.application.ask(self.conversation.id, "Check NumPy 2.0.2", "client-retry")
        second = self.application.ask(self.conversation.id, "Check NumPy 2.0.2", "client-retry")
        self.assertEqual(first[1].id, second[1].id)
        self.assertEqual(first[2].id, second[2].id)
        self.assertEqual(len(self.advisor.requests), 1)

    def test_message_limit_is_enforced(self) -> None:
        self.application.ask(self.conversation.id, "First", "client-first")
        self.application.ask(self.conversation.id, "Second", "client-second")
        with self.assertRaises(ConversationLimitError):
            self.application.ask(self.conversation.id, "Third", "client-third")
        self.assertEqual(len(self.advisor.requests), 2)

    def test_store_returns_a_copy(self) -> None:
        fetched = self.store.get(self.conversation.id)
        fetched.requirements_text = "changed==1"
        self.assertEqual(
            self.store.get(self.conversation.id).requirements_text,
            "numpy==1.26.4\npandas==2.1.4",
        )


if __name__ == "__main__":
    unittest.main()
