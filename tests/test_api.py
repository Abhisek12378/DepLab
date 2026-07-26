from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from deplab.advisor.contracts import AdvisoryResult, AnalysisRequest
from deplab.api.conversations import InMemoryConversationStore
from deplab.api.main import create_app
from deplab.api.settings import ApiSettings


class ApiAdvisor:
    def __init__(self) -> None:
        self.requests: list[AnalysisRequest] = []

    def analyze(self, request: AnalysisRequest) -> AdvisoryResult:
        self.requests.append(request)
        return AdvisoryResult(
            status="no_risk_predicted",
            summary="No risk predicted.",
            answer=f"Grounded answer: {request.question}",
        )


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.advisor = ApiAdvisor()
        app = create_app(
            settings=ApiSettings(),
            advisor=self.advisor,  # type: ignore[arg-type]
            store=InMemoryConversationStore(maximum_messages=10),
        )
        self.client = TestClient(app, raise_server_exceptions=False)

    def create_conversation(self) -> str:
        response = self.client.post(
            "/api/v1/conversations",
            json={
                "requirements_text": "numpy==1.26.4\npandas==2.1.4",
                "python_version": "3.11",
                "platform": "linux-x86_64",
            },
        )
        self.assertEqual(response.status_code, 201)
        return response.json()["id"]

    def test_conversation_remembers_follow_up_context(self) -> None:
        conversation_id = self.create_conversation()
        first = self.client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "Can I use NumPy 2.0.2?", "client_message_id": "message-001"},
        )
        second = self.client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "Why does pandas block that?", "client_message_id": "message-002"},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(second.json()["conversation"]["messages"]), 4)
        self.assertEqual(len(self.advisor.requests[-1].conversation_context), 2)
        self.assertEqual(second.headers["cache-control"], "no-store")
        self.assertIn("x-request-id", second.headers)

    def test_invalid_payload_has_safe_structured_error(self) -> None:
        response = self.client.post(
            "/api/v1/conversations",
            json={"requirements_text": "", "python_version": "3.9"},
        )
        self.assertEqual(response.status_code, 422)
        payload = response.json()["error"]
        self.assertEqual(payload["code"], "invalid_request")
        self.assertNotIn("requirements_text", payload["message"])

    def test_accepts_every_supported_python_version(self) -> None:
        for python_version in (
            "3.8",
            "3.9",
            "3.10",
            "3.11",
            "3.12",
            "3.13",
            "3.14",
        ):
            with self.subTest(python_version=python_version):
                response = self.client.post(
                    "/api/v1/conversations",
                    json={
                        "requirements_text": "numpy==1.26.4",
                        "python_version": python_version,
                        "platform": "linux-x86_64",
                    },
                )
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json()["python_version"], python_version)

    def test_missing_conversation_does_not_leak_details(self) -> None:
        response = self.client.get("/api/v1/conversations/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "conversation_not_found")

    def test_large_request_is_rejected_before_parsing(self) -> None:
        response = self.client.post(
            "/api/v1/conversations",
            content=b"x" * 120_001,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "request_too_large")

    def test_retried_client_message_is_idempotent(self) -> None:
        conversation_id = self.create_conversation()
        payload = {"content": "Check NumPy 2.0.2", "client_message_id": "stable-message-id"}
        first = self.client.post(
            f"/api/v1/conversations/{conversation_id}/messages", json=payload
        )
        second = self.client.post(
            f"/api/v1/conversations/{conversation_id}/messages", json=payload
        )
        self.assertEqual(first.json()["assistant_message"]["id"], second.json()["assistant_message"]["id"])
        self.assertEqual(len(self.advisor.requests), 1)


if __name__ == "__main__":
    unittest.main()
