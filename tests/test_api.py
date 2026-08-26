import os

import pytest
from fastapi.testclient import TestClient

os.environ["EVAL_PIPELINE_API_KEY"] = "test-key-123"

from src.api import app  # noqa: E402  (must import after env var is set)

client = TestClient(app)
VALID_HEADERS = {"x-api-key": "test-key-123"}


class TestHealth:
    def test_health_requires_no_auth(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAuthentication:
    def test_ask_without_api_key_is_rejected(self):
        resp = client.post("/ask", json={"question": "What is the capital of France?"})
        assert resp.status_code == 401

    def test_ask_with_wrong_api_key_is_rejected(self):
        resp = client.post(
            "/ask",
            json={"question": "What is the capital of France?"},
            headers={"x-api-key": "totally-wrong-key"},
        )
        assert resp.status_code == 401

    def test_ask_with_correct_api_key_is_accepted(self):
        resp = client.post(
            "/ask",
            json={"question": "What is the capital of France?"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 200

    def test_evaluate_without_api_key_is_rejected(self):
        resp = client.post("/evaluate")
        assert resp.status_code == 401

    def test_evaluate_with_correct_api_key_is_accepted(self):
        resp = client.post("/evaluate", headers=VALID_HEADERS)
        assert resp.status_code == 200


class TestAskEndpoint:
    def test_returns_an_answer_and_model_name(self):
        resp = client.post(
            "/ask",
            json={"question": "What is the capital of Germany?"},
            headers=VALID_HEADERS,
        )
        body = resp.json()
        assert "berlin" in body["answer"].lower()
        assert body["model"] == "mock-llm-v1"

    def test_rejects_empty_question(self):
        resp = client.post("/ask", json={"question": ""}, headers=VALID_HEADERS)
        assert resp.status_code == 422

    def test_rejects_missing_question_field(self):
        resp = client.post("/ask", json={}, headers=VALID_HEADERS)
        assert resp.status_code == 422

    def test_rejects_overly_long_question(self):
        resp = client.post("/ask", json={"question": "x" * 3000}, headers=VALID_HEADERS)
        assert resp.status_code == 422

    def test_reports_token_counts(self):
        resp = client.post(
            "/ask",
            json={"question": "What is 5 + 6?"},
            headers=VALID_HEADERS,
        )
        body = resp.json()
        assert body["prompt_tokens"] > 0
        assert body["completion_tokens"] > 0


class TestEvaluateEndpoint:
    def test_returns_report_shape(self):
        resp = client.post("/evaluate", headers=VALID_HEADERS)
        body = resp.json()
        for key in [
            "total_items", "golden_pass_rate", "silver_pass_rate",
            "overall_pass_rate", "mean_latency_seconds",
            "failing_golden_ids", "failing_silver_ids",
        ]:
            assert key in body

    def test_total_items_matches_sample_dataset_size(self):
        from src.sample_dataset import build_sample_dataset
        resp = client.post("/evaluate", headers=VALID_HEADERS)
        assert resp.json()["total_items"] == len(build_sample_dataset())

    def test_pass_rates_are_valid_fractions(self):
        resp = client.post("/evaluate", headers=VALID_HEADERS)
        body = resp.json()
        for key in ["golden_pass_rate", "silver_pass_rate", "overall_pass_rate"]:
            assert 0.0 <= body[key] <= 1.0
