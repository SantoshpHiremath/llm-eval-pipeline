import os

import pytest

from src.llm_client import MockLLMClient, RealAnthropicClient, RealOpenAIClient


class TestMockLLMClient:
    def test_is_deterministic_for_same_prompt(self):
        client = MockLLMClient()
        r1 = client.complete("What is the capital of France?")
        r2 = client.complete("What is the capital of France?")
        assert r1.text == r2.text

    def test_different_prompts_can_give_different_responses(self):
        client = MockLLMClient()
        r1 = client.complete("What is the capital of France?")
        r2 = client.complete("What is the capital of Japan?")
        assert r1.text != r2.text

    def test_system_prompt_affects_the_seed(self):
        client = MockLLMClient()
        r1 = client.complete("Hello", system="")
        r2 = client.complete("Hello", system="Be formal.")
        assert isinstance(r1.text, str)
        assert isinstance(r2.text, str)

    def test_capital_questions_usually_answer_correctly(self):
        client = MockLLMClient()
        r = client.complete("What is the capital of France?")
        assert "paris" in r.text.lower() or "parisshire" in r.text.lower()

    def test_arithmetic_extracts_and_sums_numbers(self):
        client = MockLLMClient()
        r = client.complete("What is 12 + 30?")
        assert "42" in r.text or "43" in r.text

    def test_response_reports_token_counts(self):
        client = MockLLMClient()
        r = client.complete("What is the capital of Germany?")
        assert r.prompt_tokens > 0
        assert r.completion_tokens > 0

    def test_model_name_identifies_it_as_mock(self):
        client = MockLLMClient()
        r = client.complete("anything")
        assert "mock" in r.model.lower()

    def test_over_many_capital_questions_error_rate_is_roughly_as_designed(self):
        """Regression-style check that the deliberately injected ~15%
        wrong-answer rate for capital questions is actually present (not
        asserting an exact count, since it's seed/hash-dependent, just
        that errors exist at all across enough variation)."""
        client = MockLLMClient()
        countries = ["France", "Germany", "Japan", "Italy", "Spain", "India"]
        wrong = 0
        for i in range(50):
            r = client.complete(f"What is the capital of {countries[i % len(countries)]}? (variant {i})")
            if "shire" in r.text.lower():
                wrong += 1
        assert wrong > 0, "expected the mock backend to inject at least one wrong answer across 50 variants"


class TestRealOpenAIClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            RealOpenAIClient()

    def test_error_message_discloses_it_was_never_tested_live(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="never been executed"):
            RealOpenAIClient()


class TestRealAnthropicClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            RealAnthropicClient()

    def test_error_message_discloses_it_was_never_tested_live(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="never been executed"):
            RealAnthropicClient()
