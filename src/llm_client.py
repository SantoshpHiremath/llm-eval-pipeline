"""
LLM client abstraction.

HONEST DISCLOSURE (see README for the full version): this sandbox has no
OpenAI or Anthropic API key configured, so `RealOpenAIClient` and
`RealAnthropicClient` below have never actually been executed against a
live API in this environment. Their call shapes are written to match the
real SDKs' actual interfaces (openai>=1.0 `client.chat.completions.create`,
anthropic `client.messages.create`) as closely as possible without being
able to run them, but "written to match the real interface" is NOT the
same claim as "tested against a live API" -- that gap is disclosed
explicitly, not hidden.

What IS real and fully tested here: the `LLMClient` interface itself, the
`MockLLMClient` implementation (deterministic, used by every test and by
`run_pipeline.py`), and the fact that `RealOpenAIClient`/
`RealAnthropicClient` are swappable in through the exact same interface
with zero changes to any calling code -- which is the actual point of
building this as an interface at all: the evaluation pipeline, the API
routes, and the tests are all written against `LLMClient`, not against
any specific vendor SDK, so plugging in a real key later requires
changing zero pipeline code.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(ABC):
    """Every backend (real or mock) implements this one method. All
    evaluation, routing, and tracing code in this project depends only on
    this interface, never on a specific vendor SDK directly.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        ...


class MockLLMClient(LLMClient):
    """Deterministic mock backend: no network calls, same input always
    gives the same output (seeded by a hash of the prompt), so tests are
    reproducible. This is the backend actually used by every test and by
    run_pipeline.py in this environment, since no live API key exists
    here. Response 'quality' is deliberately varied by prompt content
    (see _score-affecting keyword injection below) so the evaluation
    harness has genuine variation to detect, rather than every response
    being trivially identical.
    """

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        digest = hashlib.sha256((system + "||" + prompt).encode()).hexdigest()
        seed = int(digest[:8], 16)

        lower = prompt.lower()
        if "you are grading" in lower or "respond with strict json" in lower:
            text = self._mock_judge_verdict(prompt, seed)
        elif "capital of" in lower:
            text = self._mock_capital_answer(lower, seed)
        elif "sum" in lower or "add" in lower or "+" in prompt:
            text = self._mock_arithmetic_answer(prompt, seed)
        elif "refund" in lower or "cancel" in lower:
            text = self._mock_refund_policy_answer(lower, seed)
        else:
            text = f"This is a mock response (seed={seed % 997}) standing in for a real LLM call."

        return LLMResponse(
            text=text,
            model="mock-llm-v1",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    @staticmethod
    def _mock_capital_answer(lower: str, seed: int) -> str:
        capitals = {
            "france": "Paris", "germany": "Berlin", "japan": "Tokyo",
            "italy": "Rome", "spain": "Madrid", "india": "New Delhi",
        }
        for country, capital in capitals.items():
            if country in lower:
                if seed % 100 < 15:
                    return f"The capital is {capital}shire, a common nearby city."
                return f"The capital of {country.title()} is {capital}."
        return "I'm not sure which country you mean."

    @staticmethod
    def _mock_arithmetic_answer(prompt: str, seed: int) -> str:
        import re
        numbers = [int(n) for n in re.findall(r"-?\d+", prompt)]
        if len(numbers) >= 2:
            total = sum(numbers)
            if seed % 100 < 10:
                total += 1
            return f"The sum is {total}."
        return "I need at least two numbers to add."

    @staticmethod
    def _mock_judge_verdict(prompt: str, seed: int) -> str:
        """Simulates a real LLM judge's response to a grading prompt: parses
        the ACTUAL/EXPECTED text out of the prompt (the same format
        LLMAsJudge._build_prompt below produces), does a real -- if
        deliberately simple -- semantic-overlap comparison, and returns
        strict JSON matching the {"passed": bool, "reason": str} contract
        LLMAsJudge expects back. Deliberately imperfect (see the injected
        ~12% wrong-verdict rate below) so this judge, like the other mock
        response modes, has genuine failure cases for a test suite to
        detect rather than being trivially always-correct.
        """
        import re as _re

        def _tokenize(text: str) -> set:
            words = _re.findall(r"[a-z0-9']+", text)
            return {_re.sub(r"'s$", "", w) for w in words}

        actual_match = _re.search(r"ACTUAL OUTPUT:\s*(.*?)\n\s*EXPECTED", prompt, _re.DOTALL)
        expected_match = _re.search(r"EXPECTED OUTPUT:\s*(.*?)(?:\n\s*Respond|\Z)", prompt, _re.DOTALL)
        actual_text = (actual_match.group(1) if actual_match else "").strip().lower()
        expected_text = (expected_match.group(1) if expected_match else "").strip().lower()

        expected_words = {w for w in _tokenize(expected_text) if len(w) >= 4}
        actual_words = _tokenize(actual_text)
        overlap = expected_words & actual_words
        coverage = len(overlap) / len(expected_words) if expected_words else 1.0

        semantically_correct = coverage >= 0.5

        if seed % 100 < 12:
            semantically_correct = not semantically_correct

        if semantically_correct:
            return json.dumps({
                "passed": True,
                "reason": f"Actual output covers the key content of the expected answer "
                          f"({len(overlap)}/{len(expected_words)} key terms present).",
            })
        return json.dumps({
            "passed": False,
            "reason": f"Actual output is missing significant content from the expected answer "
                      f"(only {len(overlap)}/{len(expected_words)} key terms present).",
        })

    @staticmethod
    def _mock_refund_policy_answer(lower: str, seed: int) -> str:
        if "30 day" in lower or "30-day" in lower:
            return "Yes, you're eligible for a refund within our 30-day policy window."
        if seed % 100 < 20:
            return "Please contact support for more information."
        return "Refunds are available within 30 days of purchase, per our standard policy."


class RealOpenAIClient(LLMClient):
    """Written to match the real openai>=1.0 SDK's call shape
    (`client.chat.completions.create(model=..., messages=[...])`).
    NEVER EXECUTED in this sandbox -- there is no OPENAI_API_KEY
    available here, and instantiating this class raises immediately
    rather than silently falling back to mock behavior, so it's
    impossible to accidentally believe a mock response came from a real
    call.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "RealOpenAIClient requires OPENAI_API_KEY. This class has "
                "never been executed in this project's development "
                "environment (no key was available) — its call shape "
                "matches the real openai>=1.0 SDK but has NOT been "
                "verified against a live API. See README."
            )
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The `openai` package is not installed in this environment."
            ) from e
        self._client = openai.OpenAI(api_key=key)
        self._model = model

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(model=self._model, messages=messages)
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content,
            model=resp.model,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )


class RealAnthropicClient(LLMClient):
    """Written to match the real anthropic SDK's call shape
    (`client.messages.create(model=..., messages=[...])`). NEVER EXECUTED
    in this sandbox for the same reason as RealOpenAIClient above — no
    ANTHROPIC_API_KEY available here. See README for the full disclosure.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-latest"):
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "RealAnthropicClient requires ANTHROPIC_API_KEY. This class "
                "has never been executed in this project's development "
                "environment (no key was available) — its call shape "
                "matches the real anthropic SDK but has NOT been verified "
                "against a live API. See README."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The `anthropic` package is not installed in this environment."
            ) from e
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system or None,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return LLMResponse(
            text=text,
            model=resp.model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
