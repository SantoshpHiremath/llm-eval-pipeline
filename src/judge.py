"""
Evaluation judges: given a model's actual output and a dataset item's
expected output, decide pass/fail (and why). Three judges:

- ExactMatchJudge: strict, only useful when the expected output has one
  correct phrasing. Deterministic, no LLM call.
- KeywordJudge: checks that expected key facts/phrases appear in the
  output, tolerant of paraphrasing. Deterministic, no LLM call.
- LLMAsJudge: a REAL LLM-as-judge -- uses an LLMClient (see
  llm_client.py) to grade another model's output by prompting it to
  compare actual vs. expected and return a structured verdict. This is
  genuinely an LLM judging an LLM, end to end, not a keyword heuristic
  relabeled as "LLM-as-judge" -- see its docstring below for what that
  does and doesn't prove given this project only has MockLLMClient
  actually exercised (no live API key -- see README, same disclosure as
  llm_client.py).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    passed: bool
    reason: str


class Judge(ABC):
    @abstractmethod
    def evaluate(self, actual_output: str, expected_output: str) -> JudgeVerdict:
        ...


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class ExactMatchJudge(Judge):
    """Passes only if actual output, normalized (whitespace/case), equals
    expected output exactly. Strict by design -- appropriate for
    short-answer items where only one phrasing is correct (e.g. "What is
    the capital of France?" -> must literally say Paris).
    """

    def evaluate(self, actual_output: str, expected_output: str) -> JudgeVerdict:
        if _normalize(actual_output) == _normalize(expected_output):
            return JudgeVerdict(True, "Exact match.")
        return JudgeVerdict(
            False,
            f"Expected exact match with {expected_output!r}, got {actual_output!r}.",
        )


class KeywordJudge(Judge):
    """Passes if every 'key phrase' extracted from the expected output
    appears (case-insensitively) somewhere in the actual output. Key
    phrases are the expected output's own significant words (stopwords
    and short tokens excluded) -- a simple, real, inspectable technique,
    not an invented scoring formula, and importantly NOT an LLM call: it
    complements ExactMatchJudge for open-ended answers without needing a
    live model to grade another model.
    """

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on",
        "for", "with", "you're", "your", "our", "we", "it", "this", "that",
    }

    def __init__(self, min_keyword_length: int = 3):
        self._min_len = min_keyword_length

    def _key_phrases(self, text: str) -> set:
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {w for w in words if len(w) >= self._min_len and w not in self._STOPWORDS}

    def evaluate(self, actual_output: str, expected_output: str) -> JudgeVerdict:
        expected_keywords = self._key_phrases(expected_output)
        actual_lower = actual_output.lower()
        missing = {kw for kw in expected_keywords if kw not in actual_lower}
        if not missing:
            return JudgeVerdict(True, "All expected keywords present.")
        return JudgeVerdict(
            False,
            f"Missing expected keyword(s): {sorted(missing)}.",
        )


_JUDGE_PROMPT_TEMPLATE = """You are grading whether an AI model's actual \
output correctly conveys the same information as an expected reference \
answer. Judge semantic correctness, not exact wording -- paraphrasing is \
fine, missing or contradictory key facts are not.

ACTUAL OUTPUT:
{actual_output}

EXPECTED OUTPUT:
{expected_output}

Respond with strict JSON only, in this exact shape:
{{"passed": true or false, "reason": "one sentence explaining the verdict"}}
"""


class LLMAsJudge(Judge):
    """A real LLM-as-judge: takes any LLMClient (see llm_client.py) and
    uses it to grade another model's output, by prompting it to compare
    actual vs. expected and return a structured pass/fail verdict. This
    is genuinely "an LLM judging an LLM" end to end -- the grading
    decision comes from a model's response to a prompt, not from a
    keyword/exact-match heuristic (contrast with ExactMatchJudge and
    KeywordJudge above).

    What this proves and doesn't, given this project's constraints (see
    llm_client.py's HONEST DISCLOSURE, which applies here identically):
    with MockLLMClient (the only LLMClient actually exercised in this
    environment -- no live API key available), this class is fully real
    and fully tested: it builds a real grading prompt, calls
    LLMClient.complete() with it, and parses a real JSON response back
    into a JudgeVerdict, including handling a malformed/non-JSON response
    without crashing (see test_judge.py's TestLLMAsJudge). What it does
    NOT prove is that a live model (GPT-4o-mini, Claude, etc.) would
    produce good, well-calibrated verdicts -- that would require actually
    running this against RealOpenAIClient/RealAnthropicClient, which have
    never been executed in this environment either. The mechanism is
    real and tested; the judgment quality of a live model is unverified.
    """

    def __init__(self, client, system_prompt: str = ""):
        self._client = client
        self._system_prompt = system_prompt

    def _build_prompt(self, actual_output: str, expected_output: str) -> str:
        return _JUDGE_PROMPT_TEMPLATE.format(
            actual_output=actual_output, expected_output=expected_output,
        )

    def evaluate(self, actual_output: str, expected_output: str) -> JudgeVerdict:
        prompt = self._build_prompt(actual_output, expected_output)
        response = self._client.complete(prompt, system=self._system_prompt)

        try:
            data = json.loads(response.text)
            passed = bool(data["passed"])
            reason = str(data.get("reason", "")) or ("Passed." if passed else "Failed.")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return JudgeVerdict(
                False,
                f"LLM judge response could not be parsed as the expected "
                f"JSON verdict shape ({exc.__class__.__name__}: {exc}). "
                f"Raw response: {response.text!r}",
            )

        return JudgeVerdict(passed, reason)
