from src.judge import ExactMatchJudge, KeywordJudge, LLMAsJudge
from src.llm_client import LLMClient, LLMResponse, MockLLMClient


class TestExactMatchJudge:
    def test_identical_strings_pass(self):
        judge = ExactMatchJudge()
        v = judge.evaluate("Paris.", "Paris.")
        assert v.passed is True

    def test_case_and_whitespace_insensitive(self):
        judge = ExactMatchJudge()
        v = judge.evaluate("  PARIS.  ", "paris.")
        assert v.passed is True

    def test_different_strings_fail(self):
        judge = ExactMatchJudge()
        v = judge.evaluate("London.", "Paris.")
        assert v.passed is False
        assert "Paris" in v.reason


class TestKeywordJudge:
    def test_passes_when_all_keywords_present(self):
        judge = KeywordJudge()
        v = judge.evaluate(
            "The capital of France is Paris, a beautiful city.",
            "The capital of France is Paris.",
        )
        assert v.passed is True

    def test_fails_when_a_keyword_is_missing(self):
        judge = KeywordJudge()
        v = judge.evaluate(
            "I'm not sure about that country.",
            "The capital of France is Paris.",
        )
        assert v.passed is False
        assert "paris" in v.reason.lower()

    def test_tolerant_of_paraphrasing(self):
        judge = KeywordJudge()
        v = judge.evaluate(
            "Paris is the capital city of France.",
            "The capital of France is Paris.",
        )
        assert v.passed is True

    def test_stopwords_are_not_required(self):
        judge = KeywordJudge()
        v = judge.evaluate("Paris France capital.", "The capital of France is Paris.")
        assert v.passed is True

    def test_short_tokens_below_min_length_are_ignored(self):
        judge = KeywordJudge(min_keyword_length=3)
        v = judge.evaluate("no useful info", "Is it ok?")
        assert v.passed is True

    def test_missing_keywords_are_reported(self):
        judge = KeywordJudge()
        v = judge.evaluate("Berlin, and nothing else.", "The capital of Germany is Berlin.")
        assert v.passed is False
        assert "germany" in v.reason.lower()


class _FixedResponseClient(LLMClient):
    """A minimal LLMClient stand-in that always returns a pre-set response
    text, used to test LLMAsJudge's prompt-building and JSON-parsing logic
    in isolation from MockLLMClient's own judge-simulation behavior."""

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_prompt = None
        self.last_system = None

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        self.last_prompt = prompt
        self.last_system = system
        return LLMResponse(text=self._response_text, model="fixed-stub", prompt_tokens=1, completion_tokens=1)


class TestLLMAsJudgeWithFixedClient:
    """Tests LLMAsJudge's own logic (prompt construction, JSON parsing,
    error handling) against a controllable fixed-response client -- these
    do not depend on MockLLMClient's specific judge-simulation heuristics
    at all, so they'd still be valid even if that simulation changed."""

    def test_builds_a_prompt_containing_both_texts(self):
        client = _FixedResponseClient('{"passed": true, "reason": "ok"}')
        judge = LLMAsJudge(client)
        judge.evaluate("Paris is the capital.", "The capital of France is Paris.")
        assert "Paris is the capital." in client.last_prompt
        assert "The capital of France is Paris." in client.last_prompt
        assert "ACTUAL OUTPUT" in client.last_prompt
        assert "EXPECTED OUTPUT" in client.last_prompt

    def test_parses_a_passing_json_verdict(self):
        client = _FixedResponseClient('{"passed": true, "reason": "Semantically equivalent."}')
        judge = LLMAsJudge(client)
        v = judge.evaluate("Paris is the capital.", "The capital of France is Paris.")
        assert v.passed is True
        assert v.reason == "Semantically equivalent."

    def test_parses_a_failing_json_verdict(self):
        client = _FixedResponseClient('{"passed": false, "reason": "Missing the country name."}')
        judge = LLMAsJudge(client)
        v = judge.evaluate("It's Paris.", "The capital of France is Paris.")
        assert v.passed is False
        assert v.reason == "Missing the country name."

    def test_malformed_json_response_is_reported_as_a_failure_not_a_crash(self):
        client = _FixedResponseClient("Sure! I think that's correct.")
        judge = LLMAsJudge(client)
        v = judge.evaluate("actual", "expected")
        assert v.passed is False
        assert "could not be parsed" in v.reason.lower()

    def test_json_missing_passed_field_is_reported_as_a_failure_not_a_crash(self):
        client = _FixedResponseClient('{"reason": "forgot the passed field"}')
        judge = LLMAsJudge(client)
        v = judge.evaluate("actual", "expected")
        assert v.passed is False
        assert "could not be parsed" in v.reason.lower()

    def test_missing_reason_field_falls_back_to_a_default(self):
        client = _FixedResponseClient('{"passed": true}')
        judge = LLMAsJudge(client)
        v = judge.evaluate("actual", "expected")
        assert v.passed is True
        assert v.reason

    def test_system_prompt_is_passed_through_to_the_client(self):
        client = _FixedResponseClient('{"passed": true, "reason": "ok"}')
        judge = LLMAsJudge(client, system_prompt="You are a strict grader.")
        judge.evaluate("actual", "expected")
        assert client.last_system == "You are a strict grader."


class TestLLMAsJudgeWithMockLLMClient:
    """End-to-end tests against the project's actual MockLLMClient -- the
    only LLMClient implementation genuinely exercised in this environment
    (see llm_client.py's HONEST DISCLOSURE). This proves the full real
    path: LLMAsJudge builds a prompt, MockLLMClient.complete() is called
    with it and returns real JSON text (not a canned test double), and
    LLMAsJudge parses that real response back into a JudgeVerdict."""

    def test_semantically_correct_answer_passes_most_of_the_time(self):
        client = MockLLMClient()
        judge = LLMAsJudge(client)
        expected = "The capital of France is Paris."
        wrong = 0
        n = 40
        for i in range(n):
            actual = f"The capital of France is Paris, variant number {i}."
            if not judge.evaluate(actual, expected).passed:
                wrong += 1
        assert wrong / n < 0.30

    def test_semantically_wrong_answer_fails_most_of_the_time(self):
        client = MockLLMClient()
        judge = LLMAsJudge(client)
        expected = "The capital of France is Paris."
        wrong = 0
        n = 40
        for i in range(n):
            actual = f"I have no idea what you're asking about, response {i}."
            if judge.evaluate(actual, expected).passed:
                wrong += 1
        assert wrong / n < 0.30

    def test_deterministic_for_same_inputs(self):
        client = MockLLMClient()
        judge = LLMAsJudge(client)
        v1 = judge.evaluate("Paris is the capital.", "The capital of France is Paris.")
        v2 = judge.evaluate("Paris is the capital.", "The capital of France is Paris.")
        assert v1.passed == v2.passed
        assert v1.reason == v2.reason

    def test_reason_is_a_non_empty_string(self):
        client = MockLLMClient()
        judge = LLMAsJudge(client)
        v = judge.evaluate("Some answer.", "Some expected answer with enough words in it.")
        assert isinstance(v.reason, str)
        assert len(v.reason) > 0

    def test_mock_judge_has_a_genuine_error_rate_not_always_right(self):
        """Regression-style check (mirrors the analogous test in
        test_llm_client.py for capital-question answers) that the
        deliberately injected ~12% wrong-verdict rate in
        MockLLMClient._mock_judge_verdict is actually present."""
        client = MockLLMClient()
        judge = LLMAsJudge(client)
        expected = "The capital of France is Paris, a major European city."
        disagreements = 0
        for i in range(60):
            actual = f"Paris is indeed the capital of France. (variant {i})"
            v = judge.evaluate(actual, expected)
            if v.passed is False:
                disagreements += 1
        assert 0 < disagreements < 30
