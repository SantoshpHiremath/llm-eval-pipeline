import pytest

from src.datasets import DatasetItem, DatasetTier, EvalDataset
from src.evaluator import evaluate_item, run_evaluation
from src.judge import ExactMatchJudge, KeywordJudge
from src.llm_client import LLMClient, LLMResponse
from src.sample_dataset import build_sample_dataset


class ScriptedClient(LLMClient):
    """A tiny test double that always returns a fixed, known response --
    used so evaluator tests can assert exact pass/fail outcomes without
    depending on MockLLMClient's hash-based variability.
    """

    def __init__(self, fixed_text: str):
        self._text = fixed_text

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        return LLMResponse(text=self._text, model="scripted-test-model", prompt_tokens=1, completion_tokens=1)


class TestEvaluateItem:
    def test_produces_a_trace_with_llm_call_and_judge_spans(self):
        item = DatasetItem("G1", DatasetTier.GOLDEN, "q", "Paris", verified_by_human=True)
        trace = evaluate_item(ScriptedClient("Paris"), KeywordJudge(), item)
        assert [s.name for s in trace.spans] == ["llm_call", "judge"]

    def test_passing_item_scores_one(self):
        item = DatasetItem("G1", DatasetTier.GOLDEN, "q", "Paris", verified_by_human=True)
        trace = evaluate_item(ScriptedClient("Paris is correct"), KeywordJudge(), item)
        assert trace.get_score("judge_pass").value == 1.0

    def test_failing_item_scores_zero(self):
        item = DatasetItem("G1", DatasetTier.GOLDEN, "q", "Paris", verified_by_human=True)
        trace = evaluate_item(ScriptedClient("I don't know"), KeywordJudge(), item)
        assert trace.get_score("judge_pass").value == 0.0

    def test_trace_metadata_records_tier_and_output(self):
        item = DatasetItem("S1", DatasetTier.SILVER, "q", "a", verified_by_human=False)
        trace = evaluate_item(ScriptedClient("a"), ExactMatchJudge(), item)
        assert trace.metadata["tier"] == "silver"
        assert trace.metadata["actual_output"] == "a"


class TestRunEvaluation:
    def test_all_correct_gives_100_percent_pass_rates(self):
        ds = EvalDataset("t")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q1", "Paris", verified_by_human=True))
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q2", "Paris", verified_by_human=False))

        store, report = run_evaluation(ScriptedClient("Paris"), KeywordJudge(), ds)
        assert report.golden_pass_rate == 1.0
        assert report.silver_pass_rate == 1.0
        assert report.overall_pass_rate == 1.0
        assert report.failing_golden_ids == []
        assert report.failing_silver_ids == []

    def test_all_wrong_gives_zero_percent_and_lists_failures(self):
        ds = EvalDataset("t")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q1", "Paris", verified_by_human=True))
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q2", "Paris", verified_by_human=False))

        store, report = run_evaluation(ScriptedClient("wrong answer"), KeywordJudge(), ds)
        assert report.golden_pass_rate == 0.0
        assert report.silver_pass_rate == 0.0
        assert report.failing_golden_ids == ["G1"]
        assert report.failing_silver_ids == ["S1"]

    def test_golden_and_silver_rates_are_tracked_independently(self):
        """The core, most important behavior: a golden failure and a
        silver failure are reported in SEPARATE rates, not blended into
        one number that would hide which kind of failure occurred.
        """
        ds = EvalDataset("t")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q1", "Paris", verified_by_human=True))
        ds.add(DatasetItem("G2", DatasetTier.GOLDEN, "q2", "Paris", verified_by_human=True))
        ds.add(DatasetItem("S1", DatasetTier.SILVER, "q3", "Paris", verified_by_human=False))

        store, report = run_evaluation(ScriptedClient("Paris"), KeywordJudge(), ds)
        assert report.golden_pass_rate == 1.0
        assert report.silver_pass_rate == 1.0

        store2, report2 = run_evaluation(ScriptedClient("nope"), KeywordJudge(), ds)
        assert report2.golden_pass_rate == 0.0
        assert report2.silver_pass_rate == 0.0

    def test_total_items_matches_dataset_size(self):
        ds = build_sample_dataset()
        store, report = run_evaluation(ScriptedClient("Paris"), KeywordJudge(), ds)
        assert report.total_items == len(ds)

    def test_mean_latency_is_positive_after_a_real_run(self):
        ds = EvalDataset("t")
        ds.add(DatasetItem("G1", DatasetTier.GOLDEN, "q1", "Paris", verified_by_human=True))
        store, report = run_evaluation(ScriptedClient("Paris"), KeywordJudge(), ds)
        assert report.mean_latency_seconds >= 0

    def test_sample_dataset_end_to_end_with_mock_client(self):
        """Full pipeline smoke test using the actual MockLLMClient (not
        the scripted test double) against the real sample dataset --
        confirms the whole thing runs without error and produces a
        sane report, mirroring what run_pipeline.py does.
        """
        from src.llm_client import MockLLMClient
        ds = build_sample_dataset()
        store, report = run_evaluation(MockLLMClient(), KeywordJudge(), ds)
        assert 0.0 <= report.overall_pass_rate <= 1.0
        assert 0.0 <= report.golden_pass_rate <= 1.0
        assert 0.0 <= report.silver_pass_rate <= 1.0
        assert report.total_items == len(ds)
