import time

import pytest

from src.tracing import Trace, TraceStore


class TestSpan:
    def test_span_duration_after_close(self):
        trace = Trace("t1")
        trace.start_span("step1")
        time.sleep(0.01)
        span = trace.end_span()
        assert span.duration_seconds > 0

    def test_duration_raises_if_span_not_closed(self):
        trace = Trace("t1")
        span = trace.start_span("step1")
        with pytest.raises(ValueError):
            _ = span.duration_seconds


class TestTrace:
    def test_cannot_start_a_span_while_one_is_open(self):
        trace = Trace("t1")
        trace.start_span("step1")
        with pytest.raises(RuntimeError):
            trace.start_span("step2")

    def test_ending_with_no_open_span_raises(self):
        trace = Trace("t1")
        with pytest.raises(RuntimeError):
            trace.end_span()

    def test_multiple_sequential_spans_all_recorded(self):
        trace = Trace("t1")
        trace.start_span("step1")
        trace.end_span()
        trace.start_span("step2")
        trace.end_span()
        assert len(trace.spans) == 2
        assert [s.name for s in trace.spans] == ["step1", "step2"]

    def test_add_score_and_get_score(self):
        trace = Trace("t1")
        trace.add_score("judge_pass", 1.0, comment="ok")
        score = trace.get_score("judge_pass")
        assert score.value == 1.0
        assert score.comment == "ok"

    def test_get_score_raises_for_unknown_name(self):
        trace = Trace("t1")
        with pytest.raises(KeyError):
            trace.get_score("nonexistent")

    def test_get_score_returns_latest_if_added_twice(self):
        trace = Trace("t1")
        trace.add_score("s", 0.0)
        trace.add_score("s", 1.0)
        assert trace.get_score("s").value == 1.0

    def test_total_duration_sums_all_spans(self):
        trace = Trace("t1")
        trace.start_span("a")
        trace.end_span()
        trace.start_span("b")
        trace.end_span()
        assert trace.total_duration_seconds() >= 0


class TestTraceStore:
    def _passing_trace(self, name):
        t = Trace(name)
        t.add_score("judge_pass", 1.0)
        return t

    def _failing_trace(self, name):
        t = Trace(name)
        t.add_score("judge_pass", 0.0)
        return t

    def test_pass_rate_with_mixed_results(self):
        store = TraceStore()
        store.add(self._passing_trace("t1"))
        store.add(self._passing_trace("t2"))
        store.add(self._failing_trace("t3"))
        assert store.pass_rate() == pytest.approx(2 / 3)

    def test_pass_rate_is_zero_for_empty_store(self):
        store = TraceStore()
        assert store.pass_rate() == 0.0

    def test_failing_traces_returns_only_failures(self):
        store = TraceStore()
        store.add(self._passing_trace("t1"))
        store.add(self._failing_trace("t2"))
        failing = store.failing_traces()
        assert len(failing) == 1
        assert failing[0].name == "t2"

    def test_mean_latency_over_traces_with_spans(self):
        store = TraceStore()
        t1 = Trace("t1")
        t1.start_span("s")
        time.sleep(0.005)
        t1.end_span()
        store.add(t1)
        assert store.mean_latency_seconds() > 0

    def test_mean_latency_is_zero_with_no_spans(self):
        store = TraceStore()
        store.add(Trace("t1"))
        assert store.mean_latency_seconds() == 0.0
