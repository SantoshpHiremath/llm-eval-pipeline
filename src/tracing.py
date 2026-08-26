"""
Structured evaluation tracing, modeled on Langfuse's trace/span/score
concepts (a trace holds one end-to-end run; spans are timed sub-steps
within it; scores attach a named numeric/boolean judgment to a trace) --
named explicitly as "modeled on" Langfuse rather than "built with"
Langfuse: this project implements its own lightweight in-memory version
of that data model, it is NOT the actual Langfuse SDK/platform. Real
Langfuse integration would mean calling their SDK to ship traces to
their backend; this instead demonstrates understanding of the concepts
(trace = one run, span = a step, score = a judgment attached to a trace)
with a real, tested, swappable implementation of the same shape.

This distinction is stated here and repeated in the README so it's never
presented as "Langfuse experience" when it's actually "built the same
conceptual model Langfuse uses, tested independently."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            raise ValueError(f"Span {self.name!r} was never closed.")
        return self.end_time - self.start_time


@dataclass
class Score:
    name: str
    value: float
    comment: str = ""


class Trace:
    """One end-to-end evaluation run (e.g. 'evaluate item PS1 against the
    golden dataset') -- holds an ordered list of spans (its sub-steps,
    e.g. 'llm_call', 'judge') and any scores attached after the fact.
    """

    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.metadata = metadata or {}
        self.spans: list[Span] = []
        self.scores: list[Score] = []
        self._open_span: Span | None = None

    def start_span(self, name: str, metadata: dict | None = None) -> Span:
        if self._open_span is not None:
            raise RuntimeError(
                f"Span {self._open_span.name!r} is still open — "
                f"close it before starting {name!r}. Nesting isn't supported."
            )
        span = Span(name=name, start_time=time.monotonic(), metadata=metadata or {})
        self._open_span = span
        return span

    def end_span(self) -> Span:
        if self._open_span is None:
            raise RuntimeError("No open span to end.")
        span = self._open_span
        span.end_time = time.monotonic()
        self.spans.append(span)
        self._open_span = None
        return span

    def add_score(self, name: str, value: float, comment: str = "") -> None:
        self.scores.append(Score(name=name, value=value, comment=comment))

    def get_score(self, name: str) -> Score:
        matches = [s for s in self.scores if s.name == name]
        if not matches:
            raise KeyError(f"No score named {name!r} on this trace.")
        return matches[-1]

    def total_duration_seconds(self) -> float:
        return sum(s.duration_seconds for s in self.spans)


class TraceStore:
    """In-memory collection of traces from a batch evaluation run, with
    aggregate reporting -- the "analyze metrics, identify edge cases,
    track quality" part of the job posting, done over real (if mocked-
    backend) traces rather than only single-item output.
    """

    def __init__(self):
        self.traces: list[Trace] = []

    def add(self, trace: Trace) -> None:
        self.traces.append(trace)

    def pass_rate(self, score_name: str = "judge_pass") -> float:
        if not self.traces:
            return 0.0
        scored = [t for t in self.traces if any(s.name == score_name for s in t.scores)]
        if not scored:
            return 0.0
        passed = sum(1 for t in scored if t.get_score(score_name).value == 1.0)
        return passed / len(scored)

    def failing_traces(self, score_name: str = "judge_pass") -> list:
        return [
            t for t in self.traces
            if any(s.name == score_name for s in t.scores)
            and t.get_score(score_name).value == 0.0
        ]

    def mean_latency_seconds(self) -> float:
        if not self.traces:
            return 0.0
        durations = [t.total_duration_seconds() for t in self.traces if t.spans]
        return sum(durations) / len(durations) if durations else 0.0
