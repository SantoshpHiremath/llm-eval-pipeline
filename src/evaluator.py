"""
Ties together an LLMClient, a Judge, and a dataset (golden/silver) into
one evaluation run, producing a TraceStore of results plus a summary
report broken down by dataset tier -- silver-set failures are reported
separately from golden-set failures (see datasets.py docstring for why
that distinction matters: a silver failure might mean the "expected"
answer itself is wrong, not necessarily that the model regressed).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.datasets import DatasetItem, DatasetTier, EvalDataset
from src.judge import Judge
from src.llm_client import LLMClient
from src.tracing import Trace, TraceStore


@dataclass
class EvalReport:
    total_items: int
    golden_pass_rate: float
    silver_pass_rate: float
    overall_pass_rate: float
    mean_latency_seconds: float
    failing_golden_ids: list
    failing_silver_ids: list


def evaluate_item(client: LLMClient, judge: Judge, item: DatasetItem, system_prompt: str = "") -> Trace:
    trace = Trace(name=f"eval:{item.item_id}", metadata={"tier": item.tier.value, "tags": item.tags})

    trace.start_span("llm_call", metadata={"item_id": item.item_id})
    response = client.complete(item.input_text, system=system_prompt)
    trace.end_span()

    trace.start_span("judge")
    verdict = judge.evaluate(response.text, item.expected_output)
    trace.end_span()

    trace.add_score("judge_pass", 1.0 if verdict.passed else 0.0, comment=verdict.reason)
    trace.metadata["actual_output"] = response.text
    trace.metadata["model"] = response.model
    return trace


def run_evaluation(client: LLMClient, judge: Judge, dataset: EvalDataset, system_prompt: str = "") -> tuple:
    """Runs every item in the dataset through the given client + judge,
    returns (TraceStore, EvalReport).
    """
    store = TraceStore()
    for item in dataset.all_items():
        trace = evaluate_item(client, judge, item, system_prompt=system_prompt)
        store.add(trace)

    golden_ids = {i.item_id for i in dataset.golden_items()}
    silver_ids = {i.item_id for i in dataset.silver_items()}

    def _pass_rate_for(ids: set) -> float:
        subset = [t for t in store.traces if t.metadata.get("item_id", t.name.split(":")[-1]) in ids] \
            if ids else []
        # trace names are "eval:<item_id>" — recover item_id from the name
        subset = [t for t in store.traces if t.name.split("eval:", 1)[-1] in ids]
        if not subset:
            return 0.0
        passed = sum(1 for t in subset if t.get_score("judge_pass").value == 1.0)
        return passed / len(subset)

    golden_rate = _pass_rate_for(golden_ids)
    silver_rate = _pass_rate_for(silver_ids)

    failing_golden = [
        t.name.split("eval:", 1)[-1] for t in store.traces
        if t.name.split("eval:", 1)[-1] in golden_ids and t.get_score("judge_pass").value == 0.0
    ]
    failing_silver = [
        t.name.split("eval:", 1)[-1] for t in store.traces
        if t.name.split("eval:", 1)[-1] in silver_ids and t.get_score("judge_pass").value == 0.0
    ]

    report = EvalReport(
        total_items=len(dataset),
        golden_pass_rate=golden_rate,
        silver_pass_rate=silver_rate,
        overall_pass_rate=store.pass_rate(),
        mean_latency_seconds=store.mean_latency_seconds(),
        failing_golden_ids=failing_golden,
        failing_silver_ids=failing_silver,
    )
    return store, report
