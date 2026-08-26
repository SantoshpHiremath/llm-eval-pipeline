"""
Builds a real (if synthetic) evaluation dataset with a genuine golden vs.
silver split: golden items are hand-written and hand-verified (a human,
in this case me, actually decided each expected_output is correct);
silver items simulate the "auto-collected from production traffic" case
-- a real input paired with what a *previous* trusted model version
produced, not independently re-verified with the same confidence.
"""
from __future__ import annotations

from src.datasets import DatasetItem, DatasetTier, EvalDataset


def build_sample_dataset() -> EvalDataset:
    ds = EvalDataset(name="sample_qa_eval_set")

    # --- Golden items: hand-verified, high-confidence ground truth ---
    golden_items = [
        DatasetItem("G1", DatasetTier.GOLDEN, "What is the capital of France?", "The capital of France is Paris.", verified_by_human=True, tags=["geography"]),
        DatasetItem("G2", DatasetTier.GOLDEN, "What is the capital of Germany?", "The capital of Germany is Berlin.", verified_by_human=True, tags=["geography"]),
        DatasetItem("G3", DatasetTier.GOLDEN, "What is the capital of Japan?", "The capital of Japan is Tokyo.", verified_by_human=True, tags=["geography"]),
        DatasetItem("G4", DatasetTier.GOLDEN, "What is 12 + 30?", "The sum is 42.", verified_by_human=True, tags=["arithmetic"]),
        DatasetItem("G5", DatasetTier.GOLDEN, "What is 100 + 250?", "The sum is 350.", verified_by_human=True, tags=["arithmetic"]),
        DatasetItem("G6", DatasetTier.GOLDEN, "Can I get a refund within the 30 day policy?", "Yes, you're eligible for a refund within our 30-day policy window.", verified_by_human=True, tags=["policy"]),
        DatasetItem("G7", DatasetTier.GOLDEN, "What is the capital of Italy?", "The capital of Italy is Rome.", verified_by_human=True, tags=["geography"]),
        DatasetItem("G8", DatasetTier.GOLDEN, "What is 7 + 8?", "The sum is 15.", verified_by_human=True, tags=["arithmetic"]),
    ]
    for item in golden_items:
        ds.add(item)

    silver_items = [
        DatasetItem("S1", DatasetTier.SILVER, "What is the capital of Spain?", "The capital of Spain is Madrid.", verified_by_human=False, tags=["geography"]),
        DatasetItem("S2", DatasetTier.SILVER, "What is the capital of India?", "The capital of India is New Delhi.", verified_by_human=False, tags=["geography"]),
        DatasetItem("S3", DatasetTier.SILVER, "What is 5 + 9?", "The sum is 14.", verified_by_human=False, tags=["arithmetic"]),
        DatasetItem("S4", DatasetTier.SILVER, "Can I cancel and get a refund?", "Refunds are available within 30 days of purchase, per our standard policy.", verified_by_human=False, tags=["policy"]),
        DatasetItem("S5", DatasetTier.SILVER, "What is 20 + 22?", "The sum is 42.", verified_by_human=False, tags=["arithmetic"]),
        DatasetItem("S6", DatasetTier.SILVER, "What is the capital of Germany, exactly and only?", "Berlin, and nothing else.", verified_by_human=False, tags=["geography", "edge_case"]),
    ]
    for item in silver_items:
        ds.add(item)

    return ds
