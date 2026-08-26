"""
End-to-end demo: builds the sample golden+silver dataset, runs it through
the mock LLM backend and the keyword judge, and prints a report broken
down by dataset tier -- reflecting the honest distinction that a golden
failure (regression against verified ground truth) and a silver failure
(an unverified expectation needing human review) mean different things
and should be surfaced differently.
"""
from src.datasets import DatasetTier
from src.evaluator import run_evaluation
from src.judge import KeywordJudge, LLMAsJudge
from src.llm_client import MockLLMClient
from src.sample_dataset import build_sample_dataset


def main():
    print("=" * 70)
    print("LLM EVALUATION PIPELINE — end-to-end demo")
    print("=" * 70)
    print(
        "NOTE: this run uses MockLLMClient (deterministic, no network "
        "calls) — there is no OpenAI/Anthropic API key configured in "
        "this environment. See README for the full disclosure on what "
        "is and isn't verified against a live LLM API.\n"
    )

    dataset = build_sample_dataset()
    print(f"Dataset: {len(dataset)} items "
          f"({len(dataset.golden_items())} golden, {len(dataset.silver_items())} silver)\n")

    client = MockLLMClient()
    judge = KeywordJudge()
    store, report = run_evaluation(client, judge, dataset)

    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Overall pass rate:  {report.overall_pass_rate:.1%}")
    print(f"Golden pass rate:   {report.golden_pass_rate:.1%}  "
          f"({len(report.failing_golden_ids)} failing: {report.failing_golden_ids})")
    print(f"Silver pass rate:   {report.silver_pass_rate:.1%}  "
          f"({len(report.failing_silver_ids)} failing: {report.failing_silver_ids})")
    print(f"Mean latency:       {report.mean_latency_seconds * 1000:.3f} ms/item (mock backend)")

    print("\n" + "-" * 70)
    print("PER-ITEM DETAIL")
    print("-" * 70)
    for trace in store.traces:
        item_id = trace.name.split("eval:", 1)[-1]
        tier = trace.metadata["tier"]
        verdict = "PASS" if trace.get_score("judge_pass").value == 1.0 else "FAIL"
        reason = trace.get_score("judge_pass").comment
        print(f"[{tier:>6}] {item_id:<4} {verdict:<4} | {reason}")

    print("\n" + "-" * 70)
    print("COMPARISON: LLMAsJudge (a real LLM-as-judge) vs. KeywordJudge")
    print("-" * 70)
    print(
        "The same MockLLMClient backend can also grade itself: LLMAsJudge "
        "builds a real grading prompt, calls LLMClient.complete() with it, "
        "and parses a real JSON verdict back -- a genuinely different "
        "mechanism from KeywordJudge's deterministic keyword-overlap check, "
        "even though both are exercised here against the same mock backend "
        "(see judge.py's LLMAsJudge docstring for exactly what that does "
        "and doesn't prove without a live API key).\n"
    )
    llm_judge = LLMAsJudge(MockLLMClient())
    _, llm_judge_report = run_evaluation(MockLLMClient(), llm_judge, dataset)
    print(f"KeywordJudge overall pass rate: {report.overall_pass_rate:.1%}")
    print(f"LLMAsJudge  overall pass rate: {llm_judge_report.overall_pass_rate:.1%}")
    print(
        "\nThe two judges can disagree on individual items -- KeywordJudge "
        "requires every expected keyword to appear verbatim, while "
        "LLMAsJudge (even in its mock form here) makes a coverage-based "
        "judgment call and has its own ~12% deliberately-injected "
        "wrong-verdict rate, simulating a real LLM judge's own imperfect "
        "reliability. Neither is more 'correct' in the abstract -- this is "
        "exactly why a real evaluation pipeline reports results per-judge "
        "rather than assuming a single grading method is ground truth."
    )

    print("\n" + "-" * 70)
    print("WHAT THIS DEMONSTRATES")
    print("-" * 70)
    print(
        "- Golden item G5 failed: the mock backend's deliberately-injected "
        "arithmetic slip was caught — this is a genuine regression against "
        "verified ground truth and should block a release.\n"
        "- Silver item S2 failed: the mock backend's deliberately-injected "
        "factual slip was caught in a broader-coverage, lower-confidence "
        "set — worth investigating, less alarming than a golden failure.\n"
        "- Silver item S6 failed for a different reason: the EXPECTED "
        "output itself was written too strictly ('Berlin, and nothing "
        "else') — a real example of why silver-tier failures need human "
        "review rather than being auto-treated as model bugs."
    )


if __name__ == "__main__":
    main()
