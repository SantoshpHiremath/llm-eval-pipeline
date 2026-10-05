# LLM Evaluation Pipeline

A tested Python project for building and running evaluation pipelines for LLM-based systems: golden and silver datasets, grading strategies including an LLM-as-judge, structured tracing, and a REST API with authentication.

## What it does

This project does five things, chained together:

1. A Golden vs. Silver dataset model with a clear trust distinction between the two tiers.
2. An LLM client abstraction with a fully tested mock backend and OpenAI/Anthropic clients written to the real SDKs' call shapes.
3. A judge/evaluation harness with three grading strategies: exact-match, keyword-overlap, and an LLM-as-judge that uses an `LLMClient` to grade another model's output.
4. Structured tracing modeled on Langfuse's trace/span/score concepts.
5. A FastAPI service with API-key authentication.

## Scope

- **Mock backend.** Every test and `run_pipeline.py` use `MockLLMClient`, a deterministic, hash-seeded backend with no network calls. It deliberately injects realistic failure modes (an occasional wrong capital-city answer, an occasional off-by-one arithmetic slip, an occasional evasive non-answer) so the evaluation harness has varied failures to catch. All 84 tests pass against it.
- **Real-SDK clients.** `RealOpenAIClient` and `RealAnthropicClient` in `src/llm_client.py` follow the SDKs' call shapes (`client.chat.completions.create(...)` for OpenAI, `client.messages.create(...)` for Anthropic). They need an API key; instantiating either without one raises an explicit error, so a mock response is never mistaken for a live one. Running them against a live API is the natural next step.
- **LLM-as-judge.** `LLMAsJudge` in `judge.py` takes any `LLMClient` and uses it to grade another model's output by prompting it to compare actual vs. expected and parse back a structured JSON verdict. It is exercised end to end against `MockLLMClient`: prompt construction, a `.complete()` call, JSON parsing, and handling of a malformed or unparseable judge response (reported as a failure, never silently swallowed). 17 tests cover it. Verdict quality with a live model (GPT-4o-mini, Claude, etc.) would be measured by pointing the same code path at the real clients. `ExactMatchJudge` and `KeywordJudge` remain available as complementary grading strategies.
- **Tracing.** `tracing.py` is my own lightweight in-memory implementation of the trace/span/score data model that Langfuse uses; it is separate from the Langfuse SDK/platform.

## Golden vs. Silver datasets (`src/datasets.py`)

- **Golden** items are hand-verified, high-confidence ground truth: a human has reviewed the expected output and signed off that it is correct. `DatasetItem` enforces this structurally: constructing a golden item with `verified_by_human=False` raises immediately.
- **Silver** items are lower-confidence and typically broader-coverage, simulating inputs collected from real usage paired with an answer that has not been independently re-verified with the same confidence. A silver failure means "this needs human review," not automatically "the model regressed." `sample_dataset.py` includes a deliberate example (`S6`) where the *expected* output is written too strictly, and the evaluation flags it as a silver failure rather than a golden-level alarm.
- `EvalDataset.promote_to_golden()` implements the workflow by which a reviewed silver item becomes trusted golden ground truth over time.

## Results

Running `run_pipeline.py` against the mock backend: 14 items (8 golden, 6 silver).

- Golden pass rate: 87.5% (1 failing: an injected arithmetic error, `G5`).
- Silver pass rate: 66.7% (2 failing: an injected factual error `S2`, and `S6`, where the *expected* output was too strict, which illustrates why silver failures need review rather than an automatic alarm).

These numbers are the real outputs of `MockLLMClient`'s deliberately imperfect responses, which shows the evaluation logic detecting failures rather than reporting a uniform 100%.

Running the same dataset through `LLMAsJudge` instead of `KeywordJudge` (also in `run_pipeline.py`'s output) gives a different overall pass rate (71.4% vs. 78.6%). The two judges disagree on some items: `KeywordJudge` requires exact keyword presence, while `LLMAsJudge` makes a coverage-based judgment and has its own deliberately injected ~12% wrong-verdict rate, simulating a judge model's imperfect reliability. Neither is "more correct" in the abstract, so the pipeline reports per-judge results rather than treating one grading method as ground truth.

## REST API with authentication (`src/api.py`)

`GET /health` (no auth), `POST /ask` and `POST /evaluate` (both require a valid `x-api-key` header, returning 401 otherwise). It is built on FastAPI with Pydantic request validation (question length limits, non-empty checks). `tests/test_api.py` confirms both the 401-without-key and 200-with-key cases. The LLM client is dependency-injected (`get_llm_client()`), so swapping the mock for `RealOpenAIClient`/`RealAnthropicClient` means changing exactly one function, not the route handlers, the evaluator, or any test.

## Tests

84 tests across the datasets, judges, evaluator, LLM clients, tracing and API:

```bash
pytest tests/ -v
```

## Project structure

```
run_pipeline.py             end-to-end demo against the mock backend
src/datasets.py             golden/silver dataset model
src/sample_dataset.py       sample dataset (14 items)
src/llm_client.py           LLMClient interface, mock and real-SDK clients
src/judge.py                ExactMatchJudge, KeywordJudge, LLMAsJudge
src/evaluator.py            evaluation runner and reports
src/tracing.py              trace/span/score data model
src/api.py                  FastAPI service
tests/                      test suite
.github/workflows/ci.yml    CI running the tests
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v              # 84 tests
python run_pipeline.py        # end-to-end demo against the mock backend
uvicorn src.api:app --reload  # REST API (set EVAL_PIPELINE_API_KEY env var)
```

## Notes

- Results shown come from the mock backend; they illustrate the evaluation logic rather than any particular model's quality.
- The `LLMAsJudge` mechanism (prompt building, `.complete()` call, JSON parsing, error handling) is tested end to end with the mock client.

## Possible extensions

- Run `RealOpenAIClient` / `RealAnthropicClient` against a live API and handle rate limits, malformed JSON, content filtering and streaming.
- Integrate the Langfuse SDK to ship traces to its platform.
- Add retry/backoff, rate limiting and a production deployment setup for the API.
- Add A/B comparison of two prompt or model variants, with statistical tests on the judged outputs.
