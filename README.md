# LLM Evaluation Pipeline

A real, tested Python project built specifically to close a gap for AssetMetrix's
"Working Student – AI Engineering" posting. The role centers on building and
running evaluation pipelines for LLM-based systems, maintaining Golden and
Silver datasets, working with evaluation tooling like Langfuse, and
developing Python/REST-API workflows around AI systems — none of which
anything in my prior project portfolio actually demonstrated together,
despite several projects touching adjacent pieces individually.

This project does five things, chained together: (1) a real Golden vs.
Silver dataset model with an honest trust distinction between the two
tiers, (2) an LLM client abstraction with a fully-tested mock backend and
real-SDK-shaped (but never-executed) OpenAI/Anthropic clients, (3) a
judge/evaluation harness with three grading strategies -- exact-match,
keyword-overlap, and a genuine LLM-as-judge that uses an LLMClient to
grade another model's output -- (4) structured tracing modeled on
Langfuse's trace/span/score concepts, and (5) a real FastAPI service with
actual API-key authentication.

## What this is — read before citing anywhere (the most important section)

**This sandbox has no OpenAI or Anthropic API key configured.** `RealOpenAIClient`
and `RealAnthropicClient` in `src/llm_client.py` are written to match those
SDKs' real call shapes (`client.chat.completions.create(...)` for OpenAI,
`client.messages.create(...)` for Anthropic) as closely as I could without
being able to run them — but "written to match the real interface" is a
different, weaker claim than "tested against a live API," and I'm stating
that difference directly rather than letting the code's realistic shape
imply more than it demonstrates. Both classes raise immediately with an
explicit "never been executed in this environment" error if instantiated
without a real key, specifically so it's structurally impossible to
mistake a mock response for a verified real one.

**What every test in this project actually exercises is `MockLLMClient`** —
a deterministic, hash-seeded fake backend with no network calls, used by
all 84 passing tests and by `run_pipeline.py`. It deliberately injects
realistic failure modes (an occasional wrong capital-city answer, an
occasional off-by-one arithmetic slip, an occasional evasive non-answer)
so the evaluation harness downstream has genuine, varied failures to
catch — not just uniform success dressed up as a "pipeline."

**Update: a real "LLM-as-judge" is now implemented.** `judge.py`'s
`LLMAsJudge` takes any `LLMClient` and uses it to grade another model's
output by prompting it to compare actual vs. expected and parse back a
structured JSON verdict — genuinely an LLM judging an LLM, mechanically,
not a keyword heuristic relabeled. Exercised end to end against
`MockLLMClient` (the only `LLMClient` genuinely runnable in this
environment — same constraint as everywhere else in this file): real
prompt construction, a real `.complete()` call, real JSON parsing,
including honest handling of a malformed/unparseable judge response
(reported as a failure, never silently swallowed or crashed on). What
this does and does not prove is stated directly in `LLMAsJudge`'s own
docstring: the mechanism is real and tested (17 new tests), but whether
a *live* model (GPT-4o-mini, Claude, etc.) would produce well-calibrated
verdicts through this same code path is unverified, for the same
no-API-key reason as `RealOpenAIClient`/`RealAnthropicClient` below.
`ExactMatchJudge` and `KeywordJudge` remain available too — different,
complementary grading strategies, not superseded by `LLMAsJudge`.
`tracing.py` is explicitly labeled "modeled on Langfuse's concepts"
rather than "built with Langfuse" — it's my own lightweight in-memory
implementation of the trace/span/score data model Langfuse uses, not an
integration with the actual Langfuse SDK/platform.

I'm disclosing all of this up front, the same way every other project in
this application campaign discloses its real boundaries, because the
value of this project is the architecture and the tested logic around it
— not a false claim of live-API experience I don't have.

## The real gap this closes (and the real gap it doesn't)

Before building this, my portfolio's closest projects were
`eval-framework-golden-dataset` (a real golden-set evaluation harness, but
no "silver" tier concept, no LLM API shape, no REST layer) and
`rag-tool-fastapi` (a real, tested FastAPI service, but with **no
authentication at all**). This project fixes both gaps directly: a real
golden/silver two-tier dataset model (see below), and a FastAPI service
where every non-health endpoint requires a valid `x-api-key` header,
tested by confirming both the 401-without-key and 200-with-key cases
actually happen (`tests/test_api.py`).

What it does **not** close: actual hands-on experience calling a live LLM
API and handling its real failure modes (rate limits, malformed JSON,
content filtering, streaming) — that gap is honestly still open, and
would need a real API key to close for real.

## Golden vs. Silver datasets (`src/datasets.py`)

- **Golden** items are hand-verified, high-confidence ground truth — a
  human has reviewed the expected output and signed off it's correct.
  `DatasetItem` structurally enforces this: constructing a golden item
  with `verified_by_human=False` raises immediately.
- **Silver** items are lower-confidence, typically broader-coverage —
  simulating inputs collected from real usage paired with an answer that
  hasn't been independently re-verified with the same confidence. A
  silver item's failure means "this needs human review," not
  automatically "the model regressed" — `sample_dataset.py` includes a
  deliberate example (`S6`) where the *expected* output itself is written
  too strictly, and the evaluation correctly flags it as a silver failure
  rather than a golden-level alarm.
- `EvalDataset.promote_to_golden()` implements the real workflow by which
  a reviewed silver item becomes trusted golden ground truth over time.

## Evaluation results (running `run_pipeline.py` against the mock backend)

14 items (8 golden, 6 silver). Golden pass rate: 87.5% (1 failing — a
genuine injected arithmetic error, `G5`). Silver pass rate: 66.7% (2
failing — one a genuine injected factual error `S2`, one `S6` a case
where the *expected* output was too strict, illustrating exactly why
silver failures need review rather than automatic alarm). These numbers
are real outputs of `MockLLMClient`'s deliberately-imperfect responses,
not hand-picked to look good — a "pipeline" that only ever shows 100%
pass wouldn't actually prove the evaluation logic can detect a failure.

Running the same dataset through `LLMAsJudge` instead of `KeywordJudge`
(also in `run_pipeline.py`'s output) gives a genuinely different overall
pass rate (71.4% vs. 78.6%) — the two judges disagree on some items,
which is expected and honest: `KeywordJudge` requires exact keyword
presence, `LLMAsJudge` makes a coverage-based judgment call and has its
own deliberately-injected ~12% wrong-verdict rate simulating a real
judge model's imperfect reliability. Neither is "more correct" in the
abstract; a real evaluation pipeline reports per-judge results rather
than treating one grading method as ground truth.

## REST API with real authentication (`src/api.py`)

`GET /health` (no auth) — `POST /ask` and `POST /evaluate` (both require
a valid `x-api-key` header, returning 401 otherwise). Backed by FastAPI +
Pydantic request validation (question length limits, non-empty checks).
The LLM client is dependency-injected (`get_llm_client()`), so swapping
the mock for `RealOpenAIClient`/`RealAnthropicClient` once a real key
exists requires changing exactly one function, not the route handlers,
the evaluator, or any test.

## What this doesn't demonstrate

- No real LLM API usage — see disclosure above, this is the most
  important limitation to be upfront about for this specific application.
- No real Langfuse SDK/platform integration — a self-built, tested
  approximation of its trace/span/score concepts, named as such.
- No production deployment, rate limiting, retry/backoff logic, or
  streaming response handling — all real considerations for a production
  LLM-serving API that this project doesn't attempt.
- No A/B testing logic (statistically comparing two prompt/model
  variants) — `LLMAsJudge` grades a single output against an expected
  answer, it does not compare two candidate outputs against each other.
  (Separately, `pricing-ab-test-analysis` elsewhere in this portfolio has
  real two-sample hypothesis-testing statistics, just for a pricing
  domain rather than LLM outputs.)
- `LLMAsJudge`'s judgment quality against a *live* model is unverified —
  see the disclosure above. The mechanism (prompt building, real
  `.complete()` call, JSON parsing, error handling) is real and tested;
  whether a real GPT-4o-mini/Claude call through it produces
  well-calibrated verdicts has not been measured.

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v              # 84 tests
python run_pipeline.py        # end-to-end demo against the mock backend
uvicorn src.api:app --reload  # REST API (set EVAL_PIPELINE_API_KEY env var)
```
