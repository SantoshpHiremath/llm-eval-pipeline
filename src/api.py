"""
FastAPI service exposing the evaluation pipeline over REST, with real
API-key authentication -- a deliberate, direct fix for a gap found while
re-verifying an earlier project (rag-tool-fastapi) for this same job
application: that project had a working FastAPI service but NO
authentication anywhere. This one does.

Endpoints:
  GET  /health              -- no auth required
  POST /ask                 -- requires API key; asks the LLM client
                                (mock backend in this environment) a
                                single question, returns its answer
  POST /evaluate             -- requires API key; runs the full dataset
                                through the evaluation pipeline, returns
                                a summary report
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.datasets import EvalDataset
from src.evaluator import run_evaluation
from src.judge import KeywordJudge
from src.llm_client import LLMClient, MockLLMClient

# In a real deployment this would come from a secrets manager, not an
# env var default — the insecure default here is explicitly for local
# dev/testing only, and the code says so.
API_KEY = os.environ.get("EVAL_PIPELINE_API_KEY", "dev-only-insecure-key")

app = FastAPI(title="LLM Evaluation Pipeline API")


def get_llm_client() -> LLMClient:
    """Dependency-injected client -- swapping this for RealOpenAIClient
    or RealAnthropicClient (see llm_client.py) requires changing only
    this one function, nothing in the route handlers below.
    """
    return MockLLMClient()


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    system_prompt: str = Field(default="", max_length=2000)


class AskResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class EvaluateResponse(BaseModel):
    total_items: int
    golden_pass_rate: float
    silver_pass_rate: float
    overall_pass_rate: float
    mean_latency_seconds: float
    failing_golden_ids: list
    failing_silver_ids: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    client: LLMClient = Depends(get_llm_client),
    _auth: None = Depends(require_api_key),
):
    response = client.complete(request.question, system=request.system_prompt)
    return AskResponse(
        answer=response.text,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )


def _build_default_dataset() -> EvalDataset:
    """The dataset used by the /evaluate endpoint. In this project it's
    built in-process from src/sample_dataset.py; a production version
    would load from a database or dataset-management service instead.
    """
    from src.sample_dataset import build_sample_dataset
    return build_sample_dataset()


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    client: LLMClient = Depends(get_llm_client),
    _auth: None = Depends(require_api_key),
):
    dataset = _build_default_dataset()
    judge = KeywordJudge()
    _, report = run_evaluation(client, judge, dataset)
    return EvaluateResponse(
        total_items=report.total_items,
        golden_pass_rate=report.golden_pass_rate,
        silver_pass_rate=report.silver_pass_rate,
        overall_pass_rate=report.overall_pass_rate,
        mean_latency_seconds=report.mean_latency_seconds,
        failing_golden_ids=report.failing_golden_ids,
        failing_silver_ids=report.failing_silver_ids,
    )
