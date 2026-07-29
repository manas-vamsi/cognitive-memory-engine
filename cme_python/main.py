"""FastAPI server — CME as infrastructure any model can call.

The middleware shape from the brief: a client asks for context before it
prompts its model, and posts the answer back to be verified afterwards.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cme_python.clients.base import GroundedAnswer, GroundedClient, build_client
from cme_python.cme import CME, GroundedContext
from cme_python.config import settings
from cme_python.engines.evidence import GroundingReport, Justification
from cme_python.engines.memory import DEFAULT_HALF_LIFE_DAYS, MemoryStats
from cme_python.engines.reasoning import Resolution
from cme_python.models import Belief, MemoryTier, Revision, SourceKind

engine: CME | None = None
chat: GroundedClient | None = None

log = logging.getLogger(__name__)


def get_engine() -> CME:
    if engine is None:  # pragma: no cover - only if the app is used unstarted
        raise HTTPException(503, "CME is not running")
    return engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, chat
    engine = CME()
    chat = _build_chat(engine)
    yield
    engine.close()
    engine = chat = None


def _build_chat(cme: CME) -> GroundedClient | None:
    """Wire up the configured connector, or leave /ask disabled.

    A missing SDK or key must not stop the server booting: memory, retrieval and
    verification all work without a model, and taking the whole service down
    over an optional feature would be the wrong trade.
    """
    if not settings.llm:
        return None
    try:
        return GroundedClient(cme, build_client(settings.llm, settings.llm_model))
    except (RuntimeError, ValueError) as exc:
        log.warning("/ask disabled: %s", exc)
        return None


app = FastAPI(
    title="Cognitive Memory Engine",
    summary="Persistent, explainable memory for any reasoning client.",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: SourceKind = SourceKind.UNKNOWN
    locator: str | None = None
    connections: list[str] = Field(default_factory=list)
    tier: MemoryTier = MemoryTier.GENERAL
    scope: str | None = None


class ContextRequest(BaseModel):
    query: str = Field(min_length=1)
    budget: float | None = Field(default=None, gt=0)
    tier: MemoryTier | None = None
    scope: str | None = None


class VerifyRequest(BaseModel):
    answer: str = Field(min_length=1)
    tier: MemoryTier | None = None
    scope: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    budget: float | None = Field(default=None, gt=0)


class ContradictionOut(BaseModel):
    explanation: str
    overlap: float
    winner: str
    loser: str


@app.get("/health")
def health() -> dict[str, object]:
    cme = get_engine()
    return {
        "status": "ok",
        "beliefs": len(cme.store),
        "solver": settings.solver,
        "ask_enabled": chat is not None,
    }


@app.post("/ingest", response_model=list[Belief])
def ingest(request: IngestRequest) -> list[Belief]:
    """Learn from a document. Returns the beliefs filed or reinforced."""
    return get_engine().ingest(
        request.text,
        source=request.source,
        locator=request.locator,
        connections=request.connections,
        tier=request.tier,
        scope=request.scope,
    )


@app.post("/context", response_model=GroundedContext)
def context(request: ContextRequest) -> GroundedContext:
    """The best grounded memories for a query, inside a token budget."""
    return get_engine().context(
        request.query, budget=request.budget, tier=request.tier, scope=request.scope
    )


@app.post("/verify", response_model=GroundingReport)
def verify(request: VerifyRequest) -> GroundingReport:
    """Check a model's answer against the registry. Unsupported claims come back flagged."""
    return get_engine().verify(request.answer, tier=request.tier, scope=request.scope)


@app.post("/ask", response_model=GroundedAnswer)
def ask(request: AskRequest) -> GroundedAnswer:
    """Answer a question through a model, with memories attached and the reply verified.

    Disabled unless `CME_LLM` names a connector — CME is useful without one, and
    a 503 that says what to set beats an obscure auth error from a vendor SDK.
    """
    if chat is None:
        raise HTTPException(
            503,
            "No LLM configured. Set CME_LLM=claude or CME_LLM=openai (plus the "
            "vendor API key) to enable /ask. Every other endpoint works without it.",
        )
    return chat.ask(request.question, budget=request.budget)


@app.post("/split", response_model=list[Belief])
def split() -> list[Belief]:
    """Break multi-claim beliefs apart so each can be judged on its own."""
    return get_engine().beliefs.split_all()


@app.post("/decay")
def decay(half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> dict[str, int]:
    """Age beliefs nothing has reinforced. Returns how many moved."""
    return {"decayed": get_engine().decay(half_life_days=half_life_days)}


@app.get("/memory", response_model=MemoryStats)
def memory() -> MemoryStats:
    """How much is remembered, broken down by tier."""
    return get_engine().stats()


@app.get("/beliefs/{belief_id}", response_model=Justification)
def explain(belief_id: str) -> Justification:
    """Why the engine believes something: evidence for, against, and how certain."""
    justification = get_engine().explain(belief_id)
    if justification is None:
        raise HTTPException(404, f"No belief {belief_id}")
    return justification


@app.get("/beliefs/{belief_id}/timeline", response_model=list[Revision])
def timeline(belief_id: str) -> list[Revision]:
    """How a belief's confidence got to where it is: every change, in order."""
    history = get_engine().timeline(belief_id)
    if not history:
        raise HTTPException(404, f"No belief {belief_id}")
    return history


@app.post("/beliefs/{belief_id}/supersede", response_model=Belief)
def supersede(belief_id: str, replaced_by: str) -> Belief:
    """Retire a belief in favour of one that replaces it. It leaves recall."""
    retired = get_engine().supersede(belief_id, replaced_by)
    if retired is None:
        raise HTTPException(404, f"No belief {belief_id} or {replaced_by}")
    return retired


@app.get("/contradictions", response_model=list[ContradictionOut])
def contradictions() -> list[ContradictionOut]:
    """Pairs of stored beliefs that assert opposite things."""
    return [
        ContradictionOut(
            explanation=c.explain(),
            overlap=c.overlap,
            winner=c.winner.id,
            loser=c.loser.id,
        )
        for c in get_engine().contradictions()
    ]


@app.post("/contradictions/reconcile", response_model=list[Resolution])
def reconcile() -> list[Resolution]:
    """Act on every contradiction: retire the decisively beaten, weaken the rest."""
    return get_engine().reconcile()
