"""FastAPI server — CME as infrastructure any model can call.

The middleware shape from the brief: a client asks for context before it
prompts its model, and posts the answer back to be verified afterwards.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cme_python.cme import CME, GroundedContext
from cme_python.config import settings
from cme_python.engines.evidence import GroundingReport, Justification
from cme_python.models import Belief, SourceKind

engine: CME | None = None


def get_engine() -> CME:
    if engine is None:  # pragma: no cover - only if the app is used unstarted
        raise HTTPException(503, "CME is not running")
    return engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = CME()
    yield
    engine.close()
    engine = None


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


class ContextRequest(BaseModel):
    query: str = Field(min_length=1)
    budget: float | None = Field(default=None, gt=0)


class VerifyRequest(BaseModel):
    answer: str = Field(min_length=1)


class ContradictionOut(BaseModel):
    explanation: str
    overlap: float
    winner: str
    loser: str


@app.get("/health")
def health() -> dict[str, object]:
    cme = get_engine()
    return {"status": "ok", "beliefs": len(cme.store), "solver": settings.solver}


@app.post("/ingest", response_model=list[Belief])
def ingest(request: IngestRequest) -> list[Belief]:
    """Learn from a document. Returns the beliefs filed or reinforced."""
    return get_engine().ingest(
        request.text,
        source=request.source,
        locator=request.locator,
        connections=request.connections,
    )


@app.post("/context", response_model=GroundedContext)
def context(request: ContextRequest) -> GroundedContext:
    """The best grounded memories for a query, inside a token budget."""
    return get_engine().context(request.query, budget=request.budget)


@app.post("/verify", response_model=GroundingReport)
def verify(request: VerifyRequest) -> GroundingReport:
    """Check a model's answer against the registry. Unsupported claims come back flagged."""
    return get_engine().verify(request.answer)


@app.get("/beliefs/{belief_id}", response_model=Justification)
def explain(belief_id: str) -> Justification:
    """Why the engine believes something: evidence for, against, and how certain."""
    justification = get_engine().explain(belief_id)
    if justification is None:
        raise HTTPException(404, f"No belief {belief_id}")
    return justification


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
