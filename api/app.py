import asyncio
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

CREATE = re.compile(r"^\s*create\s+(table|view)\b", re.IGNORECASE)


@dataclass
class Settings:
    origins: list = field(default_factory=lambda: ["http://localhost:3000"])
    default_samples: int = 4
    max_samples: int = 16
    max_statements: int = 40
    max_statement_chars: int = 2000
    max_schema_chars: int = 8000
    max_question_chars: int = 300
    max_prompt_tokens: int = 1800
    rate_per_minute: int = 20
    request_timeout: float = 60.0
    temperature: float = 0.7
    feedback_path: str = "feedback.jsonl"


class RateLimiter:
    """At most `limit` requests per `window` seconds for each key (a sliding window)."""

    def __init__(self, limit, window=60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key):
        with self.lock:
            now = self.clock()
            if len(self.hits) > 10000:  # forget keys that have gone quiet
                stale = [k for k, q in self.hits.items() if not q or now - q[-1] > self.window]
                for k in stale:
                    del self.hits[k]
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def client_ip(request):
    """The caller's address. Behind a proxy (such as a Hugging Face Space) the first
    X-Forwarded-For entry is the caller; elsewhere that header could be spoofed."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class GenerateRequest(BaseModel):
    tables: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    samples: int | None = Field(default=None, ge=1)


class Feedback(BaseModel):
    question: str = Field(max_length=2000)
    sql: str = Field(max_length=4000)
    verdict: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)


def create_app(service, settings=None):
    settings = settings or Settings()
    app = FastAPI(title="QUERION")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )
    limiter = RateLimiter(settings.rate_per_minute)
    counters = defaultdict(int)
    feedback_lock = threading.Lock()

    def reject(status, reason, detail, headers=None):
        counters[f"rejected_{reason}"] += 1
        raise HTTPException(status, detail, headers=headers)

    def check(body):
        if len(body.question) > settings.max_question_chars:
            reject(422, "question_too_long", "the question is too long")
        if len(body.tables) > settings.max_statements:
            reject(422, "too_many_tables", "too many tables")
        for statement in body.tables:
            if len(statement) > settings.max_statement_chars:
                reject(422, "statement_too_long", "a table definition is too long")
            if not CREATE.match(statement):
                reject(422, "not_create", "every table must be a CREATE TABLE statement")
        if sum(len(s) for s in body.tables) > settings.max_schema_chars:
            reject(422, "schema_too_long", "the schema is too long")
        samples = body.samples or settings.default_samples
        if samples > settings.max_samples:
            reject(422, "too_many_samples", f"at most {settings.max_samples} samples")
        if service.prompt_tokens(body.tables, body.question) > settings.max_prompt_tokens:
            reject(413, "prompt_too_long", "the schema and question do not fit the model")
        return samples

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return {"counters": dict(counters), "engine": service.stats()}

    @app.post("/generate")
    async def generate(body: GenerateRequest, request: Request):
        if not limiter.allow(client_ip(request)):
            reject(429, "rate_limit", "rate limit exceeded", headers={"Retry-After": "60"})
        samples = check(body)
        counters["requests"] += 1
        started = time.perf_counter()
        try:
            futures = service.start(
                body.tables, body.question, samples, settings.temperature, settings.request_timeout
            )
        except ValueError as e:
            reject(422, "engine_refused", str(e))

        async def events():
            try:
                yield sse("start", {"samples": samples})
                pending = {asyncio.wrap_future(f): i for i, f in enumerate(futures)}
                candidates = [None] * samples
                while pending:
                    done, _ = await asyncio.wait(list(pending), return_when=asyncio.FIRST_COMPLETED)
                    for finished in done:
                        i = pending.pop(finished)
                        try:
                            candidates[i] = service.decode(finished.result())
                        except Exception as e:
                            counters["candidate_errors"] += 1
                            yield sse("error", {"index": i, "message": type(e).__name__})
                            continue
                        counters["candidates"] += 1
                        yield sse("candidate", {"index": i, "sql": candidates[i]})
                seconds = round(time.perf_counter() - started, 2)
                yield sse("done", {"candidates": candidates, "seconds": seconds})
            finally:
                for f in futures:  # a client that left must not keep the engine busy
                    f.cancel()

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.post("/feedback")
    def feedback(body: Feedback, request: Request):
        if not limiter.allow("feedback:" + client_ip(request)):
            reject(429, "rate_limit", "rate limit exceeded", headers={"Retry-After": "60"})
        record = {**body.model_dump(), "at": time.time()}
        with feedback_lock, open(Path(settings.feedback_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        counters["feedback"] += 1
        return {"ok": True}

    return app
