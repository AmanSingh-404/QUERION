import json
from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from api.app import RateLimiter, Settings, create_app

TABLE = "CREATE TABLE singer (singer_id INTEGER, name TEXT, age INTEGER)"


class FakeService:
    """Stands in for the engine: every candidate is ready immediately."""

    def __init__(self, prompt_tokens=100):
        self.candidates = ["SELECT 1", "SELECT 2", "SELECT 3", "SELECT 4"]
        self.tokens = prompt_tokens
        self.started = []

    def prompt_tokens(self, statements, question):
        return self.tokens

    def start(self, statements, question, samples, temperature, timeout):
        self.started.append((samples, temperature, timeout))
        futures = []
        for i in range(samples):
            future = Future()
            future.set_result({"sql": self.candidates[i % len(self.candidates)]})
            futures.append(future)
        return futures

    def decode(self, result):
        return result["sql"]

    def stats(self):
        return {"active": 0, "waiting": 0}


def make_client(service=None, **settings):
    config = Settings(origins=["http://site.test"], **settings)
    return TestClient(create_app(service or FakeService(), config))


def events(response):
    parsed, name = [], None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            parsed.append((name, json.loads(line[6:])))
    return parsed


def test_health():
    assert make_client().get("/health").json() == {"status": "ok"}


def test_generate_streams_every_candidate_then_done():
    body = {"tables": [TABLE], "question": "How many singers?", "samples": 3}
    response = make_client().post("/generate", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    got = events(response)
    names = [name for name, _ in got]
    assert names[0] == "start" and names[-1] == "done"
    assert names.count("candidate") == 3
    assert got[-1][1]["candidates"] == ["SELECT 1", "SELECT 2", "SELECT 3"]


def test_settings_reach_the_engine():
    service = FakeService()
    client = make_client(service, default_samples=2, temperature=0.5, request_timeout=12.0)
    client.post("/generate", json={"tables": [TABLE], "question": "q"})
    assert service.started == [(2, 0.5, 12.0)]


@pytest.mark.parametrize(
    "body",
    [
        {"tables": [TABLE], "question": "x" * 301},
        {"tables": [TABLE], "question": "q", "samples": 17},
        {"tables": [TABLE], "question": "q", "samples": 0},
        {"tables": ["DROP TABLE singer"], "question": "q"},
        {"tables": [], "question": "q"},
        {"tables": [TABLE] * 41, "question": "q"},
        {"tables": ["CREATE TABLE t (" + "a INTEGER, " * 300 + "b INTEGER)"], "question": "q"},
    ],
)
def test_invalid_requests_are_rejected(body):
    assert make_client().post("/generate", json=body).status_code == 422


def test_a_prompt_that_cannot_fit_is_refused():
    client = make_client(FakeService(prompt_tokens=5000))
    response = client.post("/generate", json={"tables": [TABLE], "question": "q"})
    assert response.status_code == 413


def test_rate_limit_returns_429_after_the_allowed_requests():
    client = make_client(rate_per_minute=2)
    body = {"tables": [TABLE], "question": "q", "samples": 1}
    codes = [client.post("/generate", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(2, window=60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")


def test_cors_allows_only_the_configured_origin():
    client = make_client()
    ask = {"Access-Control-Request-Method": "POST"}
    good = client.options("/generate", headers={"Origin": "http://site.test", **ask})
    assert good.headers.get("access-control-allow-origin") == "http://site.test"
    bad = client.options("/generate", headers={"Origin": "http://evil.test", **ask})
    assert "access-control-allow-origin" not in bad.headers


def test_feedback_is_appended_to_a_file(tmp_path):
    path = tmp_path / "feedback.jsonl"
    client = make_client(feedback_path=str(path))
    ok = client.post("/feedback", json={"question": "q", "sql": "SELECT 1", "verdict": "up"})
    assert ok.status_code == 200
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["verdict"] == "up"
    bad = client.post("/feedback", json={"question": "q", "sql": "SELECT 1", "verdict": "maybe"})
    assert bad.status_code == 422


def test_metrics_count_requests_and_rejections():
    client = make_client()
    client.post("/generate", json={"tables": [TABLE], "question": "q", "samples": 2})
    client.post("/generate", json={"tables": ["DROP TABLE x"], "question": "q"})
    data = client.get("/metrics").json()
    assert data["counters"]["requests"] == 1
    assert data["counters"]["rejected_not_create"] == 1
    assert data["engine"] == {"active": 0, "waiting": 0}
