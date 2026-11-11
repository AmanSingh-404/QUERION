import sqlite3
import types
from concurrent.futures import Future

import pytest

from engine.pipeline import Question, SqlPipeline

EOS = 0


class StubTok:
    def __init__(self, table):
        self.table = table

    def encode(self, text):
        return types.SimpleNamespace(ids=[5, 6, 7])

    def decode(self, ids, skip_special_tokens=False):
        return self.table[tuple(ids)]


class StubServer:
    """Answers each request with a canned token sequence, like the engine would."""

    def __init__(self, answers):
        self.answers = answers
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        future = Future()
        out = self.answers[len(self.requests) - 1]
        future.set_result({"ids": out + [EOS], "tokens": len(out) + 1, "first_token_at": 1.0})
        return future


class FailingServer:
    def submit(self, request):
        future = Future()
        future.set_exception(ValueError("prompt too long"))
        return future


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2);")
    conn.commit()
    conn.close()
    return path


def test_pipeline_submits_k_constrained_requests_and_votes(db):
    table = {
        (10,): "SELECT a FROM t WHERE a > 100;",  # the greedy candidate runs but returns nothing
        (11,): "SELECT a FROM t",
        (12,): "SELECT a FROM t ORDER BY a DESC",
        (13,): "SELECT bad FROM t",
        (14,): "SELECT a FROM t",
    }
    server = StubServer([[10], [11], [12], [13], [14]])
    pipeline = SqlPipeline(server, StubTok(table), ["x"], EOS, samples=5, workers=2)
    job = Question(["CREATE TABLE t (a INTEGER)"], "list a", db)
    result = pipeline.submit(job).result(timeout=10)
    pipeline.close()
    assert result["sql"] == "SELECT a FROM t"
    assert result["tokens"] == 10
    assert [r.temperature for r in server.requests] == [0.0, 0.7, 0.7, 0.7, 0.7]
    assert len({id(r.constraint) for r in server.requests}) == 5
    assert all(r.ids == [5, 6, 7] for r in server.requests)


def test_a_failing_request_fails_the_question(db):
    pipeline = SqlPipeline(FailingServer(), StubTok({}), ["x"], EOS, samples=3, workers=1)
    job = Question(["CREATE TABLE t (a INTEGER)"], "list a", db)
    with pytest.raises(ValueError):
        pipeline.submit(job).result(timeout=10)
    pipeline.close()
