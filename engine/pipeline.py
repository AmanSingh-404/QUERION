from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from data.prompt import format_prompt
from engine.bench import Request
from infer.constrain import SqlConstraint
from infer.select import choose


@dataclass
class Question:
    statements: list
    question: str
    db: object
    timeout: float | None = None


def clean(text):
    return text.strip().rstrip(";")


class SqlPipeline:
    """One question in, one answer out, with the engine doing the decoding.

    Several constrained candidates (the first greedy, the rest sampled) are decoded together
    and share their prompt blocks. The execution vote then picks the answer, in a helper
    thread so that running candidate queries never stalls the engine."""

    def __init__(
        self, server, tok, texts, eos_id, samples=8, temperature=0.7, max_new=200, workers=16
    ):
        self.server, self.tok, self.texts, self.eos_id = server, tok, texts, eos_id
        self.samples, self.temperature, self.max_new = samples, temperature, max_new
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, job):
        return self.pool.submit(self.ask, job)

    def close(self):
        self.pool.shutdown(wait=True)

    def ask(self, job):
        ids = self.tok.encode(format_prompt(job.statements, job.question)).ids
        schema_text = "\n".join(job.statements)
        futures = []
        for i in range(self.samples):
            request = Request(
                ids=ids,
                max_new=self.max_new,
                timeout=job.timeout,
                constraint=SqlConstraint(schema_text, self.texts, self.eos_id),
                temperature=0.0 if i == 0 else self.temperature,
            )
            futures.append(self.server.submit(request))
        try:
            results = [f.result() for f in futures]
        except Exception:
            for f in futures:
                f.cancel()
            raise
        candidates = []
        for res in results:
            out = res["ids"]
            if out and out[-1] == self.eos_id:
                out = out[:-1]
            candidates.append(clean(self.tok.decode(out, skip_special_tokens=False)))
        return {
            "sql": choose(job.db, candidates),
            "candidates": candidates,
            "tokens": sum(r["tokens"] for r in results),
            "first_token_at": min(r["first_token_at"] for r in results),
            "ids": [],
        }
