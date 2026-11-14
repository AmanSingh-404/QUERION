from data.prompt import format_prompt
from engine.bench import Request
from infer.constrain import SqlConstraint


def clean(text):
    return text.strip().rstrip(";")


class EngineService:
    """Turns table definitions and a question into constrained candidate queries."""

    def __init__(self, server, tok, texts, eos_id, max_new=200):
        self.server, self.tok, self.texts, self.eos_id = server, tok, texts, eos_id
        self.max_new = max_new

    def prompt_tokens(self, statements, question):
        return len(self.tok.encode(format_prompt(statements, question)).ids)

    def start(self, statements, question, samples, temperature, timeout):
        """Submit `samples` requests (the first greedy) and return their futures."""
        ids = self.tok.encode(format_prompt(statements, question)).ids
        schema_text = "\n".join(statements)
        futures = []
        for i in range(samples):
            request = Request(
                ids=ids,
                max_new=self.max_new,
                timeout=timeout,
                constraint=SqlConstraint(schema_text, self.texts, self.eos_id),
                temperature=0.0 if i == 0 else temperature,
            )
            futures.append(self.server.submit(request))
        return futures

    def decode(self, result):
        out = result["ids"]
        if out and out[-1] == self.eos_id:
            out = out[:-1]
        return clean(self.tok.decode(out, skip_special_tokens=False))

    def stats(self):
        s = self.server
        return {
            "active": len(s.active),
            "waiting": len(s.waiting),
            "prefix_hit_rate": round(getattr(s, "hit_rate", 0.0), 3),
            "preemptions": s.preemptions,
            "timeouts": s.timeouts,
            "kv_peak_used_gb": round(s.kv_peak_gb, 3),
        }
