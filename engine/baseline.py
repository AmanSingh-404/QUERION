import queue
import threading
import time
from concurrent.futures import Future

import torch

from infer.engine import KVCache


class BaselineServer:
    """One request at a time with a simple KV cache: the reference for later features."""

    def __init__(self, model, eos_id, device):
        self.model, self.eos_id, self.device = model.eval(), eos_id, device
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, request):
        future = Future()
        self.queue.put((request, future))
        return future

    def close(self):
        self.queue.put(None)
        self.worker.join()

    def _loop(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            request, future = item
            try:
                future.set_result(self._run(request))
            except Exception as e:  # report the failure to the caller instead of dying
                future.set_exception(e)

    @torch.no_grad()
    def _run(self, request):
        cfg = self.model.cfg
        ids = request.ids
        total = min(len(ids) + request.max_new, cfg.max_seq_len)
        dtype = next(self.model.parameters()).dtype
        cache = KVCache(cfg, 1, total, self.device, dtype)
        logits, _ = self.model(torch.tensor([ids], device=self.device), cache=cache, pos=0)
        out, pos, first = [], len(ids), None
        for _ in range(request.max_new):
            nxt = int(logits[0, -1].argmax())
            if first is None:
                first = time.perf_counter()
            out.append(nxt)
            if nxt == self.eos_id or pos >= total:
                break
            step = torch.tensor([[nxt]], device=self.device)
            logits, _ = self.model(step, cache=cache, pos=pos)
            pos += 1
        return {"tokens": len(out), "first_token_at": first, "ids": out}
