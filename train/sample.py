import torch


@torch.no_grad()
def generate(model, ids, max_new, eos_id, device):
    """Greedy decoding without a KV cache (slow but simple; used only for progress samples)."""
    model.eval()
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits, _ = model(x[:, -model.cfg.max_seq_len :])
        nxt = int(logits[0, -1].argmax())
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == eos_id:
            break
    return x[0].tolist()
