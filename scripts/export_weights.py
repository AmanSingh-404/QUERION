import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from model.config import QUERION_100M
from model.transformer import Transformer
from train.train import TOK_PATH


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "C:/querion-runs/sft-aug/step-001597.pt"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "C:/querion-release/querion-100m-sql")
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    # lm_head shares its weight with tok_emb, and safetensors rejects shared tensors
    weights = {k: v.contiguous() for k, v in state["model"].items() if k != "lm_head.weight"}
    meta = {"format": "pt", "step": str(state["step"])}
    save_file(weights, out / "model.safetensors", metadata=meta)
    (out / "config.json").write_text(json.dumps(asdict(QUERION_100M), indent=2), encoding="utf-8")
    shutil.copy(TOK_PATH, out / "tokenizer.json")

    model = Transformer(QUERION_100M)
    result = model.load_state_dict(load_file(out / "model.safetensors"), strict=False)
    if result.missing_keys != ["lm_head.weight"] or result.unexpected_keys:
        raise SystemExit(f"unexpected keys on reload: {result}")
    reloaded = model.state_dict()
    for name, tensor in state["model"].items():
        if not torch.equal(reloaded[name], tensor):
            raise SystemExit(f"tensor differs after reload: {name}")
    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"exported step {state['step']}: {model.num_params():,} parameters, {size_mb:.0f} MB")
    print("verified: every tensor identical after reload (tied output weight included)")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
