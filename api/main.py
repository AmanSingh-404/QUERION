import argparse
import os

import torch
import uvicorn

from api.app import Settings, create_app
from api.service import EngineService
from engine.prefix import PrefixServer
from infer.constrain import token_texts
from infer.quantize import quantize_int8
from scripts.ask import load
from tokenizer.special_tokens import EOS_ID


def build(args):
    model, tok = load(args.weights, args.device)
    if args.device == "cpu" and args.int8:
        model = quantize_int8(model)
    server = PrefixServer(model, EOS_ID, args.device, num_blocks=args.blocks, max_seqs=args.slots)
    service = EngineService(server, tok, token_texts(tok), EOS_ID)
    samples = args.samples or (8 if args.device == "cuda" else 4)
    settings = Settings(origins=args.origins.split(","), default_samples=samples)
    return create_app(service, settings)


def main():
    env = os.environ.get
    p = argparse.ArgumentParser(description="Serve QUERION over HTTP.")
    p.add_argument("--weights", default=env("QUERION_WEIGHTS", "SinghBro/querion-100m-sql"))
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=env("QUERION_DEVICE", default_device))
    p.add_argument("--int8", action="store_true", default=env("QUERION_INT8") == "1")
    p.add_argument("--samples", type=int, default=int(env("QUERION_SAMPLES", "0")))
    p.add_argument("--blocks", type=int, default=int(env("QUERION_BLOCKS", "1024")))
    p.add_argument("--slots", type=int, default=int(env("QUERION_SLOTS", "32")))
    p.add_argument("--origins", default=env("QUERION_ORIGINS", "http://localhost:3000"))
    p.add_argument("--port", type=int, default=int(env("PORT", "7860")))
    args = p.parse_args()
    uvicorn.run(build(args), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
