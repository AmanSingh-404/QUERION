import importlib.metadata as md
from pathlib import Path

import torch

PACKAGES = ["numpy", "tokenizers", "safetensors", "huggingface_hub", "fastapi", "uvicorn"]
OUT = Path(__file__).resolve().parents[1] / "docker"


def main():
    OUT.mkdir(exist_ok=True)
    lines = [f"{name}=={md.version(name)}" for name in PACKAGES]
    (OUT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    torch_version = torch.__version__.split("+")[0]
    (OUT / "torch-version.txt").write_text(torch_version + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("torch==" + torch_version)


if __name__ == "__main__":
    main()
