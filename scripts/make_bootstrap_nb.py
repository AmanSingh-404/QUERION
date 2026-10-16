import json

CODE = """from google.colab import drive
drive.mount('/content/drive')

import os
import subprocess

BASE = '/content/drive/MyDrive/querion'
REPO = '/content/QUERION'
URL = 'https://github.com/AmanSingh-404/QUERION.git'

if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', URL, REPO], check=True)
else:
    subprocess.run(['git', '-C', REPO, 'pull'], check=True)
os.chdir(REPO)

subprocess.run(['pip', 'install', '-q', 'tokenizers', 'wandb', 'tqdm', 'pytest'], check=True)

import torch

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', gpu)
print('drive base:', os.listdir(BASE))
"""

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": CODE.splitlines(keepends=True),
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("notebooks/00_bootstrap.ipynb", "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
