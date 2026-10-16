import shutil
import subprocess

GH = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"

LABELS = {
    "setup": "6e7781",
    "data": "0e8a16",
    "model": "1d76db",
    "training": "d93f0b",
    "eval": "c5def5",
    "engine": "5319e7",
    "web": "fbca04",
    "docs": "0075ca",
}

DONE = {f"0.{i}" for i in range(1, 9)}

DATA = """
## 0 | Setup and workflow | setup | Edit locally, push to GitHub, run in Colab
Create GitHub repo
Create folder layout
Add project config and linter hooks
Add CI workflow
Create Drive folders
Colab bootstrap notebook
W&B and Hugging Face logins
Benchmark the T4
Project board, milestones and issues
## 1 | Tokenizer | data | A custom 24k BPE tokenizer for SQL, schemas and questions
Collect tokenizer corpus
Normalize text
Define special tokens
Train 24k BPE
Write tokenizer tests
Benchmark vs GPT-2 and Llama tokenizers
Commit tokenizer and README comparison
## 2 | Data pipeline | data | Clean, deduplicated, licensed data in shards, eval sets locked
Choose sources and log licenses in DATA.md
Write download scripts
Clean text
Deduplicate
Build synthetic pair generator
Freeze prompt format
Tokenize and pack shards
Lock eval sets and run leakage check
Write data stats
## 3 | Model code | model | A correct, tested model of about 100M parameters
Config dataclass
RMSNorm, RoPE, GQA, SwiGLU and block
Assemble model and check param count
Unit tests
Overfit a tiny batch
Cross-check against a reference implementation
Profile on T4
## 4 | Training loop | training | A trainer that survives Colab disconnects and logs everything
Dataloader from shards
AdamW and cosine LR
fp16 and grad scaler
Gradient accumulation
Atomic checkpoints to Drive
Auto-resume
W&B logging
Periodic validation and samples
500-step dry run
## 5 | Ablations | training | Evidence for design choices from small experiments
Define the proxy model
GQA vs MHA
RoPE vs learned positions
Data mixes
LR sweep
Write experiment docs
Freeze final config
## 6 | Pretraining | training | Train the main model on about 2B tokens
Pre-flight checklist
Session routine
Monitor loss and spikes
Mid-run evals
Checkpoint housekeeping
Upload final weights to HF Hub
Training report
## 7 | Fine-tuning | training | Turn the base model into a text-to-SQL model, plus a draft model
SFT dataset builder
Stage 1: WikiSQL
Stage 2: Spider and synthetic
Tune hyperparameters
Train the draft model
Select best checkpoint by validation
Upload final weights
## 8 | Evaluation harness | eval | One command that produces every accuracy number
SQLite executor
Result comparator
Run all benchmarks
Error taxonomy
CI regression subset
README accuracy table
## 9 | Inference engine core | engine | Own engine with batching, paged KV cache, prefix cache
Baseline generate loop
Cached-decoding correctness test
Benchmark harness
Continuous batching
Paged KV cache
Attention over block tables
Prefix cache
Request lifecycle
Benchmark each feature
## 10 | Engine advanced | engine | int8, speculative and constrained decoding, Triton kernel
int8 quantization
Speculative decoding
Distribution-preservation test
Grammar-constrained decoding
Execute-and-retry
Triton decode-attention kernel
Final benchmark table with vLLM
## 11 | API and website | web | A live product: API, playground, landing page, status
FastAPI endpoints
Safety limits
Dockerfile and deploy to HF Spaces
Next.js scaffold
sql.js integration
Playground
Landing page
Benchmarks, how-it-works, model-card, status, docs pages
Cold-start handling
Load test
Mobile and accessibility pass
## 12 | Launch | docs | Get real users and build interview material
Polish README
Blog post
Demo video
Hugging Face model card
Launch posts
Fix top 3 issues from feedback
Update resume and LinkedIn
Interview talk track
"""


def run(args):
    out = subprocess.run([GH, *args], check=True, capture_output=True, text=True)
    return out.stdout.strip()


phases = []
for line in DATA.strip().splitlines():
    if line.startswith("## "):
        n, title, label, goal = line[3:].split(" | ")
        phases.append((int(n), title, label, goal, []))
    else:
        phases[-1][4].append(line)

for name, color in LABELS.items():
    run(["label", "create", name, "--color", color, "--force"])

for n, title, label, goal, steps in phases:
    ms = f"Phase {n}: {title}"
    run(
        ["api", "repos/{owner}/{repo}/milestones", "-f", f"title={ms}", "-f", f"description={goal}"]
    )
    lines = []
    for i, s in enumerate(steps, 1):
        sid = f"{n}.{i}"
        mark = "x" if sid in DONE else " "
        lines.append(f"- [{mark}] {sid} {s}")
    body = f"**Goal:** {goal}\n\n" + "\n".join(lines)
    url = run(
        ["issue", "create", "--title", ms, "--body", body, "--label", label, "--milestone", ms]
    )
    print(url)
