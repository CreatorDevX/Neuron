"""End-to-end smoke test (CPU, tiny): tokenizer -> data prep -> train -> eval -> generate.

Run:  python tests/e2e_smoke.py
Writes scratch artifacts under `_scratch/` and removes them at the end.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    out = r.stdout.strip()
    if out:
        print(out.splitlines()[-1])
    return r


REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
TMP = Path(os.environ.get("NEURON_SMOKE_DIR", REPO / "_scratch"))

# ---- 1. tiny corpus ---------------------------------------------------------
TMP.mkdir(exist_ok=True)
corpus = TMP / "tiny.jsonl"
rng = __import__("random").Random(0)
words = "the quick brown fox jumps over lazy dogs while tall trees grow endlessly beyond quiet rivers".split()
lines = []
for i in range(400):
    doc = " ".join(rng.choice(words) for _ in range(rng.randint(120, 400)))
    lines.append(json.dumps({"text": doc}))
corpus.write_text("\n".join(lines), encoding="utf-8")

spec = {
    "tokenizer": str(TMP / "tokenizer.json"),
    "val_interval": 7,
    "doc_min_chars": 50,
    "doc_max_chars": 2000,
    "doc_max_tokens": 2000,
    "shard_tokens": 2000,
    "buffer_tokens": 3000,
    "seq_len": 64,
    "sources": [
        {"id": "tiny", "kind": "jsonl", "path": str(corpus), "text_field": "text",
         "max_tokens": 400_000},
    ],
}
spec_path = TMP / "spec.yaml"
import yaml  # noqa: E402

spec_path.write_text(yaml.safe_dump(spec))

# ---- 2. tokenizer -----------------------------------------------------------
run([sys.executable, "-m", "neuron.train_tokenizer", "--spec", str(spec_path),
     "--out", str(TMP / "tok"), "--vocab-size", "512", "--sample-mb", "2"])

# rewriting spec to point at real tokenizer file
spec["tokenizer"] = str(TMP / "tok" / "tokenizer.json")
spec_path.write_text(yaml.safe_dump(spec))

# ---- 3. data prep -----------------------------------------------------------
run([sys.executable, "-m", "neuron.prepare_data", "--spec", str(spec_path),
     "--out", str(TMP / "data")])

# ---- 4. train 2 optim steps worth ------------------------------------------
meta = json.loads((TMP / "tok" / "tokenizer.meta.json").read_text())
cfg = yaml.safe_load(Path("configs/smoke-test.yaml").read_text())
cfg["model"]["vocab_size"] = meta["vocab_size"]
cfg["model"]["seq_len"] = 64
cfg["train"]["data_dir"] = str(TMP / "data")
cfg["train"]["out_dir"] = str(TMP / "runs" / "smoke")
cfg["train"]["micro_batch"] = 2
cfg["train"]["total_tokens"] = 20_000
cfg["train"]["ckpt_every_steps"] = 8
cfg["train"]["eval_batch"] = 4
cfg["train"]["eval_batches"] = 2
cfg_path = TMP / "smoke.yaml"
cfg_path.write_text(yaml.safe_dump(cfg))
run([sys.executable, "-m", "neuron.train", "--config", str(cfg_path), "--resume", "none"])

# ---- 5. resume = none reload from ckpt (auto-resume path) --------------------
run([sys.executable, "-m", "neuron.train", "--config", str(cfg_path)])  # auto-resume tiny

# ---- 6. evaluate -------------------------------------------------------------
run([sys.executable, "-m", "neuron.evaluate", "--ckpt", str(TMP / "runs" / "smoke" / "final.pt"),
     "--data-dir", str(TMP / "data"), "--tokens", "10000", "--batch", "4"])

# ---- 7. generate --------------------------------------------------------------
run([sys.executable, "-m", "neuron.generate", "--ckpt", str(TMP / "runs" / "smoke" / "final.pt"),
     "--tokenizer", str(TMP / "tok" / "tokenizer.json"),
     "--prompt", "the quick", "--max-new-tokens", "12", "--n", "1", "--temperature", "0.0"])

print("\nSMOKE TEST PASSED")
shutil.rmtree(TMP, ignore_errors=True)
