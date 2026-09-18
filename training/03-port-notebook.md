# Step 03 — Port Laya's training notebook to ROCm + our datasets

**Where:** inside the ROCm container shell from step 02, in `/work/training`.

**What you're changing:** Laya ships a Colab fine-tuning notebook built for an
NVIDIA T4. We (a) run it on ROCm, (b) add our two JSONL datasets alongside the
built-in typed-decisions data, and (c) raise the token window from 512.

Because ROCm PyTorch uses the CUDA API names, there is **almost no device-code to
change**. The real edits are: install deps, add a data loader, bump `max_len`.

---

## 1. Get the notebook and convert it to a plain script

Editing a `.py` file is far easier than editing notebook JSON. Convert it:

```bash
pip install "laya>=0.1.4" "transformers>=4.48.0" "safetensors>=0.4.0" \
            "datasets>=3.0.0" huggingface_hub accelerate jupyter nbconvert

# Fetch the notebook from the Laya repo (main branch):
curl -L -o laya_finetune_colab.ipynb \
  https://raw.githubusercontent.com/NandhaKishorM/laya/main/notebooks/laya_finetune_colab.ipynb

jupyter nbconvert --to script laya_finetune_colab.ipynb
# -> produces laya_finetune_colab.py

cp laya_finetune_colab.py train_laya_rocm.py
```

You will edit `train_laya_rocm.py`. Keep the original `.py` as reference.

Check the file exists and is Python:

```bash
head -40 train_laya_rocm.py
```

---

## 2. INSPECTION sub-step (do this before editing — it de-risks everything)

We need to see two things the plan could not see when it was written: the exact
signature of `build_sequence`, and how the notebook turns one example into a
training item. Print them:

```bash
python - <<'PY'
import inspect
from laya import common
for name in ("build_sequence", "build_model"):
    fn = getattr(common, name, None)
    print("=" * 60, name)
    print(inspect.signature(fn) if fn else "NOT FOUND in laya.common")
print("=" * 60, "build_sequence source")
print(inspect.getsource(common.build_sequence))
PY
```

Then open `train_laya_rocm.py` and find the block that builds `training_items`
from the typed-decisions rows — it calls `build_sequence(...)` and appends a dict
like `{"ids":..., "markers":..., "qtype":..., "target":..., "label":...}`.

**Note three things** (you'll copy them in step 4):
- `A` — the exact argument order of `build_sequence(...)` and what it returns.
- `B` — how the notebook builds the `q` argument (the question dict: what keys —
  `type`? `instructions`? `criteria`?).
- `C` — how it computes `target` (a probability list) and `label` (an int) from
  the gold answer.

Our canonical JSONL was designed to match `B`: each record already has `qtype`,
`instructions`, `criteria`, and `gold_index`. If the notebook's `q` uses
different key names (e.g. `"kind"` instead of `"type"`), note the difference — you
adjust the small `q_from_record` function in step 4, nothing else.

---

## 3. The ROCm edits (small)

In `train_laya_rocm.py`:

1. **Allocator env var.** Find:
   ```python
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
   Add a HIP equivalent right after it (harmless on both stacks):
   ```python
   os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
   ```

2. **Leave the CUDA calls alone.** `assert torch.cuda.is_available()`,
   `torch.device("cuda")`, `torch.amp.GradScaler("cuda")`,
   `torch.autocast("cuda", dtype=torch.float16)` — all correct on ROCm. Do not
   change them.

3. **(Only if you hit NaN losses later)** RDNA3 fp16 autocast can be twitchy on
   some ops. If step 04 shows `loss: nan`, switch the two autocast/scaler lines
   to bf16:
   ```python
   # scaler = torch.amp.GradScaler("cuda", enabled=True)   # old
   scaler = torch.amp.GradScaler("cuda", enabled=False)     # bf16 needs no scaler
   # with torch.autocast("cuda", dtype=torch.float16):      # old
   with torch.autocast("cuda", dtype=torch.bfloat16):       # new
   ```
   Leave it on fp16 for the first run; only do this if you see NaNs.

---

## 4. Wire in our two datasets

Add this function near the top of `train_laya_rocm.py` (after `build_sequence`
is importable, before `training_items` is built). **Align the three marked spots
with what you noted in the inspection step.**

```python
import json

# Laya's numeric qtype codes (from src/laya.ts: choice=0, score=1, noul=2).
QTYPE_CODE = {"choice": 0, "score": 1, "noul": 2}

def q_from_record(rec):
    # (B) Match the key names the notebook's own q dict uses. If it uses "type"
    # and "criteria" and "instructions", this is already correct.
    return {"type": rec["qtype"], "instructions": rec["instructions"], "criteria": rec["criteria"]}

def items_from_jsonl(path, tok, cfg):
    items = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            q = q_from_record(rec)
            # (A) Match build_sequence's real argument order from the inspection step.
            seq, markers = build_sequence(tok, rec["state"], q, cfg["max_len"], cfg["head_max_len"])
            k = len(markers)
            gold = rec["gold_index"]
            if not (0 <= gold < k):
                # Option got truncated out of the window; skip rather than mislabel.
                continue
            # (C) Match how the notebook builds target/label. One-hot is the usual form.
            target = [0.0] * k
            target[gold] = 1.0
            items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPE_CODE[rec["qtype"]],
                "target": target,
                "label": gold,
            })
    print(f"loaded {len(items)} items from {path}")
    return items
```

Then find where `training_items` is finalised (after the typed-decisions items
are built) and concatenate ours:

```python
# training_items already holds the typed-decisions items here.
training_items += items_from_jsonl("data/docnli.jsonl",  tok, cfg)
training_items += items_from_jsonl("data/quality.jsonl", tok, cfg)
import random; random.shuffle(training_items)
print("total training items:", len(training_items))
```

---

## 5. Raise the context window

Find the config dict with `"max_len": 512` and `"head_max_len": 192`. Change:

```python
"max_len": 2048,      # was 512  -- start here; try 4096 once 2048 is stable
"head_max_len": 192,  # leave as-is; the question/instructions are short
```

**Start at 2048, not 4096.** Confirm it trains and fits VRAM first (step 04),
then raise to 4096 if memory allows. Going straight to 4096 risks an out-of-memory
crash before you know anything works.

Also, so a full run doesn't take all night on the first try, temporarily shrink
the data while you smoke-test (`--limit 4000` when you ran step 01, or slice
`training_items = training_items[:2000]` right after the concatenation). Remove
the slice for the real run.

---

## 6. Save + the export gap (read this)

The notebook saves a fine-tuned checkpoint as **safetensors** (`model.safetensors`
+ tokenizer + encoder). That is what step 04 produces.

**pi-warden loads an ONNX file** (`~/.pi/agents/laya/laya_int8.onnx`), not
safetensors. Converting the retrained model to the 2-marker ONNX head that
pi-warden expects is a **separate export step that this plan does not cover** —
it belongs to the Laya repo's export tooling (look for an `export`/`onnx` script
there, or ask for that as a follow-up task). For now, the win is measured
**inside the notebook** on a held-out split (step 04). Wiring the retrained model
back into pi-warden is the next project after we confirm it's actually better.

---

**When `train_laya_rocm.py` imports cleanly (`python -c "import ast;
ast.parse(open('train_laya_rocm.py').read())"` prints nothing) and the inspection
step's three spots are aligned, step 03 is done.** Move to `04-run-training.md`.
