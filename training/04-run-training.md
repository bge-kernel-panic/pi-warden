# Step 04 — Run training and check whether it helped

**Where:** inside the ROCm container shell, in `/work/training`, with
`train_laya_rocm.py` and `data/*.jsonl` ready.

---

## 1. Smoke test first (a few minutes, not the real run)

With the temporary data slice from step 03.5 still in place (a couple thousand
items), and `max_len: 2048`:

```bash
python train_laya_rocm.py
```

Watch for, in order:
- `loaded N items from data/docnli.jsonl` / `... quality.jsonl` — the wiring works.
- `total training items:` — a sane number.
- The first few training steps printing a **decreasing, non-NaN** loss.

In a **second terminal**, watch VRAM:

```bash
watch -n 2 rocm-smi
```

The 7800 XT has 16 GB. At `max_len: 2048`, batch 1, gradient checkpointing on,
expect single-digit-to-low-teens GB. If it stays well under 16 GB, you have room
to raise `max_len` to 4096 later.

**Stop conditions:**
- `loss: nan` → apply the bf16 switch from step 03.3 and re-run the smoke test.
- Out-of-memory (`HIP out of memory`) → see Troubleshooting below.
- Loss decreasing, no OOM → smoke test passed. Continue.

---

## 2. The real run

1. Remove the temporary `training_items = training_items[:N]` slice from
   `train_laya_rocm.py` (and/or re-run step 01 without `--limit` for the full
   DocNLI set).
2. Confirm the config: `EPOCHS = 3`, `max_len: 2048` (or 4096 if the smoke test
   left plenty of VRAM headroom).
3. Run it, and keep the log:

```bash
python train_laya_rocm.py 2>&1 | tee train.log
```

**Rough time estimate** (order of magnitude — the 7800 XT is ~T4-class): with
~45k items over 3 epochs at 2048 tokens, expect a few hours. At 4096, roughly
double. It's a "leave it running" job, which is fine. `tee` means you can close
the terminal and check `train.log` later (or run under `tmux`/`nohup` so an SSH
drop doesn't kill it):

```bash
tmux new -s laya   # then run the training command inside; detach with Ctrl-b d
```

---

## 3. Confirm it saved

The notebook writes a checkpoint (safetensors + tokenizer + encoder) to an output
directory — find the path it prints near the end (something like
`./laya-finetuned/` or a `snapshot`/`output_dir`). Verify:

```bash
ls -lh <that_output_dir>
# expect model.safetensors (hundreds of MB) + a tokenizer/ + encoder/ dir
```

---

## 4. Did it actually get better?

Two levels, cheapest first:

### 4a. In-notebook held-out accuracy (do this)

The notebook evaluates on a validation/test split and prints accuracy per qtype.
Compare the retrained numbers against a baseline run of the **unmodified**
notebook (same data, `max_len: 512`). If the longer-window model scores higher on
the long examples, the retrain worked. This is the honest, self-contained signal.

### 4b. Against pi-warden's guards (needs the ONNX export — later)

pi-warden's eval scripts (`scripts/*-cases.mjs`, run with `WARDEN_JUDGE=laya`)
judge the *ONNX* model, so this only applies **after** the safetensors→ONNX export
gap from step 03.6 is closed. When it is:

```bash
# on the machine with pi-warden checked out, model exported to ~/.pi/agents/laya/
cd <pi-warden>
WARDEN_JUDGE=laya PI_WARDEN_LAYA_STATE_TOKENS=2048 node scripts/context-cases.mjs
WARDEN_JUDGE=laya PI_WARDEN_LAYA_STATE_TOKENS=2048 node scripts/rules-cases.mjs
# ... then CALIBRATE=1 to refit thresholds for the new checkpoint, per the calibration harness.
```

Higher agreement than the shipped checkpoint at the larger
`PI_WARDEN_LAYA_STATE_TOKENS` = the whole exercise paid off.

---

## Troubleshooting

**`HIP out of memory`:**
- Lower `max_len` (4096 → 2048 → 1024).
- Confirm gradient checkpointing is enabled (the notebook sets
  `gradient_checkpointing_enable()` / `head_checkpointing = True` — don't remove).
- Raise `GRAD_ACCUM` and keep the per-step batch at 1 (same effective batch, less
  peak memory).
- Shrink the dataset (`--limit`) — fewer long examples per epoch.

**Training is extremely slow / GPU sits near 0% in `rocm-smi`:**
- The run may be CPU-bound on tokenization. That happens once at load; the GPU
  should be busy during the training-step loop. If it's idle *during* steps,
  confirm the model is actually on the GPU (`next(model.parameters()).device`).

**`loss: nan` from the first step:** bf16 switch (step 03.3). If it persists,
lower the learning rates (`LR_ENCODER`, `LR_HEAD`) by 2x.

---

**Done.** You have a Laya checkpoint trained at a longer window. The remaining
piece to use it in pi-warden is the safetensors→ONNX export (step 03.6) — pick
that up as a separate task once 4a shows the retrain is worth shipping.
