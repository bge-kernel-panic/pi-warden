# Step 05 — Export the retrained model to int8 ONNX for pi-warden

**Where:** a box with `torch` + `onnx` + `onnxruntime` (the ROCm container works;
export runs on CPU, no GPU needed).

**Result:** `~/.pi/agent/laya/laya_int8.onnx` replaced with your retrained model,
a drop-in that pi-warden loads with no code changes.

## Why int8 (short version)

pi-warden runs Laya on **CPU**, on every write/edit/bash. int8 dynamic
quantization is the correct target there: ~2–4× faster and ~4× smaller than fp32,
via integer SIMD. It's exactly what the shipped `laya_int8.onnx` is. We do **not**
export a separate fp32 model to keep or benchmark — an fp32 file exists only for a
moment as the quantizer's input, then it's deleted. The real accuracy check is the
pi-warden eval cases in step 4 below, not an fp32-vs-int8 comparison.

The Laya repo ships **no** export or quantization script, so `export_to_onnx.py`
in `scripts/` is ours. It reproduces the shipped model's exact I/O contract
(2-marker head, dynamic batch + sequence length), verified against `src/laya.ts`.

---

## 1. Install deps

```bash
pip install onnx onnxruntime
# (laya, torch, safetensors already present from step 03)
```

Check:
```bash
python -c "import onnx, onnxruntime, torch; print('onnx', onnx.__version__, '| ort', onnxruntime.__version__)"
```

---

## 2. INSPECTION sub-step (align 2 spots in the script)

`export_to_onnx.py` has two `# ALIGN` markers it can't fill blind: how the model
is loaded, and what its `forward` returns. Print both:

```bash
python - <<'PY'
import inspect
from laya.common import build_model
print("build_model signature:", inspect.signature(build_model))
PY
```

Then look at `train_laya_rocm.py` (from step 03) and find the lines that:
- build `cfg`,
- call `build_model(cfg, encoder_dir=...)`,
- `load_state_dict(load_file(".../model.safetensors"))`.

**ALIGN (A):** paste those lines into the `_load_checkpoint(...)` helper at the
bottom of `scripts/export_to_onnx.py` (replace its `raise NotImplementedError`).
Use `checkpoint` as the model dir. Return the loaded `model`.

**ALIGN (B):** check what `model(...)` returns. The wrapper already handles a
`(logits, act)` tuple by taking `[0]`. Only touch it if the forward's keyword
names differ from `input_ids / attention_mask / marker_pos / marker_mask / qtype`
(they should match — that's the trained contract).

---

## 3. Run the export

Point `--checkpoint` at the training output dir from step 04 (the one holding
`model.safetensors` + `encoder/` + `tokenizer/`):

```bash
python scripts/export_to_onnx.py --checkpoint <training_output_dir>
```

Expected tail:
```
wrote fp32 intermediate: /root/.pi/agent/laya/laya_fp32.onnx
backed up existing model -> /root/.pi/agent/laya/laya_int8.onnx.bak
wrote int8 model: /root/.pi/agent/laya/laya_int8.onnx
smoke test OK: int8 runs, logits shape (1, 2) -> [[...]]
```

`smoke test OK` means the export honoured the contract (right input names, 2
logits out, finite values). If it raises before that line, see Troubleshooting.

The old model is saved as `laya_int8.onnx.bak` — keep it until step 4 confirms the
new one is at least as good.

---

## 4. Recalibrate (the new model is NOT the old one)

The calibration temperatures in `src/laya.ts` (`CARD_TEMPERATURE`) and the shipped
thresholds in `src/config.ts` / `eval/laya-calibration.json` were fit to the
**old** checkpoint. Your retrained model has a different probability curve, so:

1. **Neutralise the old temperatures** to start, via env vars (no code change):
   ```bash
   export PI_WARDEN_LAYA_T_CHOICE=1 PI_WARDEN_LAYA_T_SCORE=1 PI_WARDEN_LAYA_T_NOUL=1
   ```
   (Re-deriving proper temperatures is optional polish; refitting thresholds below
   matters far more.)

2. **Refit thresholds** with the calibration harness against the new model, at the
   larger window you trained for:
   ```bash
   cd <pi-warden>
   CALIBRATE=1 WARDEN_JUDGE=laya PI_WARDEN_LAYA_STATE_TOKENS=2048 node scripts/rules-cases.mjs
   CALIBRATE=1 WARDEN_JUDGE=laya PI_WARDEN_LAYA_STATE_TOKENS=2048 node scripts/context-cases.mjs
   # ...and the other *-cases.mjs you care about. This rewrites eval/laya-calibration.json.
   ```

---

## 5. Confirm it's better (the real check)

Run the eval cases without `CALIBRATE`, comparing against the `.bak` model:

```bash
WARDEN_JUDGE=laya PI_WARDEN_LAYA_STATE_TOKENS=2048 node scripts/context-cases.mjs
WARDEN_JUDGE=laya PI_WARDEN_LAYA_STATE_TOKENS=2048 node scripts/rules-cases.mjs
```

Higher agreement than the shipped checkpoint at the larger token budget = the
retrain + longer window paid off. If it's worse, restore the backup:

```bash
mv ~/.pi/agent/laya/laya_int8.onnx.bak ~/.pi/agent/laya/laya_int8.onnx
```

---

## Troubleshooting

**`torch.onnx.export` fails on an op / opset:** try `--opset 18` or `--opset 16`.
ModernBERT is well-supported around opset 16–18.

**`quantize_dynamic` warns "op not supported for quantization":** harmless — it
quantizes the MatMul/Linear weights (the bulk) and leaves the rest fp32. As long
as the smoke test passes, it's fine.

**Smoke test: `expected 2 logits, got shape (1, N)`:** the wrapper returned the
wrong tensor — recheck ALIGN (B) (it's returning the full N-marker head or the
`act` tensor instead of logits).

**`marker_pos ... Got: 3 Expected: 2` when pi-warden runs it:** the export fixed
markers at 2 correctly, and this is the *expected* behaviour — src/laya.ts feeds
exactly 2 markers. If you *wanted* an N-marker model (all choice options in one
pass), that's a bigger change: list `marker_pos`/`marker_mask` dim 1 as dynamic in
the export AND rewrite the choice path in src/laya.ts. Not needed for a drop-in.

---

**Done.** The retrained model is live in pi-warden. Keep `.bak` around until you're
happy, then delete it.
