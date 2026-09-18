# Retraining Laya for a bigger context window

**Goal:** fine-tune the Laya classifier (421M ModernBERT-large) on a longer token
window than the shipped checkpoint (~512), so pi-warden's guards can feed more
context and still get calibrated judgments. See the window investigation in the
main pi-warden history for *why* the shipped model degrades past ~256/512 tokens:
it was trained short, so longer inputs are out-of-distribution. The fix is to
**train at a longer length on long examples** — that is what this folder is for.

## Who this is written for

A small local model (e.g. Qwen-27B at Q3/Q4) executing on the home machine, with
a human watching. Every step has exact commands and a check to run afterwards.
**If a check fails, stop and fix that step before moving on.** Do not improvise
around a failed check.

## The three datasets

| Dataset | Laya qtype | Why | How it gets here |
|---|---|---|---|
| `LocalLLaMA/typed-decisions` | mixed (choice/score/noul) | the original Laya training data; keeps the model's existing skills | loaded natively by the notebook — **no reformat needed** |
| `saattrupdan/doc-nli` | noul (yes/no) | long premises + binary label = exactly pi-warden's dominant question shape, at length | `scripts/reformat_docnli.py` -> `data/docnli.jsonl` |
| `emozilla/quality` | choice (4-way) | long articles + multiple choice, keeps the choice head alive | `scripts/reformat_quality.py` -> `data/quality.jsonl` |

pi-warden is now mostly **noul**, so DocNLI is the most important addition;
QuALITY keeps the choice head from rotting. typed-decisions rides along unchanged.

## Order of operations (do these in order)

1. **`01-prepare-datasets.md`** — download + reformat DocNLI and QuALITY into
   `training/data/*.jsonl`. CPU only, do it anywhere (even the Mac).
2. **`02-setup-rocm.md`** — get PyTorch talking to the Radeon 7800 XT via ROCm
   (Docker container recommended). One-time, on the home machine.
3. **`03-port-notebook.md`** — turn Laya's Colab notebook into a ROCm training
   script, wire in the two JSONL files, and raise the window from 512.
4. **`04-run-training.md`** — start the run, watch VRAM, export the result, and
   compare it against the shipped checkpoint with pi-warden's eval scripts.

## Canonical JSONL format (what the reformat scripts emit)

Every line of `data/docnli.jsonl` and `data/quality.jsonl` is one training
example:

```json
{"qtype": "noul",   "instructions": "<claim/question>", "criteria": {"false": "...", "true": "..."}, "gold_index": 1, "state": "<long context>", "source": "docnli"}
{"qtype": "choice", "instructions": "<question>",       "criteria": {"A": "...", "B": "...", "C": "...", "D": "..."}, "gold_index": 2, "state": "<article>", "source": "quality"}
```

- `criteria` order **is** the option order. `gold_index` indexes into it.
- `instructions` is the thing to judge; `state` is the long context.
- This mirrors the question shape Laya's `build_sequence` already consumes for
  typed-decisions, so the notebook builds identical training items from it.

## The one honest gap

This plan was written without running it (no GPU on the authoring machine) and
without seeing the internals of `laya.common.build_sequence` / `build_model`.
Step 03 therefore has an **inspection sub-step**: you print those functions'
source and the notebook's own item-building cell, then align ~3 field names.
Everything else is fixed. If `build_sequence`'s argument shape differs from the
template, that inspection is where you catch and fix it.
