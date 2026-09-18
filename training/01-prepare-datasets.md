# Step 01 — Download and reformat the datasets

**Where:** anywhere with Python + internet (CPU only, no GPU needed). You can do
this on the Mac and copy the `data/` folder over, or do it on the home box.

**Result:** two files, `training/data/docnli.jsonl` and
`training/data/quality.jsonl`, in the canonical format described in the README.
(typed-decisions is downloaded automatically by the training notebook later — it
needs no reformatting.)

---

## 1. Install the one dependency

```bash
pip install "datasets>=3.0.0"
```

Check:

```bash
python -c "import datasets; print(datasets.__version__)"
```

Expect a version number `>= 3.0.0`. If this errors, fix it before continuing.

---

## 2. Make the output folder

```bash
cd training
mkdir -p data
```

---

## 3. Reformat DocNLI (the noul dataset)

```bash
python scripts/reformat_docnli.py --out data/docnli.jsonl
```

This streams `saattrupdan/doc-nli` (1.44M rows, do not download it all — the
script streams and stops early), keeps long premises, and balances true/false.
Defaults: premises 1,200–24,000 chars, capped at 40,000 examples total.

Expect a final line like:

```
wrote 40000 examples to data/docnli.jsonl  (true=20000, false=20000)
```

If it wrote 0, the message tells you to loosen `--min-chars`. To make a smaller
quick-test set: `--limit 4000`.

---

## 4. Reformat QuALITY (the choice dataset)

```bash
python scripts/reformat_quality.py --out data/quality.jsonl
```

QuALITY is small (~4,600 rows), so this downloads fully and keeps all of it. It
first prints a line like:

```
answer index detection: min=0 max=3 -> one_based=False
```

That line matters: it confirms the script figured out whether answers are 0- or
1-based. Then:

```
wrote 4609 examples to data/quality.jsonl
```

If it crashes with an `assert ... out of range` error, the index detection was
wrong for your mirror of the dataset — report the printed min/max and stop.

---

## 5. Verify the output (do not skip)

```bash
wc -l data/docnli.jsonl data/quality.jsonl
python - <<'PY'
import json
for path in ("data/docnli.jsonl", "data/quality.jsonl"):
    with open(path) as fh:
        rec = json.loads(fh.readline())
    print("\n===", path, "===")
    print("qtype     :", rec["qtype"])
    print("gold_index:", rec["gold_index"])
    print("options   :", list(rec["criteria"].keys()))
    print("instructions[:120]:", rec["instructions"][:120])
    print("state chars:", len(rec["state"]))
    # gold_index must point at a real option
    assert 0 <= rec["gold_index"] < len(rec["criteria"]), "gold_index out of range!"
print("\nOK: both files parse and gold_index is in range.")
PY
```

You should see, for docnli: `qtype: noul`, `options: ['false', 'true']`; for
quality: `qtype: choice`, `options: ['A', 'B', 'C', 'D']`, and a large
`state chars` (thousands). The final line must be `OK: ...`.

**When both files exist and that check prints OK, step 01 is done.** Move to
`02-setup-rocm.md`.
