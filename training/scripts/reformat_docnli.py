#!/usr/bin/env python3
"""Reformat DocNLI (saattrupdan/doc-nli) into Laya canonical JSONL.

DocNLI is document-level natural language inference: a long `premise`, a short
`hypothesis`, and a `label` of "entailment" / "not_entailment". That maps
straight onto Laya's `noul` (yes/no) head:

    state        <- premise        (the long context; this is what we want long)
    instructions <- hypothesis      (the claim to judge)
    gold         <- true if entailment else false

We keep only examples whose premise is long enough to be worth training a bigger
window on, and cap the very longest so one example can't dominate. We also
balance true/false so the model does not just learn "say false".

Run:  python reformat_docnli.py --out ../data/docnli.jsonl
See --help for the knobs. Nothing here needs a GPU.
"""
import argparse
import json
import sys

# A rough chars-per-token figure for filtering only. Real tokenization happens
# in the notebook; this is just to pick "long enough" premises cheaply.
CHARS_PER_TOKEN = 4


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .jsonl path")
    p.add_argument("--splits", default="train,validation", help="comma-separated HF splits to pull")
    p.add_argument("--min-chars", type=int, default=1200, help="drop premises shorter than this (~300 tokens)")
    p.add_argument("--max-chars", type=int, default=24000, help="drop premises longer than this (~6000 tokens)")
    p.add_argument("--limit", type=int, default=40000, help="max examples to keep total (0 = no cap)")
    p.add_argument("--balance", action="store_true", default=True, help="keep equal true/false counts")
    p.add_argument("--no-balance", dest="balance", action="store_false")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `pip install datasets` first (do this inside the ROCm container / venv).", file=sys.stderr)
        return 1

    kept = {"true": 0, "false": 0}
    per_label_cap = (args.limit // 2) if (args.limit and args.balance) else None

    with open(args.out, "w") as fh:
        for split in args.splits.split(","):
            split = split.strip()
            print(f"loading DocNLI split '{split}' (streaming)...", file=sys.stderr)
            ds = load_dataset("saattrupdan/doc-nli", split=split, streaming=True)
            for row in ds:
                premise = (row.get("premise") or "").strip()
                hypothesis = (row.get("hypothesis") or "").strip()
                label = (row.get("label") or "").strip().lower()
                if not premise or not hypothesis or label not in ("entailment", "not_entailment"):
                    continue
                if not (args.min_chars <= len(premise) <= args.max_chars):
                    continue
                gold = "true" if label == "entailment" else "false"
                if per_label_cap and kept[gold] >= per_label_cap:
                    continue
                if args.limit and not args.balance and sum(kept.values()) >= args.limit:
                    break

                rec = {
                    "qtype": "noul",
                    "instructions": hypothesis,
                    "criteria": {
                        "false": "the statement is not supported by the text",
                        "true": "the statement is supported by the text",
                    },
                    "gold_index": 1 if gold == "true" else 0,  # matches criteria order [false, true]
                    "state": premise,
                    "source": "docnli",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept[gold] += 1
                if args.limit and sum(kept.values()) >= args.limit:
                    break
            if args.limit and sum(kept.values()) >= args.limit:
                break

    total = sum(kept.values())
    print(f"wrote {total} examples to {args.out}  (true={kept['true']}, false={kept['false']})", file=sys.stderr)
    if total == 0:
        print("ERROR: wrote 0 examples. Loosen --min-chars/--max-chars or check the dataset loaded.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
