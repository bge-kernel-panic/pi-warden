#!/usr/bin/env python3
"""Reformat QuALITY (emozilla/quality) into Laya canonical JSONL.

QuALITY is 4-way multiple-choice reading comprehension over long articles
(~10k-35k characters each). That maps onto Laya's `choice` head:

    state        <- article        (the long context)
    instructions <- question
    criteria     <- {"A": opt0, "B": opt1, "C": opt2, "D": opt3}
    gold         <- the correct letter

The `answer` field is a 0-based index (0..3) into `options`. Some HF mirrors of
QuALITY use 1-based indices instead, so this script DETECTS which and normalises,
then asserts the result is in range. That guard is deliberate: a silent off-by-one
would train the model on wrong labels.

Run:  python reformat_quality.py --out ../data/quality.jsonl
Nothing here needs a GPU.
"""
import argparse
import json
import string
import sys

LETTERS = string.ascii_uppercase  # A, B, C, D, ...


def normalise_answer(answer: int, n_options: int, one_based: bool) -> int:
    idx = answer - 1 if one_based else answer
    assert 0 <= idx < n_options, f"answer {answer} out of range for {n_options} options (one_based={one_based})"
    return idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .jsonl path")
    p.add_argument("--splits", default="train,validation", help="comma-separated HF splits to pull")
    p.add_argument("--limit", type=int, default=0, help="max examples total (0 = all; QuALITY is small, ~4.6k)")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `pip install datasets` first (do this inside the ROCm container / venv).", file=sys.stderr)
        return 1

    # Detect 0- vs 1-based indexing from the first non-streaming split so the
    # whole run uses one consistent convention.
    probe = load_dataset("emozilla/quality", split=args.splits.split(",")[0].strip())
    answers = [r["answer"] for r in probe.select(range(min(200, len(probe))))]
    one_based = min(answers) >= 1 and max(answers) >= len(probe[0]["options"])
    print(f"answer index detection: min={min(answers)} max={max(answers)} -> one_based={one_based}", file=sys.stderr)

    written = 0
    with open(args.out, "w") as fh:
        for split in args.splits.split(","):
            split = split.strip()
            print(f"loading QuALITY split '{split}'...", file=sys.stderr)
            ds = load_dataset("emozilla/quality", split=split)
            for row in ds:
                article = (row.get("article") or "").strip()
                question = (row.get("question") or "").strip()
                options = list(row.get("options") or [])
                if not article or not question or len(options) < 2:
                    continue
                gold_index = normalise_answer(int(row["answer"]), len(options), one_based)
                criteria = {LETTERS[i]: opt for i, opt in enumerate(options)}

                rec = {
                    "qtype": "choice",
                    "instructions": question,
                    "criteria": criteria,
                    "gold_index": gold_index,  # indexes into criteria insertion order (A,B,C,D)
                    "state": article,
                    "source": "quality",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    break
            if args.limit and written >= args.limit:
                break

    print(f"wrote {written} examples to {args.out}", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
