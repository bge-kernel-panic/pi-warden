#!/usr/bin/env python3
"""Reformat kotoba-lang/typed-decisions .jsonl files into Laya canonical JSONL.

That repo is a SEPARATE reimplementation of the typed-decision idea (not
convaiinnovations/Laya), but its data is the same shape and converts trivially.
Each source line is one state with several questions:

    {"state": "...", "questions": [
        {"kind": "choice"|"score"|"noul", "instructions": "...",
         "options": ["...", ...], "gold": <int index>}, ...]}

We flatten every question into its own canonical record (sharing the state).

IMPORTANT: these are SHORT texts (sentences), so they do NOT help the bigger-
window goal — they are general-diversity/robustness data. Use them as a minority
of the corpus. `data-code` is off-domain (code-symbol graph); skip unless you
specifically want it. See 01-prepare-datasets.md for guidance.

Run:  python reformat_kotoba.py --out data/kotoba.jsonl data-fam/train.jsonl data-multi/train.jsonl
Nothing here needs a GPU.
"""
import argparse
import json
import sys

# kotoba "kind" -> Laya qtype (identical vocabulary already).
KINDS = {"choice", "score", "noul"}


def to_records(row: dict, keep_kinds: set, min_chars: int, max_chars: int):
    state = (row.get("state") or "").strip()
    if not (min_chars <= len(state) <= max_chars):
        return
    source = row.get("source", "kotoba")
    for q in row.get("questions", []):
        kind = q.get("kind")
        options = list(q.get("options") or [])
        gold = q.get("gold")
        instructions = (q.get("instructions") or "").strip()
        if kind not in keep_kinds or not instructions or gold is None or not (0 <= gold < len(options)):
            continue
        if kind == "noul":
            # Laya's noul head is false/true; kotoba lists them as [no, yes] in that order.
            criteria = {"false": options[0], "true": options[1]} if len(options) == 2 else None
            if criteria is None:
                continue
        else:
            # choice/score: index-string keys preserve option order; gold indexes into it.
            criteria = {str(i): opt for i, opt in enumerate(options)}
        yield {
            "qtype": kind,
            "instructions": instructions,
            "criteria": criteria,
            "gold_index": gold,
            "state": state,
            "source": source,
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="kotoba .jsonl file(s) to convert")
    p.add_argument("--out", required=True, help="output canonical .jsonl path")
    p.add_argument("--kinds", default="choice,noul", help="comma list of kinds to keep (default skips score)")
    p.add_argument("--min-chars", type=int, default=0, help="drop states shorter than this")
    p.add_argument("--max-chars", type=int, default=100000, help="drop states longer than this")
    p.add_argument("--limit", type=int, default=0, help="max output records (0 = no cap)")
    args = p.parse_args()

    keep = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if not keep <= KINDS:
        print(f"ERROR: --kinds must be a subset of {sorted(KINDS)}", file=sys.stderr)
        return 1

    written = 0
    counts: dict[str, int] = {}
    with open(args.out, "w") as out:
        for path in args.inputs:
            print(f"reading {path}...", file=sys.stderr)
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    for rec in to_records(json.loads(line), keep, args.min_chars, args.max_chars):
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                        counts[rec["qtype"]] = counts.get(rec["qtype"], 0) + 1
                        if args.limit and written >= args.limit:
                            break
                    if args.limit and written >= args.limit:
                        break
            if args.limit and written >= args.limit:
                break

    print(f"wrote {written} records to {args.out}  ({counts})", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
