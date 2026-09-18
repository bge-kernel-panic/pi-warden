#!/usr/bin/env python3
"""Export a fine-tuned Laya checkpoint to int8 ONNX for pi-warden (CPU inference).

Pipeline: torch checkpoint -> fp32 ONNX (throwaway intermediate) -> int8 ONNX.

int8 dynamic quantization is the right target for CPU: ~2-4x faster and ~4x
smaller than fp32, using integer SIMD. It is exactly what pi-warden loads
(~/.pi/agents/laya/laya_int8.onnx). The fp32 file only exists as the quantizer's
input and is deleted afterwards (you said don't bother keeping/testing it).

The export reproduces the SAME I/O contract as the shipped model, so it is a
drop-in replacement and src/laya.ts needs no changes:
  inputs : input_ids (b,seq) i64, attention_mask (b,seq) i64,
           marker_pos (b,2) i64, marker_mask (b,2) bool, qtype (b,) i64
  output : logits (b,2)
  dynamic axes: batch + seq_length.  markers stay FIXED at 2 (pi-warden does
  per-option fan-out in src/laya.ts, so the exported head is 2-marker, like the
  current file — which empirically rejects 3 markers).

Two spots marked `# ALIGN` must be matched to how train_laya_rocm.py loads the
model and what its forward returns. See 05-export-onnx.md step 2 for the inspect
commands. CPU is fine for export; run it in the ROCm container or anywhere with
torch + onnx + onnxruntime.
"""
import argparse
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="training output dir (model.safetensors + encoder/ + tokenizer/)")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/.pi/agents/laya"), help="where laya_int8.onnx goes")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        from safetensors.torch import load_file
        from laya.common import build_model  # same loader the training notebook uses
    except ImportError as e:
        print(f"ERROR: missing dep ({e}). Install: pip install laya safetensors onnx onnxruntime", file=sys.stderr)
        return 1

    # ---- ALIGN (A): load cfg + model EXACTLY as train_laya_rocm.py does. --------
    # Copy the cfg construction and the two model-loading lines from the training
    # script. They look like:
    #     model_dir = args.checkpoint
    #     model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    #     model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    # Replace the next line with those.
    model = _load_checkpoint(args.checkpoint, build_model, load_file)  # <- see helper below / paste inline
    model.eval().to("cpu").float()

    # ---- ALIGN (B): the model's forward signature + return type. ----------------
    # Print it first (05-export-onnx.md step 2). If forward returns (logits, act),
    # the wrapper below already takes [0]. If the kwarg names differ, fix them here.
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                marker_pos=marker_pos,
                marker_mask=marker_mask,
                qtype=qtype,
            )
            return out[0] if isinstance(out, (tuple, list)) else out

    wrapped = Wrapper(model).eval()

    seq = 32
    dummy = (
        torch.ones(1, seq, dtype=torch.long),          # input_ids
        torch.ones(1, seq, dtype=torch.long),          # attention_mask
        torch.tensor([[3, 6]], dtype=torch.long),      # marker_pos (2 markers)
        torch.tensor([[True, True]]),                  # marker_mask (bool)
        torch.tensor([2], dtype=torch.long),           # qtype (noul)
    )

    os.makedirs(args.out_dir, exist_ok=True)
    fp32 = os.path.join(args.out_dir, "laya_fp32.onnx")
    torch.onnx.export(
        wrapped,
        dummy,
        fp32,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch"},   # dim 1 omitted => markers fixed at 2
            "marker_mask": {0: "batch"},
            "qtype": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print("wrote fp32 intermediate:", fp32)

    # int8 dynamic quantization: weight-only, no calibration dataset, best for
    # transformers on CPU.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = os.path.join(args.out_dir, "laya_int8.onnx")
    if os.path.exists(int8):
        backup = int8 + ".bak"
        os.replace(int8, backup)
        print("backed up existing model ->", backup)
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    os.remove(fp32)  # intermediate only; not kept or tested per instructions
    print("wrote int8 model:", int8)

    # Smoke test: does the int8 model load and honour the I/O contract? This
    # checks the export didn't break shapes/names — NOT accuracy (that's the
    # pi-warden eval cases in step 5).
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {
        "input_ids": np.ones((1, 16), dtype=np.int64),
        "attention_mask": np.ones((1, 16), dtype=np.int64),
        "marker_pos": np.array([[3, 6]], dtype=np.int64),
        "marker_mask": np.array([[True, True]]),
        "qtype": np.array([2], dtype=np.int64),
    })[0]
    assert logits.shape[-1] == 2, f"expected 2 logits, got shape {logits.shape}"
    assert np.isfinite(logits).all(), "non-finite logits from int8 model"
    print("smoke test OK: int8 runs, logits shape", logits.shape, "->", logits.tolist())
    print("\nNext: ensure tokenizer.json is in", args.out_dir, "then recalibrate (05-export-onnx.md step 4).")
    return 0


def _load_checkpoint(checkpoint, build_model, load_file):
    """ALIGN (A) lives here if you prefer a helper. Replace the body with the
    cfg + build_model + load_state_dict lines copied from train_laya_rocm.py."""
    raise NotImplementedError(
        "ALIGN (A): paste the model-loading lines from train_laya_rocm.py into "
        "_load_checkpoint (build cfg, build_model(cfg, encoder_dir=...), "
        "load_state_dict(load_file(.../model.safetensors)))."
    )


if __name__ == "__main__":
    raise SystemExit(main())
