"""Export the fine-tuned classifier to ONNX and quantize its weights to int8.

The quantized model runs on ONNX Runtime without PyTorch, which shrinks the serving image and
memory footprint enough for small free-tier hosts and speeds up CPU inference.

Usage:
    python -m training.export_onnx --model models/injection-deberta --out models/injection-onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.rails.injection import ONNX_FILE


def export(model_dir: Path, out: Path, opset: int = 17, quantize: bool = True) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sample = tokenizer(["export sample"], return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic["logits"] = {0: "batch"}

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.inner(**dict(zip(names, args, strict=True))).logits

    fp32 = out / "model.fp32.onnx"
    torch.onnx.export(
        Wrapper(model),
        tuple(sample[n] for n in names),
        str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_axes=dynamic,
        opset_version=opset,
        dynamo=False,
    )
    if quantize:
        quantize_dynamic(str(fp32), str(out / ONNX_FILE), weight_type=QuantType.QInt8)
        fp32.unlink()
    else:
        fp32.replace(out / ONNX_FILE)
    tokenizer.save_pretrained(out)
    shutil.copy(model_dir / "config.json", out / "config.json")
    return out / ONNX_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/injection-deberta"))
    parser.add_argument("--out", type=Path, default=Path("models/injection-onnx"))
    parser.add_argument("--no-quantize", action="store_true", help="Keep fp32 weights")
    args = parser.parse_args()
    path = export(args.model, args.out, quantize=not args.no_quantize)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
