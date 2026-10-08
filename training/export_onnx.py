"""Export the fine-tuned classifier to ONNX and quantize its weights to int8.

The quantized model runs on ONNX Runtime without PyTorch, which shrinks the serving image and
memory footprint enough for small free-tier hosts and speeds up CPU inference. The weights are
stored in a separate, page-aligned `model.onnx.data` file, which ONNX Runtime memory-maps instead
of copying onto the heap: they load lazily, are shared between processes, and the kernel can
drop them under memory pressure instead of killing the service.

Usage:
    python -m training.export_onnx --model models/injection-deberta --out models/injection-onnx
    # Move the weights of an already exported model into model.onnx.data:
    python -m training.export_onnx --externalize models/injection-onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.rails.injection import ONNX_FILE

DATA_FILE = ONNX_FILE + ".data"
# ONNX Runtime memory-maps external weights only when each one starts at an offset aligned to
# the platform's allocation granularity (4 KiB on Linux, 64 KiB on Windows).
ALIGNMENT = 64 * 1024
# Tensors smaller than this stay inline; mapping them would waste most of a page.
MIN_EXTERNAL_BYTES = 1024


def externalize(model_dir: Path) -> Path:
    """Move the model's weights into a page-aligned model.onnx.data file next to it."""
    import onnx
    from onnx import TensorProto

    path = model_dir / ONNX_FILE
    model = onnx.load(str(path))
    with open(model_dir / DATA_FILE, "wb") as data:
        for tensor in model.graph.initializer:
            if not tensor.HasField("raw_data") or len(tensor.raw_data) < MIN_EXTERNAL_BYTES:
                continue
            data.write(b"\0" * (-data.tell() % ALIGNMENT))
            offset, length = data.tell(), len(tensor.raw_data)
            data.write(tensor.raw_data)
            tensor.ClearField("raw_data")
            tensor.data_location = TensorProto.EXTERNAL
            del tensor.external_data[:]
            for key, value in (("location", DATA_FILE), ("offset", offset), ("length", length)):
                entry = tensor.external_data.add()
                entry.key, entry.value = key, str(value)
    onnx.save(model, str(path))
    return path


def export(model_dir: Path, out: Path, opset: int = 17, quantize: bool = True) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    externalize(out)
    tokenizer.save_pretrained(out)
    shutil.copy(model_dir / "config.json", out / "config.json")
    return out / ONNX_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/injection-deberta"))
    parser.add_argument("--out", type=Path, default=Path("models/injection-onnx"))
    parser.add_argument("--no-quantize", action="store_true", help="Keep fp32 weights")
    parser.add_argument(
        "--externalize", type=Path, metavar="DIR", help="Only externalize an exported model"
    )
    args = parser.parse_args()
    if args.externalize:
        path = externalize(args.externalize)
    else:
        path = export(args.model, args.out, quantize=not args.no_quantize)
    size = (path.stat().st_size + (path.parent / DATA_FILE).stat().st_size) / 1e6
    print(f"wrote {path} and {DATA_FILE} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
