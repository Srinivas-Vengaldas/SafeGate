"""Compare injection detectors on the same test sets: the rules rail alone, SafeGate's classifier
and public baseline classifiers (for example ProtectAI's DeBERTa prompt-injection model).

Every model is scored the way the gateway serves it (same max length and window cap), on:
- the in-distribution test split: precision, recall, F1, false-positive rate
- the held-out source no SafeGate model saw in training: recall
- the tricky-benign set: false-positive rate
- CPU latency per prompt at batch size 1

Each model is given as LABEL=PATH@THRESHOLD, where THRESHOLD is a number or "val" (best F1 on
the validation split, as SafeGate's own threshold is chosen).

Usage:
    python -m training.compare --data data --out reports/compare \
        --model "SafeGate (ONNX int8)=models/injection-onnx@0.975" \
        --model "ProtectAI v2=models/protectai-onnx@0.5"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.rails.injection import InjectionClassifier
from training.evaluate import (
    best_f1_threshold,
    binary_metrics,
    evaluate_sets,
    latency_ms,
    read_jsonl,
    rules_baseline,
)


def parse_spec(spec: str) -> tuple[str, str, str]:
    label, _, rest = spec.partition("=")
    path, _, threshold = rest.rpartition("@")
    if not (label and path and threshold):
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH@THRESHOLD, got {spec!r}")
    return label, path, threshold


def size_mb(path: str) -> float | None:
    root = Path(path)
    if not root.is_dir():
        return None
    return round(sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**20, 1)


def to_markdown(report: dict) -> str:
    lines = [
        f"Held-out source: `{report['holdout_source']}`. Test split: {report['sizes']['test']} "
        f"prompts; held-out: {report['sizes']['holdout']} attacks; tricky-benign: "
        f"{report['sizes']['tricky_benign']} safe prompts. Every model is scored as served "
        f"(max {report['max_length']} tokens per window, at most {report['max_windows']} "
        "windows per prompt), on the same CPU.",
        "",
        "| Detector | Threshold | Test precision | Test recall | Test F1 | Test FPR "
        "| Held-out recall | Tricky-benign FPR | Latency p50 / p95 | Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["detectors"]:
        m = row["metrics"]
        lat = row.get("latency_ms")
        latency = f"{lat['p50']} / {lat['p95']} ms" if lat else "<1 ms"
        size = f"{row['size_mb']:.0f} MB" if row.get("size_mb") else "–"
        threshold = f"{row['threshold']:.3f}" if row.get("threshold") is not None else "–"
        lines.append(
            f"| {row['label']} | {threshold} | {m['test']['precision']:.1%} "
            f"| {m['test']['recall']:.1%} | {m['test']['f1']:.1%} | {m['test']['fpr']:.1%} "
            f"| {m['holdout']['recall']:.1%} | {m['tricky_benign']['fpr']:.1%} | {latency} "
            f"| {size} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=parse_spec, action="append", default=[], dest="models")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--tricky", type=Path, default=Path("eval/tricky_benign.jsonl"))
    parser.add_argument("--policy", type=Path, default=Path("policies/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/compare"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sets = {
        "test": read_jsonl(args.data / "test.jsonl"),
        "holdout": read_jsonl(args.data / "holdout.jsonl"),
        "tricky_benign": read_jsonl(args.tricky, default_label=0),
    }
    val_texts, val_labels = read_jsonl(args.data / "val.jsonl")

    rules = rules_baseline(args.policy)
    detectors = [
        {
            "label": "Rules only",
            "metrics": evaluate_sets(lambda t: int(rules.check(t).action.value == "block"), sets),
        }
    ]
    flagged = {}
    for label, path, threshold_spec in args.models:
        print(f"scoring {label} ({path})", flush=True)
        classifier = InjectionClassifier(
            path, max_length=args.max_length, max_windows=args.max_windows
        )
        if threshold_spec == "val":
            threshold = best_f1_threshold(val_labels, np.array(classifier(val_texts)))
        else:
            threshold = float(threshold_spec)
        scores = {name: np.array(classifier(texts)) for name, (texts, _) in sets.items()}
        detectors.append(
            {
                "label": label,
                "model": path,
                "backend": classifier.backend,
                "threshold": round(threshold, 4),
                "threshold_from": "validation" if threshold_spec == "val" else "fixed",
                "metrics": {
                    name: binary_metrics(labels, (scores[name] >= threshold).astype(int))
                    for name, (_, labels) in sets.items()
                },
                "latency_ms": latency_ms(classifier, sets["test"][0]),
                "size_mb": size_mb(path),
            }
        )
        flagged[label] = [
            t
            for t, s in zip(sets["tricky_benign"][0], scores["tricky_benign"], strict=True)
            if s >= threshold
        ]

    stats_path = args.data / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    report = {
        "holdout_source": stats.get("holdout_source", "unknown"),
        "sizes": {name: len(texts) for name, (texts, _) in sets.items()},
        "max_length": args.max_length,
        "max_windows": args.max_windows,
        "detectors": detectors,
        "tricky_benign_flagged": flagged,
    }
    (args.out / "compare.json").write_text(json.dumps(report, indent=2))
    (args.out / "compare.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
