"""Evaluate the injection classifier against the rules-only baseline.

Reports, for each model:
- in-distribution test split: precision, recall, F1, false-positive rate, PR AUC
- held-out source the model never saw: recall (it contains only attacks)
- tricky-benign set: false-positive rate (prompts that look dangerous but are safe)
- CPU latency per prompt at batch size 1

The operating threshold is chosen on the validation split (best F1), never on the test sets.

Usage:
    python -m training.evaluate --model models/injection-deberta --data data --out reports
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

from app.rails.injection import InjectionClassifier
from app.rails.rules import RulesRail
from training.build_dataset import read_jsonl as read_jsonl_rows


def read_jsonl(path: Path, default_label: int | None = None) -> tuple[list[str], np.ndarray]:
    rows = read_jsonl_rows(path)
    labels = [r.get("label", default_label) for r in rows]
    return [r["text"] for r in rows], np.array(labels, dtype=int)


def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | int]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def rules_baseline(policy_path: Path) -> RulesRail:
    cfg = (yaml.safe_load(policy_path.read_text()) or {}).get("input_rails", {}).get("rules", {})
    return RulesRail(**cfg)


def evaluate_sets(predict, sets: dict[str, tuple[list[str], np.ndarray]]) -> dict[str, dict]:
    out = {}
    for name, (texts, labels) in sets.items():
        out[name] = binary_metrics(labels, np.array([predict(t) for t in texts], dtype=int))
    return out


def latency_ms(scorer, texts: list[str], n: int = 200) -> dict[str, float]:
    sample = (texts * (n // max(len(texts), 1) + 1))[:n]
    scorer(sample[:5])  # warm up
    timings = []
    for text in sample:
        started = time.perf_counter()
        scorer([text])
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "p50": round(statistics.median(timings), 2),
        "p95": round(timings[int(0.95 * (len(timings) - 1))], 2),
    }


def plot_pr(labels: np.ndarray, scores: np.ndarray, threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot(recall, precision, color="#2f5bea", linewidth=2)
    ax.scatter([recall[idx]], [precision[idx]], color="#c2272d", zorder=3)
    ax.annotate(
        f"threshold {threshold:.2f}",
        (recall[idx], precision[idx]),
        textcoords="offset points",
        xytext=(-90, -18),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Injection classifier, test split")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)


def to_markdown(report: dict) -> str:
    rows = [
        ("Test precision", "test", "precision"),
        ("Test recall", "test", "recall"),
        ("Test F1", "test", "f1"),
        ("Test false-positive rate", "test", "fpr"),
        ("Held-out source recall", "holdout", "recall"),
        ("Tricky-benign false-positive rate", "tricky_benign", "fpr"),
    ]
    lines = [
        f"Held-out source: `{report['holdout_source']}`. Threshold {report['threshold']:.3f} "
        "chosen on the validation split (best F1).",
        "",
        "| Metric | Rules only | Classifier |",
        "| --- | ---: | ---: |",
    ]
    for label, split, key in rows:
        base = report["rules_baseline"][split][key]
        model = report["classifier"][split][key]
        lines.append(f"| {label} | {base:.1%} | {model:.1%} |")
    lines.append(f"| Test PR AUC | – | {report['classifier']['test_pr_auc']:.3f} |")
    lat = report["classifier_latency_ms"]
    lines.append(f"| CPU latency per prompt (p50 / p95) | <1 ms | {lat['p50']} / {lat['p95']} ms |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/injection-deberta")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--tricky", type=Path, default=Path("eval/tricky_benign.jsonl"))
    parser.add_argument("--policy", type=Path, default=Path("policies/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sets = {
        "test": read_jsonl(args.data / "test.jsonl"),
        "holdout": read_jsonl(args.data / "holdout.jsonl"),
        "tricky_benign": read_jsonl(args.tricky, default_label=0),
    }
    val_texts, val_labels = read_jsonl(args.data / "val.jsonl")

    classifier = InjectionClassifier(args.model, max_length=args.max_length)
    threshold = best_f1_threshold(val_labels, np.array(classifier(val_texts)))
    scores = {name: np.array(classifier(texts)) for name, (texts, _) in sets.items()}

    classifier_report = {
        name: binary_metrics(labels, (scores[name] >= threshold).astype(int))
        for name, (_, labels) in sets.items()
    }
    classifier_report["test_pr_auc"] = round(
        float(average_precision_score(sets["test"][1], scores["test"])), 4
    )
    rules = rules_baseline(args.policy)
    baseline = evaluate_sets(lambda t: int(rules.check(t).action.value == "block"), sets)

    stats_path = args.data / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    report = {
        "model": args.model,
        "threshold": round(threshold, 4),
        "holdout_source": stats.get("holdout_source", "unknown"),
        "sizes": {name: len(texts) for name, (texts, _) in sets.items()},
        "classifier": classifier_report,
        "rules_baseline": baseline,
        "classifier_latency_ms": latency_ms(classifier, sets["test"][0]),
        "tricky_benign_flagged": [
            t
            for t, s in zip(sets["tricky_benign"][0], scores["tricky_benign"], strict=True)
            if s >= threshold
        ],
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    (args.out / "metrics.md").write_text(to_markdown(report))
    plot_pr(sets["test"][1], scores["test"], threshold, args.out / "pr_curve.png")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
