"""Sanity-check an exported toxicity model on a few clear-cut replies before serving it.

Off-the-shelf models are not retrained here, so this is a smoke test of the export (labels,
sigmoid scoring, tokenizer) rather than a benchmark. Fails if any example lands on the wrong
side of the threshold.

Usage:
    python -m training.check_toxicity models/toxicity-onnx --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.rails.toxicity import ToxicityRail

CLEAN = [
    "The capital of France is Paris.",
    "Here is a recipe for banana bread that takes about an hour.",
    "I can't help with that, but here is a safer alternative.",
    "Your code fails because the list index starts at zero.",
    "Thanks for asking! The meeting moved to Thursday at 3 pm.",
    "To kill a Python process, run `kill` with its process id.",
    "The villain in the novel threatens the town, but the hero stops him.",
    "That's a fair criticism of the plan; let's revise the budget.",
]
TOXIC = [
    "You are a worthless idiot and nobody wants you here.",
    "Shut up, you stupid moron.",
    "I will find where you live and hurt you.",
    "People like you are disgusting garbage.",
    "What a pathetic loser, go away and die.",
    "You're too dumb to understand anything, you useless fool.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, help="Write results as JSON here")
    args = parser.parse_args()

    rail = ToxicityRail(str(args.model), threshold=args.threshold)
    rail.warm_up()
    rows, latencies = [], []
    for text, expected in [(t, "allow") for t in CLEAN] + [(t, "block") for t in TOXIC]:
        started = time.perf_counter()
        verdict = rail.check(text)
        latencies.append((time.perf_counter() - started) * 1000)
        rows.append(
            {
                "text": text,
                "expected": expected,
                "got": verdict.action.value,
                "score": round(verdict.score, 4),
            }
        )
    wrong = [r for r in rows if r["got"] != r["expected"]]
    for r in rows:
        mark = "ok " if r["got"] == r["expected"] else "BAD"
        print(f"{mark} {r['expected']:5} {r['score']:.3f}  {r['text']}")
    p50 = statistics.median(latencies)
    print(f"{len(rows) - len(wrong)}/{len(rows)} correct, median {p50:.1f} ms per reply")
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "median_ms": round(p50, 2)}, indent=2))
    raise SystemExit(1 if wrong else 0)


if __name__ == "__main__":
    main()
