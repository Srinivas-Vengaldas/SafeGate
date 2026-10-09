"""Turns two garak report files (the bare LLM and the same LLM behind SafeGate) into an attack
success rate table by probe category.

Attack success rate = outputs a probe's detector flagged as a successful attack / outputs it
scored. Lower is better. Outputs garak could not score are left out of both counts.

Usage:
    python eval/garak_summary.py --bare bare.report.jsonl --safegate safegate.report.jsonl \\
        --out reports/redteam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = {
    "promptinject": "Prompt injection (PromptInject hijacks)",
    "dan": "Jailbreaks (DAN in the wild)",
    "encoding": "Encoded injection (Base64)",
    "latentinjection": "Indirect injection in documents",
    "realtoxicityprompts": "Toxic continuations (RealToxicityPrompts)",
}


def load(path: Path) -> dict[str, list[int]]:
    """{category: [hits, scored]} summed over the report's eval entries."""
    counts: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("entry_type") != "eval":
            continue
        module = entry["probe"].split(".")[0]
        passed = int(entry["passed"])
        fails = int(entry.get("fails", entry.get("total", 0) - passed))
        bucket = counts.setdefault(module, [0, 0])
        bucket[0] += fails
        bucket[1] += passed + fails
    return counts


def rate(bucket: list[int] | None) -> str:
    if not bucket or not bucket[1]:
        return "n/a"
    return f"{100 * bucket[0] / bucket[1]:.1f}% ({bucket[0]}/{bucket[1]})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bare", type=Path, required=True)
    parser.add_argument("--safegate", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", type=Path, default=Path("reports/redteam"))
    args = parser.parse_args()

    bare, guarded = load(args.bare), load(args.safegate)
    rows = [
        "| Attack category | Bare LLM | Behind SafeGate |",
        "|---|---|---|",
    ]
    summary = {}
    for module in sorted(
        set(bare) | set(guarded), key=lambda m: list(CATEGORIES).index(m) if m in CATEGORIES else 99
    ):
        label = CATEGORIES.get(module, module)
        rows.append(f"| {label} | {rate(bare.get(module))} | {rate(guarded.get(module))} |")
        summary[module] = {"bare": bare.get(module), "safegate": guarded.get(module)}
    total_bare = [sum(b[0] for b in bare.values()), sum(b[1] for b in bare.values())]
    total_guarded = [sum(b[0] for b in guarded.values()), sum(b[1] for b in guarded.values())]
    rows.append(f"| **All** | **{rate(total_bare)}** | **{rate(total_guarded)}** |")
    table = "\n".join(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "garak.json").write_text(
        json.dumps({"model": args.model, "categories": summary}, indent=2)
    )
    header = f"garak attack success rate against {args.model} (lower is better).\n\n"
    (args.out / "garak.md").write_text(header + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
