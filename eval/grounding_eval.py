"""Measures the grounding check (app/rag/grounding.py) on claims built from the sample knowledge
bases, where the right answer is known:

- supported: a sentence from a passage, cited to that passage, verbatim and as a lossy
  restatement (a quarter of its words dropped and a connective added, the way answers condense).
- wrong number: the same sentence with one number changed, cited to the same passage.
- wrong source: a sentence from a different document, cited to this passage.
- wrong passage: a sentence from another passage of the same document (same topic), cited to
  this passage. The hardest case: shared vocabulary.

Synthetic by design: it checks the failures the check is built for (made-up figures and claims
pinned on the wrong source), not paraphrases that keep the words and change the meaning.

    python -m eval.grounding_eval --out reports/grounding
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from app.rag.grounding import check, sentences
from app.rag.service import SAMPLE_SETS

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CONNECTIVES = ["In short, ", "Specifically, ", "Overall, ", "Notably, "]


def passages() -> list[tuple[str, str]]:
    out = []
    for make in SAMPLE_SETS.values():
        for document in make().documents:
            if not document.planted:
                out += [(document.name, p) for p in document.passages]
    return out


def _claims(passage: str) -> list[str]:
    # Drop the "Title: Section." prefix the sample passages carry.
    return [s for s in sentences(passage) if len(s.split()) >= 8 and not s.endswith(":")]


def restate(sentence: str, rng: random.Random) -> str:
    words = sentence.rstrip(".").split()
    keep = [w for w in words if rng.random() > 0.25 or _NUMBER.search(w)]
    return rng.choice(_CONNECTIVES) + " ".join(keep) + "."


def change_number(sentence: str, rng: random.Random) -> str | None:
    matches = list(_NUMBER.finditer(sentence))
    if not matches:
        return None
    m = rng.choice(matches)
    value = m.group().replace(",", "")
    try:
        new = str(round(float(value) * rng.choice([1.5, 2, 0.5, 3]) + 1, 1)).removesuffix(".0")
    except ValueError:
        return None
    return sentence[: m.start()] + new + sentence[m.end() :]


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = passages()
    cases = []
    for name, passage in items:
        others = [p for n, p in items if n != name]
        siblings = [p for n, p in items if n == name and p != passage]
        for sentence in _claims(passage):
            cases.append({"kind": "supported (verbatim)", "claim": sentence, "source": passage})
            restated = restate(sentence, rng)
            cases.append({"kind": "supported (restated)", "claim": restated, "source": passage})
            if changed := change_number(sentence, rng):
                cases.append({"kind": "wrong number", "claim": changed, "source": passage})
            foreign = rng.choice(_claims(rng.choice(others)) or [sentence])
            if foreign != sentence:
                cases.append({"kind": "wrong source", "claim": foreign, "source": passage})
            sibling = _claims(rng.choice(siblings)) if siblings else []
            if sibling:
                claim = rng.choice(sibling)
                cases.append({"kind": "wrong passage", "claim": claim, "source": passage})
    return cases


def evaluate(cases: list[dict]) -> dict:
    results: dict[str, list[bool]] = {}
    for case in cases:
        claims = check(f"{case['claim']} [1]", {1: case["source"]})
        flagged = any(c.status != "supported" for c in claims)
        results.setdefault(case["kind"], []).append(flagged)
    return {
        k: {"n": len(v), "flagged": sum(v), "rate": sum(v) / len(v)} for k, v in results.items()
    }


def report(results: dict) -> str:
    lines = [
        "## Grounding check",
        "",
        "| Claim | Cases | Flagged as unsupported |",
        "| --- | ---: | ---: |",
    ]
    for kind, r in results.items():
        lines.append(f"| {kind} | {r['n']} | {r['rate']:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/grounding"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    results = evaluate(build(args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    markdown = report(results)
    (args.out / "grounding.md").write_text(markdown)
    (args.out / "grounding.json").write_text(json.dumps(results, indent=2))
    print(markdown)


if __name__ == "__main__":
    main()
