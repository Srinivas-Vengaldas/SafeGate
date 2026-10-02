"""Download, normalize, deduplicate and split the prompt-injection training data.

Leakage controls, in order:
1. Exact duplicates (after normalization) are collapsed; groups with conflicting labels are dropped.
2. Near-duplicates (MinHash over word 5-gram shingles, Jaccard >= --near-dup) are collapsed the
   same way, so paraphrase-level copies cannot land on both sides of a split.
3. One entire source is held out (default: Lakera/gandalf_ignore_instructions), and any held-out
   prompt that is a near-duplicate of a training-side prompt is removed from the held-out set.

Usage:
    python -m training.build_dataset --out data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

INJECTION = 1
BENIGN = 0


@dataclass
class Example:
    text: str
    label: int
    source: str


@dataclass(frozen=True)
class Source:
    name: str
    text_field: str
    label_fn: Callable[[dict], int | None]
    splits: tuple[str, ...] = ("train",)
    max_rows: int | None = None


def _label_from_int(row: dict) -> int | None:
    return int(row["label"])


def _label_from_type(row: dict) -> int | None:
    return {"jailbreak": INJECTION, "benign": BENIGN}.get(str(row["type"]).lower())


SOURCES: dict[str, Source] = {
    "deepset": Source("deepset/prompt-injections", "text", _label_from_int, ("train", "test")),
    "jackhhao": Source(
        "jackhhao/jailbreak-classification", "prompt", _label_from_type, ("train", "test")
    ),
    "gandalf": Source(
        "Lakera/gandalf_ignore_instructions",
        "text",
        lambda row: INJECTION,
        ("train", "validation", "test"),
    ),
    "dolly": Source(
        "databricks/databricks-dolly-15k", "instruction", lambda row: BENIGN, ("train",), 2500
    ),
    # Hard negatives: benign coding requests full of words like ignore, override, kill, execute.
    "codealpaca": Source(
        "sahil2801/CodeAlpaca-20k", "instruction", lambda row: BENIGN, ("train",), 2500
    ),
}


def load_source(key: str, seed: int) -> list[Example]:
    from datasets import load_dataset  # imported lazily so unit tests need no network

    source = SOURCES[key]
    dataset = load_dataset(source.name)
    rows: list[Example] = []
    for split in source.splits:
        if split not in dataset:
            continue
        columns = dataset[split].column_names
        if source.text_field not in columns:
            raise KeyError(f"{source.name}[{split}] has no {source.text_field!r}; has {columns}")
        for row in dataset[split]:
            text = row[source.text_field]
            label = source.label_fn(row)
            if isinstance(text, str) and text.strip() and label is not None:
                rows.append(Example(text.strip(), label, key))
    if source.max_rows and len(rows) > source.max_rows:
        rows = random.Random(seed).sample(rows, source.max_rows)
    return rows


# --- normalization and deduplication -------------------------------------------------------

_NON_WORD = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def shingles(text: str, n: int = 5) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class MinHashLSH:
    """Small MinHash + banding LSH for near-duplicate detection, no extra dependencies."""

    _PRIME = (1 << 61) - 1

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 0) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = random.Random(seed)
        self.params = [
            (rng.randrange(1, self._PRIME), rng.randrange(0, self._PRIME)) for _ in range(num_perm)
        ]
        self.bands = bands
        self.rows = num_perm // bands

    def signature(self, items: set[str]) -> tuple[int, ...]:
        hashes = [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest()) for s in items
        ]
        return tuple(min((a * h + b) % self._PRIME for h in hashes) for a, b in self.params)

    def band_keys(self, sig: tuple[int, ...]) -> Iterable[tuple[int, tuple[int, ...]]]:
        for band in range(self.bands):
            yield band, sig[band * self.rows : (band + 1) * self.rows]


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def dedupe(
    examples: list[Example], threshold: float = 0.8, seed: int = 0
) -> tuple[list[Example], dict]:
    """Collapse exact and near-duplicate groups. Groups with conflicting labels are dropped."""
    # Union-find over example indices.
    parent = list(range(len(examples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_norm: dict[str, int] = {}
    for i, ex in enumerate(examples):
        key = normalize(ex.text)
        if key in by_norm:
            union(i, by_norm[key])
        else:
            by_norm[key] = i

    lsh = MinHashLSH(seed=seed)
    shingle_sets = [shingles(ex.text) for ex in examples]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i in by_norm.values():
        for key in lsh.band_keys(lsh.signature(shingle_sets[i])):
            buckets[key].append(i)
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1 :]:
                pair = (min(a, b), max(a, b))
                if pair in checked:
                    continue
                checked.add(pair)
                if find(a) != find(b) and jaccard(shingle_sets[a], shingle_sets[b]) >= threshold:
                    union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(examples)):
        groups[find(i)].append(i)
    kept: list[Example] = []
    conflicts = 0
    for members in groups.values():
        labels = {examples[i].label for i in members}
        if len(labels) > 1:
            conflicts += 1
            continue
        kept.append(examples[min(members)])
    stats = {
        "input": len(examples),
        "groups": len(groups),
        "duplicates_removed": len(examples) - len(groups),
        "conflicting_groups_dropped": conflicts,
        "output": len(kept),
    }
    return kept, stats


def remove_overlap(
    holdout: list[Example], reference: list[Example], threshold: float = 0.8, seed: int = 0
) -> tuple[list[Example], int]:
    """Drop held-out examples that exactly or nearly duplicate anything on the training side."""
    lsh = MinHashLSH(seed=seed)
    ref_norm = {normalize(ex.text) for ex in reference}
    ref_shingles = [shingles(ex.text) for ex in reference]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i, sh in enumerate(ref_shingles):
        for key in lsh.band_keys(lsh.signature(sh)):
            buckets[key].append(i)
    kept: list[Example] = []
    for ex in holdout:
        if normalize(ex.text) in ref_norm:
            continue
        sh = shingles(ex.text)
        candidates = {i for key in lsh.band_keys(lsh.signature(sh)) for i in buckets.get(key, [])}
        if any(jaccard(sh, ref_shingles[i]) >= threshold for i in candidates):
            continue
        kept.append(ex)
    return kept, len(holdout) - len(kept)


def stratified_split(
    examples: list[Example], val: float, test: float, seed: int
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split per (source, label) stratum so every split keeps the same mix."""
    strata: dict[tuple[str, int], list[Example]] = defaultdict(list)
    for ex in examples:
        strata[(ex.source, ex.label)].append(ex)
    rng = random.Random(seed)
    train, val_set, test_set = [], [], []
    for key in sorted(strata):
        items = strata[key]
        rng.shuffle(items)
        n_test = round(len(items) * test)
        n_val = round(len(items) * val)
        test_set += items[:n_test]
        val_set += items[n_test : n_test + n_val]
        train += items[n_test + n_val :]
    for split in (train, val_set, test_set):
        rng.shuffle(split)
    return train, val_set, test_set


def write_jsonl(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line. Iterates the file rather than calling str.splitlines(),
    which would also split on Unicode separators such as U+2028 inside prompt text."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(examples: list[Example]) -> dict:
    counts = Counter((ex.source, ex.label) for ex in examples)
    return {
        "total": len(examples),
        "injection": sum(1 for ex in examples if ex.label == INJECTION),
        "benign": sum(1 for ex in examples if ex.label == BENIGN),
        "by_source": {
            f"{s}/{'injection' if lbl else 'benign'}": n for (s, lbl), n in sorted(counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--holdout", default="gandalf", choices=sorted(SOURCES))
    parser.add_argument("--near-dup", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude",
        type=Path,
        default=Path("eval/tricky_benign.jsonl"),
        help="Evaluation prompts that must not appear (even nearly) in any split",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    loaded = {key: load_source(key, args.seed) for key in SOURCES}
    for key, rows in loaded.items():
        print(f"loaded {len(rows):>6} from {SOURCES[key].name}")

    pool = [ex for key, rows in loaded.items() if key != args.holdout for ex in rows]
    pool, dedupe_stats = dedupe(pool, args.near_dup, args.seed)
    excluded = 0
    if args.exclude.exists():
        eval_rows = [Example(r["text"], BENIGN, "eval") for r in read_jsonl(args.exclude)]
        pool, excluded = remove_overlap(pool, eval_rows, args.near_dup, args.seed)
    holdout, holdout_dedupe = dedupe(loaded[args.holdout], args.near_dup, args.seed)
    holdout, overlap = remove_overlap(holdout, pool, args.near_dup, args.seed)
    train, val, test = stratified_split(pool, args.val, args.test, args.seed)

    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "val.jsonl", val)
    write_jsonl(args.out / "test.jsonl", test)
    write_jsonl(args.out / "holdout.jsonl", holdout)
    stats = {
        "seed": args.seed,
        "near_dup_threshold": args.near_dup,
        "holdout_source": SOURCES[args.holdout].name,
        "dedupe": dedupe_stats,
        "holdout_dedupe": holdout_dedupe,
        "holdout_removed_for_overlap_with_training_side": overlap,
        "removed_for_overlap_with_eval_set": excluded,
        "splits": {
            "train": summarize(train),
            "val": summarize(val),
            "test": summarize(test),
            "holdout": summarize(holdout),
        },
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
