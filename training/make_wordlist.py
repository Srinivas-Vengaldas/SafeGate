"""Writes the data the secrets rail uses to tell an ordinary word from a made-up string such as
a password: app/rails/data/english_words.txt.gz, a word list, and app/rails/data/trigrams.json.gz,
a character trigram model of English spelling that scores how word-like a string is.

Needs wordfreq (pip install wordfreq), which is not a runtime dependency.
"""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from wordfreq import top_n_list

DATA = Path(__file__).resolve().parent.parent / "app/rails/data"


def trigram_model(words: list[str]) -> dict:
    """log P(c | previous two characters), add-k smoothed, over words padded with ^ and $."""
    k, alphabet = 0.1, 27
    trigrams, bigrams = Counter(), Counter()
    for word in words:
        padded = f"^{word}$"
        for i in range(len(padded) - 2):
            trigrams[padded[i : i + 3]] += 1
            bigrams[padded[i : i + 2]] += 1
    seen = {
        t: round(math.log((n + k) / (bigrams[t[:2]] + k * alphabet)), 3)
        for t, n in trigrams.items()
    }
    unseen = {b: round(math.log(k / (n + k * alphabet)), 3) for b, n in bigrams.items()}
    return {"seen": seen, "unseen": unseen, "default": round(math.log(1 / alphabet), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=200_000, help="most frequent English words")
    parser.add_argument("--model-top", type=int, default=100_000, help="words for the trigrams")
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    english = [w for w in top_n_list("en", args.top) if w.isascii() and w.isalpha()]
    # Shorter strings are never treated as secrets, so they need no entry.
    words = sorted({w for w in english if len(w) >= 5})
    (args.out / "english_words.txt.gz").write_bytes(
        gzip.compress("\n".join(words).encode(), compresslevel=9, mtime=0)
    )
    model = trigram_model(english[: args.model_top])
    (args.out / "trigrams.json.gz").write_bytes(
        gzip.compress(json.dumps(model, sort_keys=True).encode(), compresslevel=9, mtime=0)
    )
    print(f"wrote {len(words)} words and {len(model['seen'])} trigrams to {args.out}")


if __name__ == "__main__":
    main()
