"""Checks each sentence of a generated answer against the sources it cites.

No model: a sentence counts as supported when it cites a source, every number in it appears in
the cited sources, and most of its content words do too. That catches the failures that matter
most in a cited answer (a wrong figure, a claim pinned on a source that doesn't contain it, a
claim with no citation) in well under a millisecond. It does not judge paraphrases that keep
the words but change the meaning; an NLI model would, at the cost of memory the free demo host
does not have.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.rag.embed import _STOPWORDS, _stem

_CITATION = re.compile(r"\[(\d{1,2})\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
_WORD = re.compile(r"[a-z][a-z0-9']*")
# Connective words an answer adds when it paraphrases, which no source needs to contain.
_GLUE = frozenset(
    "also because rather via using use used uses include includes including provide provides "
    "provided should specifically easily only entirely just such like well then thus overall "
    "example instance".split()
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A sentence needs this many content words to be checked; shorter ones ("It helps:") are glue.
_MIN_WORDS = 4


@dataclass
class Claim:
    text: str
    citations: list[int]
    status: str  # supported, unsupported or uncited
    overlap: float = 0.0  # share of the sentence's content words found in its cited sources
    missing_numbers: list[str] = field(default_factory=list)
    missing_words: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[str]:
    out = set()
    for match in _NUMBER.findall(text):
        value = match.replace(",", "").rstrip(".")
        # 2,500 and 2500 match; 3.0 and 3 match.
        out.add(value[:-2] if value.endswith(".0") else value)
    return out


def _norm(word: str) -> str:
    word = _stem(word.strip("'"))
    # "duplicate", "duplicates" and "duplicating" all end up as "duplicat".
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def _words(text: str) -> set[str]:
    return {
        _norm(w)
        for w in _WORD.findall(text.lower().replace("-", " "))
        if w not in _STOPWORDS and w not in _GLUE and len(w) > 2
    }


_TRAILING_CITATIONS = re.compile(r"([.!?])[ \t]*((?:\[\d{1,2}\])+)")


def sentences(answer: str) -> list[str]:
    # "...Actions. [1] It was..." cites the sentence before it: move the citation inside.
    answer = _TRAILING_CITATIONS.sub(r" \2\1", answer)
    parts = [p.strip() for p in _SENTENCE.split(answer)]
    return [p.lstrip("-*• ").strip() for p in parts if p and p.strip("-*• ")]


def check(answer: str, sources: dict[int, str], min_overlap: float = 0.5) -> list[Claim]:
    """One claim per answer sentence. `sources` maps citation numbers to source text."""
    words = {n: _words(text) for n, text in sources.items()}
    numbers = {n: _numbers(text) for n, text in sources.items()}
    claims = []
    for sentence in sentences(answer):
        cited = [int(n) for n in _CITATION.findall(sentence) if int(n) in sources]
        bare = _CITATION.sub("", sentence)
        content = _words(bare)
        if (len(content) < _MIN_WORDS and not _numbers(bare)) or bare.rstrip().endswith(":"):
            continue  # glue ("It helps:"), not a claim
        if not cited:
            claims.append(Claim(bare.strip(), [], "uncited"))
            continue
        source_words = set().union(*(words[n] for n in cited))
        source_numbers = set().union(*(numbers[n] for n in cited))
        missing_numbers = sorted(_numbers(bare) - source_numbers)
        missing_words = sorted(content - source_words)
        overlap = 1 - len(missing_words) / len(content) if content else 1.0
        supported = not missing_numbers and overlap >= min_overlap
        claims.append(
            Claim(
                bare.strip(),
                cited,
                "supported" if supported else "unsupported",
                round(overlap, 3),
                missing_numbers,
                missing_words,
            )
        )
    return claims


def summary(claims: list[Claim]) -> dict:
    counts = {
        s: sum(c.status == s for c in claims) for s in ("supported", "unsupported", "uncited")
    }
    return {
        "claims": [asdict(c) for c in claims],
        **counts,
        "grounded": counts["supported"] / len(claims) if claims else None,
    }
