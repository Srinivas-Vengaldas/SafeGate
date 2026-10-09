import gzip
import json
import math
import re
from collections import Counter
from functools import cache
from pathlib import Path

from app.rails.base import Action, Verdict

# Credentials with a recognizable shape, from the prefixes their issuers document. SK_API_KEY
# covers the "sk-" keys several LLM providers use.
KNOWN_FORMATS: dict[str, str] = {
    "PRIVATE_KEY": (
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?"
        r"(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----|$)"
    ),
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b",
    "SK_API_KEY": r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "SLACK_TOKEN": r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b",
    "STRIPE_KEY": r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "CONNECTION_STRING": r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+",
}

_CREDENTIAL = (
    r"(?:pass(?:word|wd|phrase)?|pwd|api[\s_-]?(?:key|secret|token)|secret(?:[\s_-]?key)?"
    r"|access[\s_-]?(?:key|token)|auth[\s_-]?token|bearer|token|client[\s_-]?secret)"
)
_VALUE = r"[\"'`]?(?P<value>[^\s\"'`,;]{6,})"

# A value introduced by a credential word: "password is hunter2!", "API_KEY=...", "token: ...",
# "changed my password to Tr0ub4dor".
_ASSIGNED = re.compile(
    rf"\b{_CREDENTIAL}(?:\s+(?:value|for\s+\S+))?\s*(?:is|=|:|was|->|to)\s*{_VALUE}",
    re.IGNORECASE,
)
# A value named as a credential after the fact: "she gave me 1jeunen as a password",
# "hunter2! is my password".
_NAMED = re.compile(
    rf"{_VALUE}[\"'`]?\s+(?:as|is|was)\s+(?:(?:my|a|the|our|your|his|her|their)\s+)?"
    rf"(?:new\s+|current\s+|old\s+)?{_CREDENTIAL}\b",
    re.IGNORECASE,
)


# How far after a credential word a made-up value can still sit: "a shared secret to set up the
# channel, like infjenkf".
_NEARBY_TOKENS = 12
_CREDENTIAL_WORD = re.compile(rf"\b{_CREDENTIAL}", re.IGNORECASE)
_TOKEN = re.compile(r"\S+")
_DATA = Path(__file__).parent / "data"
# Mean trigram log-probability at or below which a lowercase string reads as keyboard mashing
# ("infjenkf" -4.1, "qwertyuiop" -3.9) rather than jargon ("nextjs" -3.6, "runbook" -2.4).
_GIBBERISH = -3.75

# Security vocabulary that is no dictionary word but comes up when people talk about secrets.
_TERMS = frozenset(
    "base32 base58 base64 base64url sha1 sha224 sha256 sha384 sha512 sha3 md5 hmac hs256 hs384"
    " hs512 rs256 rs512 es256 ps256 eddsa ed25519 x25519 aes128 aes256 chacha20 poly1305 utf8"
    " utf16 bcrypt scrypt argon2 argon2i argon2d argon2id pbkdf2 oauth oauth2 openid webauthn"
    " passkey passkeys totp hotp dotenv keycloak kerberos tiktoken bitwarden keepass lastpass"
    " 1password".split()
)


@cache
def _english_words() -> frozenset[str]:
    """The 150k most common English words of five letters or more (training/make_wordlist.py)."""
    return frozenset(
        gzip.decompress((_DATA / "english_words.txt.gz").read_bytes()).decode().split()
    )


@cache
def _trigrams() -> dict:
    return json.loads(gzip.decompress((_DATA / "trigrams.json.gz").read_bytes()))


def _wordlikeness(word: str) -> float:
    """Mean log-probability of a lowercase word's letters under a trigram model of English."""
    model = _trigrams()
    padded = f"^{word}$"
    scores = [
        model["seen"].get(padded[i : i + 3])
        or model["unseen"].get(padded[i : i + 2], model["default"])
        for i in range(len(padded) - 2)
    ]
    return sum(scores) / len(scores)


def _entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(n / len(value) * math.log2(n / len(value)) for n in counts.values())


def _looks_secret(value: str) -> bool:
    """For a value a credential word introduces. Real words ("password is required", "token is
    missing") are not secrets; made-up strings and values that mix character classes are."""
    if value.lower() in _TERMS:
        return False
    if value.isalpha():
        return value.lower() not in _english_words()
    classes = sum(
        bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]")
    )
    return classes >= 2 or _entropy(value) >= 3.0


def _looks_made_up(value: str) -> bool:
    """For a value that merely appears near a credential word, so stricter: a lowercase string
    that is no English word and does not read like one ("infjenkf"), or letters and digits
    interleaved ("1jeunen", "i0ijieqwjd9u9"). Jargon ("nextjs"), names, acronyms, URLs and words
    with a version number ("OAuth2", "base64") do not count."""
    if len(value) < 6 or re.search(r"[/@.:]", value) or value.lower() in _TERMS:
        return False
    if value.isalpha():
        return (
            value.islower() and value not in _english_words() and _wordlikeness(value) <= _GIBBERISH
        )
    letters = re.sub(r"[^A-Za-z]", "", value)
    if not letters or not re.search(r"[0-9]", value):
        return False
    if re.fullmatch(r"[A-Za-z]+[0-9]+", value):
        # A word or name with a number after it; only an explicit "password is hunter2" counts.
        return False
    return True


class SecretsRail:
    """Detects credentials (API keys, tokens, private keys, passwords) and redacts or blocks them.

    Known formats are matched by shape. Anything else counts when a credential word introduces a
    value that looks like one ("my API secret is i0ijieqwjd9u9", "1jeunen as a password"), or
    when a made-up string such as "infjenkf" follows a credential word within a few words; an
    English word list tells made-up strings from real words. Runs on prompts (keep keys
    from reaching a third-party LLM and its logs) and on replies (keep the model from leaking
    ones it saw)."""

    name = "secrets"

    def __init__(self, mode: str = "redact", formats: list[str] | None = None) -> None:
        if mode not in ("redact", "block"):
            raise ValueError(f"secrets mode must be 'redact' or 'block', got {mode!r}")
        self.mode = mode
        names = formats or list(KNOWN_FORMATS)
        unknown = set(names) - set(KNOWN_FORMATS)
        if unknown:
            raise ValueError(f"unknown secret formats: {sorted(unknown)}")
        self.patterns = {n: re.compile(KNOWN_FORMATS[n]) for n in names}

    def warm_up(self) -> None:
        _english_words()
        _trigrams()

    def _find(self, text: str) -> list[tuple[int, int, str]]:
        spans = [
            (m.start(), m.end(), name)
            for name, pattern in self.patterns.items()
            for m in pattern.finditer(text)
        ]
        for m in [*_ASSIGNED.finditer(text), *_NAMED.finditer(text)]:
            value = m.group("value")
            if _looks_secret(value.rstrip(".!?)")):
                spans.append((m.start("value"), m.end("value"), "SECRET"))
        for m in _CREDENTIAL_WORD.finditer(text):
            for i, token in enumerate(_TOKEN.finditer(text, m.end())):
                if i == _NEARBY_TOKENS:
                    break
                value = token.group().strip("\"'`()[]{}<>,;.!?")
                if _looks_made_up(value):
                    start = token.start() + token.group().index(value)
                    spans.append((start, start + len(value), "SECRET"))
        # Keep the earliest of overlapping spans (a known format inside an assignment, say).
        kept: list[tuple[int, int, str]] = []
        for span in sorted(spans):
            if not kept or span[0] >= kept[-1][1]:
                kept.append(span)
        return kept

    def check(self, text: str) -> Verdict:
        spans = self._find(text)
        if not spans:
            return Verdict(self.name, Action.ALLOW)
        found = ", ".join(sorted({kind for _, _, kind in spans}))
        if self.mode == "block":
            return Verdict(self.name, Action.BLOCK, 1.0, f"found {found}")
        redacted = text
        for start, end, kind in reversed(spans):
            redacted = f"{redacted[:start]}<{kind}>{redacted[end:]}"
        return Verdict(self.name, Action.REDACT, 1.0, f"redacted {found}", redacted)
