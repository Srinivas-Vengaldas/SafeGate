import math
import re
from collections import Counter

from app.rails.base import Action, Verdict

# Credentials with a recognizable shape, from the prefixes their issuers document.
KNOWN_FORMATS: dict[str, str] = {
    "PRIVATE_KEY": (
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?"
        r"(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----|$)"
    ),
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b",
    "OPENAI_KEY": r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "SLACK_TOKEN": r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b",
    "STRIPE_KEY": r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    "CONNECTION_STRING": r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+",
}

# A value introduced by a credential word: "password is hunter2!", "API_KEY=...", "token: ...".
_ASSIGNED = re.compile(
    r"\b(?:pass(?:word|wd|phrase)?|pwd|api[\s_-]?(?:key|secret|token)|secret(?:[\s_-]?key)?"
    r"|access[\s_-]?(?:key|token)|auth[\s_-]?token|bearer|token|client[\s_-]?secret)"
    r"(?:\s+(?:value|for\s+\S+))?\s*(?:is|=|:|was|->)\s*[\"'`]?(?P<value>[^\s\"'`,;]{6,})",
    re.IGNORECASE,
)


def _entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(n / len(value) * math.log2(n / len(value)) for n in counts.values())


def _looks_secret(value: str) -> bool:
    """Plain words ("password is required", "token is missing") are not secrets; values that
    mix character classes or look random are."""
    if value.lower() in {"required", "missing", "invalid", "expired", "correct", "incorrect"}:
        return False
    classes = sum(
        bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]")
    )
    return classes >= 2 or _entropy(value) >= 3.0


class SecretsRail:
    """Detects credentials (API keys, tokens, private keys, passwords) and redacts or blocks them.

    Known formats are matched by shape; anything else counts when a credential word introduces a
    value that looks like one, e.g. "my API secret is i0ijieqwjd9u9". Runs on prompts (keep keys
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

    def _find(self, text: str) -> list[tuple[int, int, str]]:
        spans = [
            (m.start(), m.end(), name)
            for name, pattern in self.patterns.items()
            for m in pattern.finditer(text)
        ]
        for m in _ASSIGNED.finditer(text):
            value = m.group("value")
            if _looks_secret(value):
                spans.append((m.start("value"), m.end("value"), "SECRET"))
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
