import re

from app.rails.base import Action, Verdict


class RulesRail:
    """Cheap deterministic checks: length limit, case-insensitive denylist, regex patterns."""

    name = "rules"

    def __init__(
        self,
        max_chars: int = 8000,
        denylist: list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.denylist = [d.lower() for d in denylist or []]
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns or []]

    def check(self, text: str) -> Verdict:
        if len(text) > self.max_chars:
            return Verdict(self.name, Action.BLOCK, 1.0, f"length {len(text)} > {self.max_chars}")
        lowered = text.lower()
        for phrase in self.denylist:
            if phrase in lowered:
                return Verdict(self.name, Action.BLOCK, 1.0, f"denylist: {phrase!r}")
        for pattern in self.patterns:
            if pattern.search(text):
                return Verdict(self.name, Action.BLOCK, 1.0, f"pattern: {pattern.pattern!r}")
        return Verdict(self.name, Action.ALLOW)
