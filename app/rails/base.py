from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True)
class Verdict:
    rail: str
    action: Action
    score: float = 0.0
    reason: str = ""
    # Set only when action is REDACT: the text later rails and the LLM should see.
    redacted_text: str | None = None


class Rail(Protocol):
    name: str

    def check(self, text: str) -> Verdict: ...
