import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.rails.base import Rail
from app.rails.pii import PiiRail
from app.rails.rules import RulesRail

_APP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RulesConfig(BaseModel):
    max_chars: int = 8000
    denylist: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)


class PiiConfig(BaseModel):
    mode: str = "redact"
    threshold: float = 0.4
    entities: list[str] = Field(
        default_factory=lambda: ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"]
    )


class Policy(BaseModel):
    name: str = "default"
    # Message roles whose content is screened. System prompts come from the app and are trusted.
    screen_roles: list[str] = Field(default_factory=lambda: ["user"])
    input_rails: dict[str, Any] = Field(default_factory=dict)

    def build_input_rails(self, spacy_model: str) -> list[Rail]:
        rails: list[Rail] = []
        # dict preserves YAML order, so the policy file decides rail order.
        for rail_name, cfg in self.input_rails.items():
            cfg = cfg or {}
            if rail_name == "rules":
                rails.append(RulesRail(**RulesConfig(**cfg).model_dump()))
            elif rail_name == "pii":
                rails.append(PiiRail(**PiiConfig(**cfg).model_dump(), spacy_model=spacy_model))
            else:
                raise ValueError(f"unknown input rail {rail_name!r} in policy {self.name!r}")
        return rails


class PolicyRegistry:
    """Loads policies/<app>.yaml on demand and falls back to default.yaml."""

    def __init__(self, policy_dir: str | Path, spacy_model: str) -> None:
        self.policy_dir = Path(policy_dir)
        self.spacy_model = spacy_model
        self._cache: dict[str, tuple[Policy, list[Rail]]] = {}

    def _load(self, name: str) -> tuple[Policy, list[Rail]] | None:
        path = self.policy_dir / f"{name}.yaml"
        if not path.is_file():
            return None
        data = yaml.safe_load(path.read_text()) or {}
        policy = Policy(name=name, **data)
        return policy, policy.build_input_rails(self.spacy_model)

    def get(self, app: str | None) -> tuple[Policy, list[Rail]]:
        name = app if app and _APP_ID.match(app) else "default"
        if name not in self._cache:
            loaded = self._load(name)
            if loaded is None:
                if name == "default":
                    raise FileNotFoundError(self.policy_dir / "default.yaml")
                return self.get("default")
            self._cache[name] = loaded
        return self._cache[name]
