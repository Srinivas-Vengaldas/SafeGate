import hashlib
import re
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml
from pydantic import BaseModel, Field

from app.rails.base import Rail
from app.rails.injection import InjectionRail
from app.rails.pii import PiiRail
from app.rails.rules import RulesRail
from app.rails.secrets import SecretsRail
from app.rails.toxicity import ToxicityRail

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


class SecretsConfig(BaseModel):
    mode: str = "redact"
    formats: list[str] | None = None  # None: every known format in app/rails/secrets.py


class InjectionConfig(BaseModel):
    model: str  # local directory or Hugging Face Hub id
    threshold: float = 0.5
    max_length: int = 256
    max_windows: int | None = 4  # first and last windows of very long prompts


class ToxicityConfig(BaseModel):
    model: str  # local directory or Hugging Face Hub id
    threshold: float = 0.5
    labels: list[str] | None = None  # None watches every label except a negative class
    max_length: int = 256
    max_windows: int | None = 4


class AttachmentsConfig(BaseModel):
    """Images and files attached to chat requests (see app/attachments.py)."""

    screen: bool = True
    max_bytes: int = 10 * 2**20  # per attachment
    max_chars: int = 200_000  # of text extracted from one attachment
    max_pages: int = 50
    ocr_lang: str = "eng"
    # Extracted text is screened in pieces of this size, each like a prompt.
    chunk_chars: int = 4000
    # Attachments SafeGate cannot read (remote URLs, uploaded file ids, unknown formats) cannot
    # be vouched for, so they are blocked unless this is "allow".
    unreadable: Literal["block", "allow"] = "block"


class RateLimitConfig(BaseModel):
    requests: int = 60
    per_seconds: int = 60


class Policy(BaseModel):
    name: str = "default"
    # Message roles whose content is screened. System prompts come from the app and are trusted.
    screen_roles: list[str] = Field(default_factory=lambda: ["user"])
    input_rails: dict[str, Any] = Field(default_factory=dict)
    # Screen the model's reply before the client sees it. Streams are buffered when set.
    output_rails: dict[str, Any] = Field(default_factory=dict)
    # Rails for third-party text an app retrieves (RAG passages and the documents indexed for
    # them). Unset: the input rails. A classifier trained on prompts can misread documents that
    # merely discuss LLMs as attacks, so a policy may screen documents differently.
    context_rails: dict[str, Any] | None = None
    # How a blocked chat request is answered: an OpenAI-style 400 error, or a normal completion
    # whose message explains the block (finish_reason "content_filter"), for clients that treat
    # errors as outages.
    on_block: Literal["error", "refuse"] = "error"
    # Reply text for refused requests and withheld replies; {rail} names the blocking rail.
    refusal_message: str = (
        "I'm sorry, but I can't help with that request. (Blocked by SafeGate: {rail} rail.)"
    )
    withheld_message: str = (
        "I'm sorry, but I can't share that response. (Withheld by SafeGate: {rail} rail.)"
    )
    rate_limit: RateLimitConfig | None = None
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)

    def build_input_rails(self, spacy_model: str) -> list[Rail]:
        return self._build(self.input_rails, "input", spacy_model)

    def build_context_rails(self, spacy_model: str) -> list[Rail] | None:
        if self.context_rails is None:
            return None
        return self._build(self.context_rails, "context", spacy_model)

    def build_output_rails(self, spacy_model: str) -> list[Rail]:
        return self._build(self.output_rails, "output", spacy_model)

    def _build(self, section: dict[str, Any], stage: str, spacy_model: str) -> list[Rail]:
        rails: list[Rail] = []
        # dict preserves YAML order, so the policy file decides rail order.
        for rail_name, cfg in section.items():
            cfg = cfg or {}
            if rail_name == "rules":
                rails.append(RulesRail(**RulesConfig(**cfg).model_dump()))
            elif rail_name == "pii":
                rails.append(PiiRail(**PiiConfig(**cfg).model_dump(), spacy_model=spacy_model))
            elif rail_name == "secrets":
                rails.append(SecretsRail(**SecretsConfig(**cfg).model_dump()))
            elif rail_name == "injection":
                rails.append(InjectionRail(**InjectionConfig(**cfg).model_dump()))
            elif rail_name == "toxicity":
                rails.append(ToxicityRail(**ToxicityConfig(**cfg).model_dump()))
            else:
                raise ValueError(f"unknown {stage} rail {rail_name!r} in policy {self.name!r}")
        return rails


class LoadedPolicy(NamedTuple):
    policy: Policy
    input_rails: list[Rail]
    output_rails: list[Rail]
    fingerprint: str  # changes whenever the policy file does, so cached verdicts expire with it
    context_rails: list[Rail]


class PolicyRegistry:
    """Loads policies/<app>.yaml on demand and falls back to default.yaml."""

    def __init__(self, policy_dir: str | Path, spacy_model: str) -> None:
        self.policy_dir = Path(policy_dir)
        self.spacy_model = spacy_model
        self._cache: dict[str, LoadedPolicy] = {}

    def _load(self, name: str) -> LoadedPolicy | None:
        path = self.policy_dir / f"{name}.yaml"
        if not path.is_file():
            return None
        source = path.read_text()
        policy = Policy(name=name, **(yaml.safe_load(source) or {}))
        input_rails = policy.build_input_rails(self.spacy_model)
        context_rails = policy.build_context_rails(self.spacy_model)
        return LoadedPolicy(
            policy,
            input_rails,
            policy.build_output_rails(self.spacy_model),
            hashlib.sha256(f"{name}\0{source}".encode()).hexdigest()[:16],
            input_rails if context_rails is None else context_rails,
        )

    def get(self, app: str | None) -> LoadedPolicy:
        name = app if app and _APP_ID.match(app) else "default"
        if name not in self._cache:
            loaded = self._load(name)
            if loaded is None:
                if name == "default":
                    raise FileNotFoundError(self.policy_dir / "default.yaml")
                return self.get("default")
            self._cache[name] = loaded
        return self._cache[name]
