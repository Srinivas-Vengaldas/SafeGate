from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from app.rails.base import Action, Verdict


@lru_cache(maxsize=2)
def _engines(spacy_model: str) -> tuple[AnalyzerEngine, AnonymizerEngine]:
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    return analyzer, AnonymizerEngine()


class PiiRail:
    """Detects PII with Presidio and either redacts it (default) or blocks the request."""

    name = "pii"

    def __init__(
        self,
        entities: list[str],
        mode: str = "redact",
        threshold: float = 0.4,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        if mode not in ("redact", "block"):
            raise ValueError(f"pii mode must be 'redact' or 'block', got {mode!r}")
        self.entities = entities
        self.mode = mode
        self.threshold = threshold
        self.spacy_model = spacy_model

    def warm_up(self) -> None:
        # The first analyze() call initializes recognizers lazily; pay that cost at startup.
        self.check("warm up: mail a.b@example.com or call 212-555-0100")

    def check(self, text: str) -> Verdict:
        analyzer, anonymizer = _engines(self.spacy_model)
        results = [
            r
            for r in analyzer.analyze(text=text, entities=self.entities, language="en")
            if r.score >= self.threshold
        ]
        if not results:
            return Verdict(self.name, Action.ALLOW)
        found = sorted({r.entity_type for r in results})
        score = max(r.score for r in results)
        if self.mode == "block":
            return Verdict(self.name, Action.BLOCK, score, f"found {', '.join(found)}")
        redacted = anonymizer.anonymize(text=text, analyzer_results=results).text
        return Verdict(self.name, Action.REDACT, score, f"redacted {', '.join(found)}", redacted)
