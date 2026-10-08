from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from app.rails.base import Action, Verdict
from app.rails.classifier import SequenceClassifier

# Maps texts to per-label probabilities.
LabelScorer = Callable[[Sequence[str]], list[dict[str, float]]]


@lru_cache(maxsize=4)
def load_toxicity_classifier(
    model_path: str, max_length: int = 256, max_windows: int | None = None
) -> LabelScorer:
    classifier = SequenceClassifier(model_path, max_length=max_length, max_windows=max_windows)

    def score(texts: Sequence[str]) -> list[dict[str, float]]:
        rows = classifier.label_scores(texts)
        return [dict(zip(classifier.labels, map(float, row), strict=True)) for row in rows]

    return score


class ToxicityRail:
    """Blocks text whose score for any watched toxicity label reaches the threshold.

    Built for multi-label models such as Detoxify's (toxicity, severe_toxicity, obscene, threat,
    insult, identity_attack). `labels` limits which ones count; by default all of them do,
    except a negative class such as "non-toxic" in single-label models.
    """

    name = "toxicity"

    def __init__(
        self,
        model: str,
        threshold: float = 0.5,
        labels: list[str] | None = None,
        max_length: int = 256,
        max_windows: int | None = 4,
        scorer: LabelScorer | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.labels = labels
        self.max_length = max_length
        self.max_windows = max_windows
        self._scorer = scorer

    @property
    def scorer(self) -> LabelScorer:
        if self._scorer is None:
            self._scorer = load_toxicity_classifier(self.model, self.max_length, self.max_windows)
        return self._scorer

    def warm_up(self) -> None:
        self.check("warm up")

    def check(self, text: str) -> Verdict:
        scores = self.scorer([text])[0]
        watched = {
            label: score
            for label, score in scores.items()
            if (label in self.labels if self.labels else not _is_negative(label))
        }
        if not watched:
            return Verdict(self.name, Action.ALLOW)
        label, score = max(watched.items(), key=lambda item: item[1])
        if score >= self.threshold:
            return Verdict(self.name, Action.BLOCK, score, f"{label} score {score:.3f}")
        return Verdict(self.name, Action.ALLOW, score)


def _is_negative(label: str) -> bool:
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    return normalized in {"non_toxic", "not_toxic", "neutral", "label_0", "benign", "safe"}
