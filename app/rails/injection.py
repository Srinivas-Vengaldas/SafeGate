from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from app.rails.base import Action, Verdict
from app.rails.classifier import ONNX_FILE, SequenceClassifier, cpu_budget

__all__ = ["ONNX_FILE", "InjectionClassifier", "InjectionRail", "cpu_budget", "load_classifier"]

Scorer = Callable[[Sequence[str]], list[float]]


class InjectionClassifier(SequenceClassifier):
    """Scores the probability that each text is a prompt injection."""

    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        upper = [label.upper() for label in self.labels]
        self.positive = upper.index("INJECTION") if "INJECTION" in upper else 1

    def __call__(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        return self.label_scores(texts, batch_size)[:, self.positive].tolist()


@lru_cache(maxsize=4)
def load_classifier(
    model_path: str, max_length: int = 256, max_windows: int | None = None
) -> InjectionClassifier:
    return InjectionClassifier(model_path, max_length=max_length, max_windows=max_windows)


class InjectionRail:
    """Blocks text whose prompt-injection score reaches the policy threshold."""

    name = "injection"

    def __init__(
        self,
        model: str,
        threshold: float = 0.5,
        max_length: int = 256,
        max_windows: int | None = 4,
        scorer: Scorer | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.max_length = max_length
        self.max_windows = max_windows
        self._scorer = scorer

    @property
    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = load_classifier(self.model, self.max_length, self.max_windows)
        return self._scorer

    def warm_up(self) -> None:
        self.check("warm up")

    def check(self, text: str) -> Verdict:
        score = float(self.scorer([text])[0])
        if score >= self.threshold:
            return Verdict(self.name, Action.BLOCK, score, f"injection score {score:.3f}")
        return Verdict(self.name, Action.ALLOW, score)
