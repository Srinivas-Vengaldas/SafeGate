from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from app.rails.base import Action, Verdict

Scorer = Callable[[Sequence[str]], list[float]]


class InjectionClassifier:
    """Scores texts with a fine-tuned sequence classifier (CPU by default).

    Long inputs are split into overlapping token windows and a text's score is the maximum
    window score, so an injection buried at the end of a long prompt is not truncated away.
    """

    def __init__(
        self, model_path: str, max_length: int = 256, stride: int = 64, threads: int | None = None
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        self.max_length = max_length
        self.stride = stride
        labels = {v.upper(): int(k) for k, v in self.model.config.id2label.items()}
        self.positive = labels.get("INJECTION", 1)

    def __call__(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            mapping = enc.pop("overflow_to_sample_mapping").tolist()
            with self._torch.inference_mode():
                logits = self.model(**enc).logits
            probs = logits.softmax(dim=-1)[:, self.positive].tolist()
            for window, sample in enumerate(mapping):
                index = start + sample
                scores[index] = max(scores[index], probs[window])
        return scores


@lru_cache(maxsize=4)
def load_classifier(model_path: str, max_length: int = 256) -> InjectionClassifier:
    return InjectionClassifier(model_path, max_length=max_length)


class InjectionRail:
    """Blocks text whose prompt-injection score reaches the policy threshold."""

    name = "injection"

    def __init__(
        self,
        model: str,
        threshold: float = 0.5,
        max_length: int = 256,
        scorer: Scorer | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.max_length = max_length
        self._scorer = scorer

    @property
    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = load_classifier(self.model, self.max_length)
        return self._scorer

    def warm_up(self) -> None:
        self.check("warm up")

    def check(self, text: str) -> Verdict:
        score = float(self.scorer([text])[0])
        if score >= self.threshold:
            return Verdict(self.name, Action.BLOCK, score, f"injection score {score:.3f}")
        return Verdict(self.name, Action.ALLOW, score)
