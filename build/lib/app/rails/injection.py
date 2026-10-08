from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.rails.base import Action, Verdict

Scorer = Callable[[Sequence[str]], list[float]]


ONNX_FILE = "model.onnx"


class InjectionClassifier:
    """Scores texts with a fine-tuned sequence classifier on CPU.

    Long inputs are split into overlapping token windows and a text's score is the maximum
    window score, so an injection buried at the end of a long prompt is not truncated away.

    A model directory containing `model.onnx` runs on ONNX Runtime (no PyTorch needed);
    otherwise the Hugging Face checkpoint runs on PyTorch.
    """

    def __init__(
        self, model_path: str, max_length: int = 256, stride: int = 64, threads: int | None = None
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.stride = stride
        onnx_path = Path(model_path) / ONNX_FILE
        if onnx_path.is_file():
            self.backend = "onnx"
            self._run = self._onnx_runner(onnx_path, threads)
            id2label = json.loads((Path(model_path) / "config.json").read_text())["id2label"]
        else:
            self.backend = "torch"
            self._run, id2label = self._torch_runner(model_path, threads)
        labels = {str(v).upper(): int(k) for k, v in id2label.items()}
        self.positive = labels.get("INJECTION", 1)

    @staticmethod
    def _onnx_runner(path: Path, threads: int | None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = {i.name for i in session.get_inputs()}

        def run(enc: dict) -> np.ndarray:
            feed = {k: np.asarray(v, dtype=np.int64) for k, v in enc.items() if k in inputs}
            return session.run(None, feed)[0]

        return run

    @staticmethod
    def _torch_runner(model_path: str, threads: int | None):
        import torch
        from transformers import AutoModelForSequenceClassification

        if threads:
            torch.set_num_threads(threads)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()

        def run(enc: dict) -> np.ndarray:
            with torch.inference_mode():
                tensors = {k: torch.as_tensor(np.asarray(v)) for k, v in enc.items()}
                return model(**tensors).logits.float().numpy()

        return run, model.config.id2label

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
                return_tensors="np",
            )
            mapping = enc.pop("overflow_to_sample_mapping").tolist()
            logits = self._run(dict(enc))
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = (exp / exp.sum(axis=-1, keepdims=True))[:, self.positive].tolist()
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
