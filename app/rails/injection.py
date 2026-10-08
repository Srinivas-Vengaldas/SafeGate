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

    A model directory containing `model.onnx` runs on ONNX Runtime with the plain `tokenizers`
    library, which keeps PyTorch and most of transformers out of memory. Otherwise the Hugging
    Face checkpoint runs on PyTorch.
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 256,
        stride: int = 64,
        threads: int | None = None,
        max_windows: int | None = None,
    ) -> None:
        self.max_length = max_length
        self.stride = stride
        # Cap on windows scored per text, keeping the first and last ones, where injections
        # usually sit. Bounds latency and memory for very long prompts. None scores them all.
        self.max_windows = max_windows
        path = Path(model_path)
        if (path / ONNX_FILE).is_file():
            self.backend = "onnx"
            self._encode, self._run = self._onnx_backend(path, threads)
            id2label = json.loads((path / "config.json").read_text())["id2label"]
        else:
            self.backend = "torch"
            self._encode, self._run, id2label = self._torch_backend(model_path, threads)
        labels = {str(v).upper(): int(k) for k, v in id2label.items()}
        self.positive = labels.get("INJECTION", 1)

    def _onnx_backend(self, path: Path, threads: int | None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        tokenizer.enable_truncation(self.max_length, stride=self.stride)
        tokenizer.no_padding()
        config = json.loads((path / "tokenizer_config.json").read_text())
        pad_id = tokenizer.token_to_id(config.get("pad_token") or "[PAD]") or 0

        options = ort.SessionOptions()
        # The arena and weight prepacking each keep extra copies of the weights; without them
        # the int8 model needs roughly a quarter of the memory, which matters on small hosts.
        options.enable_cpu_mem_arena = False
        options.add_session_config_entry("session.disable_prepacking", "1")
        if threads:
            options.intra_op_num_threads = threads
        # Constant folding would dequantize the int8 embedding table back to fp32 (+170 MB).
        # Weights exported to model.onnx.data are memory-mapped rather than copied.
        session = ort.InferenceSession(
            str(path / ONNX_FILE),
            options,
            providers=["CPUExecutionProvider"],
            disabled_optimizers=["ConstantFolding"],
        )
        inputs = {i.name for i in session.get_inputs()}

        def encode(batch: list[str]) -> tuple[dict, list[int]]:
            windows, mapping = [], []
            for index, encoding in enumerate(tokenizer.encode_batch(batch)):
                for window in self._limit([encoding, *encoding.overflowing]):
                    windows.append(window)
                    mapping.append(index)
            width = max(len(w.ids) for w in windows)
            ids = np.full((len(windows), width), pad_id, dtype=np.int64)
            mask = np.zeros((len(windows), width), dtype=np.int64)
            types = np.zeros((len(windows), width), dtype=np.int64)
            for row, window in enumerate(windows):
                n = len(window.ids)
                ids[row, :n] = window.ids
                mask[row, :n] = 1
                types[row, :n] = window.type_ids
            enc = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            return {k: v for k, v in enc.items() if k in inputs}, mapping

        def run(enc: dict) -> np.ndarray:
            # One window at a time: DeBERTa's attention activations grow quickly with batch
            # size, and on CPU batching barely helps throughput.
            rows = len(next(iter(enc.values())))
            return np.concatenate(
                [
                    session.run(None, {k: v[i : i + 1] for k, v in enc.items()})[0]
                    for i in range(rows)
                ]
            )

        return encode, run

    def _torch_backend(self, model_path: str, threads: int | None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()

        def encode(batch: list[str]) -> tuple[dict, list[int]]:
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
            keep = []
            for index in sorted(set(mapping)):
                rows = [row for row, sample in enumerate(mapping) if sample == index]
                keep += self._limit(rows)
            return {k: v[keep] for k, v in enc.items()}, [mapping[row] for row in keep]

        def run(enc: dict) -> np.ndarray:
            with torch.inference_mode():
                return model(**enc).logits.float().numpy()

        return encode, run, model.config.id2label

    def _limit(self, windows: list) -> list:
        if self.max_windows is None or len(windows) <= self.max_windows:
            return windows
        head = (self.max_windows + 1) // 2
        tail = self.max_windows - head
        return windows[:head] + (windows[-tail:] if tail else [])

    def __call__(self, texts: Sequence[str], batch_size: int = 16) -> list[float]:
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), batch_size):
            enc, mapping = self._encode(list(texts[start : start + batch_size]))
            logits = self._run(enc)
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = (exp / exp.sum(axis=-1, keepdims=True))[:, self.positive].tolist()
            for window, sample in enumerate(mapping):
                index = start + sample
                scores[index] = max(scores[index], probs[window])
        return scores


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
