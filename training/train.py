"""Fine-tune a DeBERTa-v3 prompt-injection classifier.

Usage:
    python -m training.train --data data --out models/injection-deberta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ID2LABEL = {0: "BENIGN", 1: "INJECTION"}


def read_jsonl(path: Path, limit: int | None = None, seed: int = 0) -> Dataset:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = Dataset.from_list([{"text": r["text"], "label": int(r["label"])} for r in rows])
    if limit and len(data) > limit:
        data = data.shuffle(seed=seed).select(range(limit))
    return data


class WeightedTrainer(Trainer):
    """Cross-entropy with class weights, since benign prompts outnumber attacks."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--out", type=Path, default=Path("models/injection-deberta"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train = read_jsonl(args.data / "train.jsonl", args.max_train_samples, args.seed)
    val = read_jsonl(args.data / "val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = train.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val.map(tokenize, batched=True, remove_columns=["text"])

    counts = np.bincount(np.array(train["label"]), minlength=2)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float)
    print(f"train={len(train)} val={len(val)} class_counts={counts.tolist()}")

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_steps=0.1,  # float = ratio of total steps (transformers >= 5)
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights,
    )
    trainer.train()
    final = trainer.evaluate()
    print(json.dumps(final, indent=2))

    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    (args.out / "training_args.json").write_text(
        json.dumps({**{k: str(v) for k, v in vars(args).items()}, "val": final}, indent=2)
    )


if __name__ == "__main__":
    main()
