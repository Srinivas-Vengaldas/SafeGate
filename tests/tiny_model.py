"""Builds a tiny randomly initialized BERT classifier on disk, so ML code paths can be tested
without downloading a real checkpoint."""

from pathlib import Path


def make_tiny_classifier(
    path: Path,
    words: list[str],
    init_range: float = 0.02,
    labels: tuple[str, ...] = ("BENIGN", "INJECTION"),
    problem_type: str | None = None,
) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(set(words))]
    (path / "vocab.txt").write_text("\n".join(vocab))
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        num_labels=len(labels),
        initializer_range=init_range,
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        problem_type=problem_type,
    )
    BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
