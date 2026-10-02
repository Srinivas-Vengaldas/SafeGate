import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from training.evaluate import best_f1_threshold, binary_metrics  # noqa: E402


def test_threshold_is_mid_range_when_best_f1_ties():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    threshold = best_f1_threshold(labels, scores)
    # Every threshold in (0.03, 0.97] separates perfectly; the choice sits well inside it.
    assert 0.2 < threshold < 0.8
    assert binary_metrics(labels, (scores >= threshold).astype(int))["f1"] == 1.0


def test_threshold_keeps_lowest_positive_when_classes_overlap():
    labels = np.array([0, 0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    threshold = best_f1_threshold(labels, scores)
    preds = (scores >= threshold).astype(int)
    assert binary_metrics(labels, preds)["f1"] >= 0.85


def test_binary_metrics_counts():
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["fpr"] == 0.5
