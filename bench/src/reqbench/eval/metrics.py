"""Metrics of plan §6.3, computed from labels and (optionally) class probabilities.

Conventions:
- Labels are leaf names; `labels` fixes the class order and the probability columns.
- Level 1 = macro-F1 over {functional, non_functional}; level 2 = macro-F1 over the five NFR
  leaves, on the rows whose gold label is non-functional (a `functional` prediction there is a
  miss for the gold class).
- ECE uses equal-mass bins (15 by default); top-label ECE bins the max probability, class-wise
  ECE bins each class's one-vs-rest probability.
- Selective prediction ranks by the max probability. Tied confidences form one block that a
  threshold cannot split: the risk inside it is the expectation over random orders, and the
  coverage at a target precision can only stop at the end of a block. (Until 2026-11-11 ties
  kept the input order, which mattered for Jev's 2-decimal and B5's k = 5 probabilities.)
"""

from collections.abc import Sequence

import numpy as np

N_BINS = 15
TARGET_PRECISION = 0.95


def _codes(values: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    return np.array([index.get(v, -1) for v in values])


def confusion(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Rows = gold, columns = predicted; pairs with a value outside `labels` are ignored."""
    g, p = _codes(gold, labels), _codes(pred, labels)
    keep = (g >= 0) & (p >= 0)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (g[keep], p[keep]), 1)
    return matrix


def per_class(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict:
    """Precision, recall and F1 per class (0 when undefined), with supports."""
    g, p = _codes(gold, labels), _codes(pred, labels)
    out = {}
    for i, label in enumerate(labels):
        tp = int(np.sum((g == i) & (p == i)))
        fp = int(np.sum((g != i) & (p == i)))
        fn = int(np.sum((g == i) & (p != i)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> float:
    return float(np.mean([m["f1"] for m in per_class(gold, pred, labels).values()]))


def level1_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    def to_level1(v: str) -> str:
        return "functional" if v == "functional" else "non_functional"

    return macro_f1([to_level1(v) for v in gold], [to_level1(v) for v in pred],
                    ["functional", "non_functional"])  # fmt: skip


def level2_macro_f1(gold: Sequence[str], pred: Sequence[str], nfr_labels: Sequence[str]) -> float:
    rows = [(g, p) for g, p in zip(gold, pred, strict=True) if g in nfr_labels]
    if not rows:
        return float("nan")
    g, p = zip(*rows, strict=True)
    return macro_f1(g, p, nfr_labels)


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    return float(np.mean([g == p for g, p in zip(gold, pred, strict=True)]))


# ----- probabilities -------------------------------------------------------------------------


def _onehot(gold: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    codes = _codes(gold, labels)
    if (codes < 0).any():
        raise ValueError("gold labels outside `labels`")
    return np.eye(len(labels))[codes]


def log_loss(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str]) -> float:
    y = _onehot(gold, labels)
    return float(-np.mean(np.log(np.clip(np.sum(probs * y, axis=1), 1e-15, 1.0))))


def brier(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str]) -> float:
    """Multiclass Brier score: mean over rows of the squared error summed over classes."""
    return float(np.mean(np.sum((probs - _onehot(gold, labels)) ** 2, axis=1)))


def _ece(confidence: np.ndarray, outcome: np.ndarray, n_bins: int) -> float:
    order = np.argsort(confidence, kind="stable")
    total = len(confidence)
    error = 0.0
    for chunk in np.array_split(order, min(n_bins, total)):
        if len(chunk):
            error += abs(outcome[chunk].mean() - confidence[chunk].mean()) * len(chunk) / total
    return float(error)


def ece_top_label(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str],
                  n_bins: int = N_BINS) -> float:  # fmt: skip
    correct = probs.argmax(axis=1) == _codes(gold, labels)
    return _ece(probs.max(axis=1), correct.astype(float), n_bins)


def ece_per_class(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str],
                  n_bins: int = N_BINS) -> dict[str, float]:  # fmt: skip
    codes = _codes(gold, labels)
    return {
        label: _ece(probs[:, k], (codes == k).astype(float), n_bins)
        for k, label in enumerate(labels)
    }


# ----- selective prediction ------------------------------------------------------------------


def _ranked(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str]):
    """Errors sorted by decreasing confidence, and the end index of each block of tied
    confidences (rounded to 12 decimals, so float noise from averaging does not split one)."""
    correct = probs.argmax(axis=1) == _codes(gold, labels)
    confidence = np.round(probs.max(axis=1), 12)
    order = np.argsort(-confidence, kind="stable")
    sorted_conf = confidence[order]
    ends = np.append(np.nonzero(np.diff(sorted_conf))[0] + 1, len(order))
    return (~correct[order]).astype(float), ends


def risk_coverage(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Risk (error rate) of the k most confident predictions, for k = 1..n.

    Inside a block of tied confidences no threshold can separate the rows, so the cumulative
    errors are the expectation over random orders of the block (linear within it). This makes
    the curve independent of the input order, which otherwise decides the ties.
    """
    errors, ends = _ranked(gold, probs, labels)
    cum = np.cumsum(errors)
    expected = np.empty_like(cum)
    start, before = 0, 0.0
    for end in ends:
        size, block = end - start, cum[end - 1] - before
        expected[start:end] = before + block * np.arange(1, size + 1) / size
        start, before = end, cum[end - 1]
    return expected / np.arange(1, len(errors) + 1)


def aurc(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str]) -> float:
    return float(risk_coverage(gold, probs, labels).mean())


def coverage_at_precision(gold: Sequence[str], probs: np.ndarray, labels: Sequence[str],
                          target: float = TARGET_PRECISION) -> float:  # fmt: skip
    """Largest share of rows decided automatically by a confidence threshold while keeping
    accuracy >= target. Only block ends of tied confidences are reachable by a threshold."""
    errors, ends = _ranked(gold, probs, labels)
    cum = np.cumsum(errors)
    ok = [end for end in ends if 1 - cum[end - 1] / end >= target]
    return float(max(ok) / len(errors)) if ok else 0.0


# ----- summary -------------------------------------------------------------------------------


def summarize(gold: Sequence[str], pred: Sequence[str], probs: np.ndarray | None,
              labels: Sequence[str], nfr_labels: Sequence[str]) -> dict:  # fmt: skip
    """All point metrics of one system on one set of rows."""
    gold, pred = list(gold), list(pred)
    out = {
        "n": len(gold),
        "macro_f1": macro_f1(gold, pred, labels),
        "level1_f1": level1_f1(gold, pred),
        "level2_macro_f1": level2_macro_f1(gold, pred, nfr_labels),
        "accuracy": accuracy(gold, pred),
        "per_class": per_class(gold, pred, labels),
    }
    if probs is not None:
        out |= {
            "log_loss": log_loss(gold, probs, labels),
            "brier": brier(gold, probs, labels),
            "ece": ece_top_label(gold, probs, labels),
            "ece_per_class": ece_per_class(gold, probs, labels),
            "aurc": aurc(gold, probs, labels),
            "coverage_at_95": coverage_at_precision(gold, probs, labels),
        }
    return out
