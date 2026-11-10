"""Cheap phase-1 baselines (plan §6.2): B0 majority, B1 keywords, B2 TF-IDF + LR, B3 bge + LR.

Every model has `fit(texts, labels, groups)` and `predict(texts)`; models with probabilities
also have `predict_proba(texts)`, whose columns follow `LABELS` (the taxonomy's leaf order).
`groups` are the training rows' dev folds: hyperparameters are chosen by an inner
leave-one-fold-out CV, so inner validation is grouped by project too.
"""

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from reqcore.paths import repo_root
from reqcore.questions import load_taxonomy, load_yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

LABELS: tuple[str, ...] = load_taxonomy().leaves
KEYWORDS_VERSION = "type_keywords_v1"


def _aligned(proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Reorder probability columns to LABELS (a class unseen in training gets 0)."""
    out = np.zeros((proba.shape[0], len(LABELS)))
    for j, label in enumerate(classes):
        out[:, LABELS.index(label)] = proba[:, j]
    return out


class Majority:
    """B0: the most frequent training label (ties broken by taxonomy order)."""

    name, has_proba = "B0_majority", False

    def fit(self, texts, labels, groups=None) -> "Majority":
        counts = Counter(labels)
        self.label_ = max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))
        self.params_ = {"label": self.label_}
        return self

    def predict(self, texts) -> list[str]:
        return [self.label_] * len(texts)


def _keyword_pattern(term: str) -> str:
    stem = term.rstrip("*")
    body = r"\s+".join(re.escape(word) for word in stem.split())
    return rf"(?<![\w-]){body}" + (r"[\w-]*" if term.endswith("*") else r"(?![\w-])")


class Keywords:
    """B1: keyword lists from rules/<version>.yaml; fixed priority; no match -> default."""

    name, has_proba = "B1_keywords", False

    def __init__(self, version: str = KEYWORDS_VERSION) -> None:
        spec = load_yaml(version, repo_root() / "rules")
        self.version = version
        self.default = spec["default"]
        self.patterns = [
            (label, re.compile("|".join(map(_keyword_pattern, spec["keywords"][label])),
                               re.IGNORECASE))
            for label in spec["priority"]
        ]  # fmt: skip
        self.params_ = {"version": version}

    def fit(self, texts, labels, groups=None) -> "Keywords":
        return self  # the lists are fixed; nothing is learned per fold

    def classify(self, text: str) -> tuple[str, str | None]:
        """(label, matched keyword or None)."""
        for label, pattern in self.patterns:
            match = pattern.search(text)
            if match:
                return label, match.group(0)
        return self.default, None

    def predict(self, texts) -> list[str]:
        return [self.classify(text)[0] for text in texts]


class _GridLR:
    """Shared logic of B2/B3: grid search by inner leave-one-fold-out, scored by macro-F1."""

    has_proba = True
    grid: dict

    def _estimator(self):
        raise NotImplementedError

    def _features(self, texts):
        return list(texts)

    def fit(self, texts, labels, groups) -> "_GridLR":
        search = GridSearchCV(self._estimator(), self.grid, cv=LeaveOneGroupOut(),
                              scoring="f1_macro", n_jobs=1)  # fmt: skip
        search.fit(self._features(texts), list(labels), groups=list(groups))
        self.model_ = search.best_estimator_
        self.params_ = search.best_params_
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return _aligned(self.model_.predict_proba(self._features(texts)), self.model_.classes_)

    def predict(self, texts) -> list[str]:
        return [LABELS[i] for i in self.predict_proba(texts).argmax(axis=1)]


class TfidfLR(_GridLR):
    """B2: TF-IDF (sublinear) + multinomial logistic regression."""

    name = "B2_tfidf_lr"
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "lr__C": [0.3, 1.0, 3.0, 10.0, 30.0],
        "lr__class_weight": [None, "balanced"],
    }

    def _estimator(self):
        return Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True, lowercase=True)),
            ("lr", LogisticRegression(max_iter=5000)),
        ])  # fmt: skip


class EmbedLR(_GridLR):
    """B3: frozen bge-small-en-v1.5 sentence embeddings + logistic regression."""

    name = "B3_bge_small_lr"
    grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0], "class_weight": [None, "balanced"]}

    def __init__(self, embed) -> None:
        self.embed = embed  # callable: list[str] -> np.ndarray (cached)

    def _estimator(self):
        return LogisticRegression(max_iter=5000)

    def _features(self, texts):
        return self.embed(list(texts))
