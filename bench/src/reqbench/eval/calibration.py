"""Post-hoc calibration fitted on dev out-of-fold probabilities only (plan §6.4).

Candidates, in order of simplicity (a tie in log loss goes to the simpler one):

- multiclass (a Choice or a whole system): identity, temperature (one scale on log p),
  vector scaling (per-class scale and bias on log p, Guo et al. 2017) and one-vs-rest
  isotonic regression (renormalized);
- binary (a Noul, given as two columns [P(false), P(true)]): identity, Platt (logistic on the
  logit) and isotonic.

Log p is taken after flooring the probabilities at `EPS`: Jev rounds its probabilities to two
decimals (exact zeros) and B5's are frequencies over 5 samples. Identity is never floored, so
its log loss is the raw one. Vector scaling and isotonic can change the argmax; temperature,
Platt and identity cannot.

Selection: pooled log loss of each candidate over the project-grouped dev folds (fit on four
folds, score the fifth). `nested_oof` wraps that selection in an outer loop, so the calibrated
dev probabilities used for the reported ECE and for the review thresholds never come from a
calibrator (or a choice of calibrator) that saw their fold.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-3
VECTOR_L2 = 1e-2  # mild pull of vector scaling towards identity (w = 1, b = 0)
MULTICLASS = ("identity", "temperature", "vector", "isotonic")
BINARY = ("identity", "platt", "isotonic")


def _log(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, EPS, 1.0))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log likelihood with the same 1e-15 floor as metrics.log_loss."""
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))))


class Calibrator:
    name: str

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "Calibrator":
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class Identity(Calibrator):
    name = "identity"

    def transform(self, probs):
        return np.asarray(probs, dtype=float)


class Temperature(Calibrator):
    name = "temperature"

    def fit(self, probs, y):
        z = _log(probs)

        def loss(log_t: float) -> float:
            return -np.mean(log_softmax(z / np.exp(log_t), axis=1)[np.arange(len(y)), y])

        self.t_ = float(np.exp(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x))
        return self

    def transform(self, probs):
        return softmax(_log(probs) / self.t_, axis=1)

    def params(self):
        return {"T": self.t_}


class Vector(Calibrator):
    name = "vector"

    def fit(self, probs, y):
        z, k = _log(probs), probs.shape[1]
        onehot = np.eye(k)[y]

        def loss(theta):
            w, b = theta[:k], theta[k:]
            logp = log_softmax(z * w + b, axis=1)
            reg = VECTOR_L2 * (np.sum((w - 1) ** 2) + np.sum(b**2))
            grad_logits = (np.exp(logp) - onehot) / len(y)
            grad_w = np.sum(grad_logits * z, axis=0) + 2 * VECTOR_L2 * (w - 1)
            grad_b = np.sum(grad_logits, axis=0) + 2 * VECTOR_L2 * b
            return -np.mean(np.sum(logp * onehot, axis=1)) + reg, np.concatenate([grad_w, grad_b])

        theta0 = np.concatenate([np.ones(k), np.zeros(k)])
        res = minimize(loss, theta0, jac=True, method="L-BFGS-B")
        self.w_, self.b_ = res.x[:k], res.x[k:]
        return self

    def transform(self, probs):
        return softmax(_log(probs) * self.w_ + self.b_, axis=1)

    def params(self):
        return {"w": self.w_.tolist(), "b": self.b_.tolist()}


class Isotonic(Calibrator):
    """One-vs-rest isotonic per column, then rows renormalized (uniform if all are 0)."""

    name = "isotonic"

    def fit(self, probs, y):
        self.models_ = [
            IsotonicRegression(y_min=EPS, y_max=1.0, out_of_bounds="clip").fit(
                probs[:, j], (y == j).astype(float))
            for j in range(probs.shape[1])
        ]  # fmt: skip
        return self

    def transform(self, probs):
        out = np.column_stack([m.predict(probs[:, j]) for j, m in enumerate(self.models_)])
        total = out.sum(axis=1, keepdims=True)
        return np.divide(out, total, out=np.full_like(out, 1 / out.shape[1]), where=total > 0)

    def params(self):
        return {"thresholds": [m.X_thresholds_.tolist() for m in self.models_],
                "values": [m.y_thresholds_.tolist() for m in self.models_]}  # fmt: skip


class Platt(Calibrator):
    """Binary: P(true) = sigmoid(a * logit(p) + b), p floored at EPS."""

    name = "platt"

    def fit(self, probs, y):
        x = (_log(probs)[:, 1] - _log(probs)[:, 0])[:, None]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
        self.a_, self.b_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, probs):
        x = _log(probs)[:, 1] - _log(probs)[:, 0]
        p = 1 / (1 + np.exp(-(self.a_ * x + self.b_)))
        return np.column_stack([1 - p, p])

    def params(self):
        return {"a": self.a_, "b": self.b_}


FACTORIES = {"identity": Identity, "temperature": Temperature, "vector": Vector,
             "isotonic": Isotonic, "platt": Platt}  # fmt: skip


def candidates_for(n_classes: int) -> tuple[str, ...]:
    return BINARY if n_classes == 2 else MULTICLASS


def cv_log_loss(probs: np.ndarray, y: np.ndarray, folds: np.ndarray, name: str) -> float:
    """Pooled log loss of calibrator `name` fitted on all folds but one, scored on that one."""
    out = np.empty_like(probs, dtype=float)
    for f in np.unique(folds):
        held = folds == f
        out[held] = FACTORIES[name]().fit(probs[~held], y[~held]).transform(probs[held])
    return nll(out, y)


def select(probs: np.ndarray, y: np.ndarray, folds: np.ndarray,
           candidates: Sequence[str] | None = None) -> tuple[str, dict[str, float]]:  # fmt: skip
    candidates = candidates or candidates_for(probs.shape[1])
    scores = {name: cv_log_loss(probs, y, folds, name) for name in candidates}
    best = min(candidates, key=lambda name: (round(scores[name], 9), candidates.index(name)))
    return best, scores


@dataclass
class NestedResult:
    calibrated: np.ndarray  # out-of-fold calibrated probabilities (outer loop)
    chosen_per_fold: dict[int, str]
    final_name: str  # chosen with all dev folds
    final_scores: dict[str, float]  # CV log loss of every candidate on all dev folds
    final: Calibrator  # fitted on all dev rows


def nested_oof(probs: np.ndarray, y: np.ndarray, folds: np.ndarray,
               candidates: Sequence[str] | None = None,
               fit_mask: np.ndarray | None = None) -> NestedResult:  # fmt: skip
    """`fit_mask` limits fitting and selection to some rows (e.g. J4's `nfr_subtype`, a
    conditional question, uses only gold-NFR rows); every row is still transformed."""
    probs, y, folds = np.asarray(probs, dtype=float), np.asarray(y), np.asarray(folds)
    mask = np.ones(len(y), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    out = np.empty_like(probs)
    chosen = {}
    for f in np.unique(folds):
        held, train = folds == f, (folds != f) & mask
        name, _ = select(probs[train], y[train], folds[train], candidates)
        chosen[int(f)] = name
        out[held] = FACTORIES[name]().fit(probs[train], y[train]).transform(probs[held])
    final_name, scores = select(probs[mask], y[mask], folds[mask], candidates)
    final = FACTORIES[final_name]().fit(probs[mask], y[mask])
    return NestedResult(out, chosen, final_name, scores, final)


def to_dict(cal: Calibrator) -> dict:
    return {"name": cal.name, "params": cal.params()}


def from_dict(spec: dict) -> Calibrator:
    """Rebuild a frozen calibrator (identity, temperature, vector, platt or isotonic)."""
    cal, p = FACTORIES[spec["name"]](), spec["params"]
    if spec["name"] == "temperature":
        cal.t_ = float(p["T"])
    elif spec["name"] == "vector":
        cal.w_, cal.b_ = np.array(p["w"]), np.array(p["b"])
    elif spec["name"] == "platt":
        cal.a_, cal.b_ = float(p["a"]), float(p["b"])
    elif spec["name"] == "isotonic":
        cal.models_ = []
        for xs, ys in zip(p["thresholds"], p["values"], strict=True):
            m = IsotonicRegression(y_min=EPS, y_max=1.0, out_of_bounds="clip")
            m.fit(np.array(xs), np.array(ys))  # refit on its own knots: same step function
            cal.models_.append(m)
    return cal
