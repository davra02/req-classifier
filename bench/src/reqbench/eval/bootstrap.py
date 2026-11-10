"""Bootstrap confidence intervals by resampling requirements (plan §6.3).

`metric(idx)` receives an array of row indices (a resample) and returns a float. Resamples are
drawn once per call from a seeded generator, so two systems evaluated with the same seed and n
see the same resamples (paired bootstrap).
"""

from collections.abc import Callable

import numpy as np

N_RESAMPLES = 10_000
SEED = 20261010


def resamples(n: int, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def ci(metric: Callable[[np.ndarray], float], n: int, n_resamples: int = N_RESAMPLES,
       seed: int = SEED, level: float = 0.95) -> dict:  # fmt: skip
    """Point estimate on all rows plus a percentile interval over the resamples."""
    return _ci(metric, n, _draws(metric, n, n_resamples, seed), level)


def _draws(metric, n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.array([metric(idx) for idx in resamples(n, n_resamples, seed)])


def _ci(metric, n: int, values: np.ndarray, level: float = 0.95) -> dict:
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return {"value": float(metric(np.arange(n))), "ci95": [float(low), float(high)]}


def paired_difference(metric_a: Callable[[np.ndarray], float],
                      metric_b: Callable[[np.ndarray], float], n: int,
                      n_resamples: int = N_RESAMPLES, seed: int = SEED) -> dict:  # fmt: skip
    """CI of metric_a - metric_b on the same resamples, plus a two-sided bootstrap p-value
    (twice the smaller share of resampled differences on either side of 0, capped at 1)."""

    def diff(idx: np.ndarray) -> float:
        return metric_a(idx) - metric_b(idx)

    values = _draws(diff, n, n_resamples, seed)
    out = _ci(diff, n, values)
    out["p_two_sided"] = float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0))))
    return out


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m, running, out = len(ordered), 0.0, {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    return out
