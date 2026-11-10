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
    values = np.array([metric(idx) for idx in resamples(n, n_resamples, seed)])
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return {"value": float(metric(np.arange(n))), "ci95": [float(low), float(high)]}


def paired_difference(metric_a: Callable[[np.ndarray], float],
                      metric_b: Callable[[np.ndarray], float], n: int,
                      n_resamples: int = N_RESAMPLES, seed: int = SEED) -> dict:  # fmt: skip
    """CI of metric_a - metric_b on the same resamples."""
    return ci(lambda idx: metric_a(idx) - metric_b(idx), n, n_resamples, seed)
