"""Budget guard: refuses API calls that would exceed the configured spending limits.

Spending is recorded in an append-only JSONL ledger kept apart from the inference cache, so
clearing the cache never resets what has been spent.
"""

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from reqcore.paths import configs_dir, repo_root
from reqcore.pricing import PRICES


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetConfig:
    total_usd: float
    buckets: dict[str, float]

    @classmethod
    def load(cls, path: Path | None = None) -> "BudgetConfig":
        with (path or configs_dir() / "budget.yaml").open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(total_usd=float(raw["total_usd"]), buckets=dict(raw["buckets"]))


class Budget:
    def __init__(self, config: BudgetConfig | None = None, ledger_path: Path | None = None):
        self.config = config or BudgetConfig.load()
        self.ledger_path = ledger_path or repo_root() / "cache" / "spend_ledger.jsonl"
        self._lock = threading.Lock()

    def spent_by_bucket(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        if self.ledger_path.exists():
            with self.ledger_path.open(encoding="utf-8") as fh:
                for line in fh:
                    entry = json.loads(line)
                    totals[entry["bucket"]] += entry["cost_usd"]
        return dict(totals)

    def remaining(self) -> dict[str, float]:
        spent = self.spent_by_bucket()
        remaining = {b: cap - spent.get(b, 0.0) for b, cap in self.config.buckets.items()}
        remaining["total"] = self.config.total_usd - sum(spent.values())
        return remaining

    def ensure_affordable(self, model: str, estimated_usd: float) -> None:
        """Raise BudgetExceededError if `estimated_usd` more for `model` would break a limit."""
        bucket = PRICES[model].budget_bucket
        remaining = self.remaining()
        if estimated_usd > remaining[bucket]:
            raise BudgetExceededError(
                f"{model}: estimated ${estimated_usd:.4f} exceeds the remaining "
                f"${remaining[bucket]:.4f} of bucket '{bucket}'."
            )
        if estimated_usd > remaining["total"]:
            raise BudgetExceededError(
                f"{model}: estimated ${estimated_usd:.4f} exceeds the remaining "
                f"${remaining['total']:.4f} of the total budget."
            )

    def record(self, model: str, cost_usd: float, cache_key: str) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "model": model,
            "bucket": PRICES[model].budget_bucket,
            "cost_usd": cost_usd,
            "cache_key": cache_key,
        }
        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
