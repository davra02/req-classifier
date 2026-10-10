"""Per-token prices used for cost accounting. Update `AS_OF` whenever a price changes."""

from dataclasses import dataclass

AS_OF = "2026-10-10"
BATCH_DISCOUNT = 0.5  # Anthropic Message Batches API


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    budget_bucket: str


PRICES: dict[str, Price] = {
    # TypeSafe bills input tokens only; output tokens are free.
    "jev-1.13.0": Price(input_per_mtok=0.042, output_per_mtok=0.0, budget_bucket="jev"),
    # Haiku 5.5 price applies to prompts up to 100K tokens; ours are ~1K.
    "claude-haiku-5-5": Price(input_per_mtok=0.10, output_per_mtok=0.50, budget_bucket="haiku"),
    "claude-opus-5-5": Price(input_per_mtok=4.00, output_per_mtok=20.00, budget_bucket="opus"),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
    price = PRICES[model]
    cost = (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
    return cost * BATCH_DISCOUNT if batch else cost
