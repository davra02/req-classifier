"""B4: fine-tuned microsoft/deberta-v3-small (plan §6.2), trained locally (MPS, fp32).

Fixed, standard recipe (AdamW, lr 3e-5, weight decay 0.01, batch 16, linear decay with 10 %
warmup over `max_epochs`, plain cross-entropy, max length 64 tokens; the longest dev text
has 53). The only tuned value is the number of epochs, chosen without seeing the held-out
fold: inside `fit`, the lowest-numbered training fold is an inner validation set; a model is
trained on the other folds and scored (macro-F1) after every epoch; the best epoch count
(ties -> fewer epochs) is then used to retrain on all training folds. The learning-rate
schedule is always defined over `max_epochs`, so the retrain follows the same trajectory and
simply stops at the chosen epoch.

Seeds fix initialization, shuffling and dropout; MPS kernels are not bit-for-bit
deterministic, which is why the plan reports the spread over 5 seeds.
"""

import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

from reqbench.eval import metrics  # noqa: E402
from reqbench.models.baselines import LABELS  # noqa: E402

DEBERTA = "microsoft/deberta-v3-small"
DEBERTA_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"  # MIT, verified 2026-11-11
SEEDS: tuple[int, ...] = (20261010, 20261011, 20261012, 20261013, 20261014)


@dataclass(frozen=True)
class Recipe:
    lr: float = 3e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 15
    warmup: float = 0.1
    max_length: int = 64
    eval_batch_size: int = 64


def device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class DebertaClassifier:
    """B4 with the `fit(texts, labels, groups)` / `predict_proba(texts)` interface of B2/B3."""

    name, has_proba = "B4_deberta_v3_small", True

    def __init__(self, seed: int, recipe: Recipe | None = None, log=None) -> None:
        self.seed, self.recipe = seed, recipe or Recipe()
        self.log = log or (lambda message: None)
        self.device = device()
        self._tokenizer = None

    # -- helpers ---------------------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(DEBERTA, revision=DEBERTA_REVISION)
        return self._tokenizer

    def _batch(self, texts: Sequence[str]):
        encoded = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.recipe.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _new_model(self):
        from transformers import AutoModelForSequenceClassification
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # the "newly initialized classifier" notice
        set_seed(self.seed)
        import torch

        # transformers >= 5 defaults to the checkpoint dtype (fp16 here); the plan trains in fp32.
        model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA, revision=DEBERTA_REVISION, num_labels=len(LABELS), dtype=torch.float32,
            id2label=dict(enumerate(LABELS)), label2id={k: i for i, k in enumerate(LABELS)},
        )  # fmt: skip
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("DeBERTa did not load in fp32")
        return model.to(self.device)

    def _train(self, texts, labels, epochs: int, val_texts=None, val_labels=None):
        """Train for `epochs` (schedule over max_epochs); macro-F1 on val after each epoch."""
        import torch
        from transformers import get_linear_schedule_with_warmup

        r = self.recipe
        model = self._new_model()
        y = torch.tensor([LABELS.index(label) for label in labels])
        steps_per_epoch = -(-len(texts) // r.batch_size)
        total = steps_per_epoch * r.max_epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
        schedule = get_linear_schedule_with_warmup(optimizer, int(r.warmup * total), total)
        generator = torch.Generator().manual_seed(self.seed)
        scores = []
        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(texts), generator=generator).tolist()
            for start in range(0, len(order), r.batch_size):
                idx = order[start:start + r.batch_size]
                out = model(**self._batch([texts[i] for i in idx]),
                            labels=y[idx].to(self.device))  # fmt: skip
                out.loss.backward()
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
            if val_texts is not None:
                pred = self._proba(model, val_texts).argmax(axis=1)
                scores.append(metrics.macro_f1(val_labels, [LABELS[i] for i in pred], LABELS))
        return model, scores

    def _proba(self, model, texts) -> np.ndarray:
        import torch

        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.recipe.eval_batch_size):
                logits = model(**self._batch(texts[start:start + self.recipe.eval_batch_size]))
                chunks.append(torch.softmax(logits.logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks)

    # -- interface -------------------------------------------------------------------------

    def fit(self, texts, labels, groups) -> "DebertaClassifier":
        texts, labels, groups = list(texts), list(labels), np.asarray(groups)
        started = time.perf_counter()
        val_group = int(np.min(groups))
        inner = groups != val_group
        _, scores = self._train(
            [t for t, keep in zip(texts, inner, strict=True) if keep],
            [y for y, keep in zip(labels, inner, strict=True) if keep],
            self.recipe.max_epochs,
            [t for t, keep in zip(texts, inner, strict=True) if not keep],
            [y for y, keep in zip(labels, inner, strict=True) if not keep],
        )
        best = int(np.argmax(scores)) + 1  # argmax returns the first maximum: fewer epochs
        self.model_, _ = self._train(texts, labels, best)
        self.params_ = {
            "seed": self.seed, "epochs": best, "inner_val_fold": val_group,
            "inner_val_macro_f1": [round(s, 4) for s in scores],
            "fit_seconds": round(time.perf_counter() - started, 1),
            "device": str(self.device), "recipe": asdict(self.recipe),
        }  # fmt: skip
        self.log(f"seed {self.seed}: {best} epochs (inner F1 {max(scores):.3f}), "
                 f"{self.params_['fit_seconds']} s")  # fmt: skip
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self._proba(self.model_, list(texts))

    def predict(self, texts) -> list[str]:
        return [LABELS[i] for i in self.predict_proba(texts).argmax(axis=1)]
