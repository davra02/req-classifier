"""Sentence embeddings with a pinned model revision, cached on disk by text hash.

Computed on CPU (more reproducible than MPS) and L2-normalized. The cache lives in
cache/embeddings/ (git-ignored) and is keyed by model, revision and `text_hash`.
"""

import numpy as np
from reqcore.normalize import text_hash
from reqcore.paths import repo_root

BGE_SMALL = "BAAI/bge-small-en-v1.5"
BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # MIT, verified 2026-11-10


class CachedEmbedder:
    def __init__(self, model: str = BGE_SMALL, revision: str = BGE_SMALL_REVISION) -> None:
        self.model_name, self.revision = model, revision
        name = model.replace("/", "__")
        self.path = repo_root() / "cache" / "embeddings" / f"{name}@{revision[:12]}.npz"
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as stored:
                self._vectors = {key: stored[key] for key in stored.files}
        self._model = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.revision,
                                              device="cpu")  # fmt: skip
        return self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)  # fmt: skip

    def __call__(self, texts: list[str]) -> np.ndarray:
        keys = [text_hash(t) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts, strict=True)
                          if k not in self._vectors}.items())  # fmt: skip
        if missing:
            vectors = self._encode([t for _, t in missing])
            self._vectors.update({k: v for (k, _), v in zip(missing, vectors, strict=True)})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, **self._vectors)
        return np.stack([self._vectors[k] for k in keys])
