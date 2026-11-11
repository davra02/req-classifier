"""Phase 1, step 6c: the trained systems fitted once on all of dev, before the test is read.

Same code and recipes as the dev CV (reqbench.eval.cv.run_cv), trained on every dev row with
the dev folds as groups for the inner hyperparameter search: B0, B1, B2, B3 and B4 on the 449
dev rows (the examples included, as in CV training), J3 on the 437 rows Jev was queried on.
B4 is trained once per seed (5 seeds; epochs chosen on the inner validation fold, as in CV).

Models go to runs/phase1/final/ (git-ignored: B4 weights are ~0.5 GB per seed); their SHA-256
hashes and the chosen hyperparameters go to frozen/phase1/final_models.json (versioned), and
the test runner refuses a model whose file hash differs.
"""

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
from reqcore.paths import repo_root

from reqbench.eval import cv
from reqbench.eval.calibrate_dev import frozen_dir
from reqbench.models import baselines, jev
from reqbench.models.finetune import SEEDS, DebertaClassifier

B4 = "B4_deberta_v3_small"


def final_dir() -> Path:
    return repo_root() / "runs" / "phase1" / "final"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path() -> Path:
    return frozen_dir() / "final_models.json"


def _load_manifest() -> dict:
    path = _manifest_path()
    return json.loads(path.read_text()) if path.exists() else {}


def _save_manifest(manifest: dict) -> None:
    frozen_dir().mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def _pickle(name: str, model) -> dict:
    path = final_dir() / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "embed"):
        model.embed = None  # the embedder is rebuilt at load time (pinned revision)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    return {"file": str(path.relative_to(repo_root())), "sha256": file_hash(path),
            "params": {k: str(v) for k, v in model.params_.items()}}  # fmt: skip


def fit_classical(dev: pd.DataFrame | None = None) -> dict:
    """B0-B3 and J3 (seconds). Returns the manifest entries."""
    from reqbench.models.embeddings import CachedEmbedder

    dev = cv.load_dev() if dev is None else dev
    texts, labels, groups = dev["text"].tolist(), dev["leaf"].tolist(), dev["fold"].tolist()
    embedder = CachedEmbedder()
    entries = {}
    for model in (baselines.Majority(), baselines.Keywords(), baselines.TfidfLR(),
                  baselines.EmbedLR(embedder)):  # fmt: skip
        entries[model.name] = _pickle(model.name, model.fit(texts, labels, groups))

    rows = dev[~dev["is_example"]].reset_index(drop=True)
    answers = pd.read_csv(cv.runs_dir() / "jev_answers.csv")
    table = jev.features(answers, jev.ROTATIONS)
    table.index = rows.set_index("req_id").loc[table.index, "text"]
    j3 = jev.JevLR(table).fit(rows["text"].tolist(), rows["leaf"].tolist(),
                              rows["fold"].tolist())  # fmt: skip
    j3.table = None  # test features are passed at prediction time
    entries[j3.name] = _pickle(j3.name, j3)
    manifest = _load_manifest() | entries
    _save_manifest(manifest)
    return entries


def fit_b4(seeds=SEEDS, log=print) -> dict:
    """B4, one model per seed (~3 min each on the M4); finished seeds are skipped."""
    dev = cv.load_dev()
    texts, labels, groups = dev["text"].tolist(), dev["leaf"].tolist(), dev["fold"].tolist()
    entries = {}
    for k, seed in enumerate(seeds, 1):
        name = f"{B4}_s{k}"
        out = final_dir() / name
        weights = out / "model.safetensors"
        if not weights.exists():
            clf = DebertaClassifier(seed, log=log).fit(texts, labels, groups)
            clf.model_.save_pretrained(out)
            (out / "params.json").write_text(json.dumps(clf.params_, indent=1) + "\n")
        params = json.loads((out / "params.json").read_text())
        entries[name] = {"file": str(weights.relative_to(repo_root())),
                         "sha256": file_hash(weights),
                         "params": {k: str(v) for k, v in params.items()}}  # fmt: skip
        _save_manifest(_load_manifest() | {name: entries[name]})
        log(f"{name}: {params['epochs']} epochs, saved")
    return entries


def load(name: str):
    """A frozen final model, after checking its file hash against the manifest."""
    entry = _load_manifest()[name]
    path = repo_root() / entry["file"]
    if file_hash(path) != entry["sha256"]:
        raise RuntimeError(f"{name}: {path} does not match its frozen hash")
    if name.startswith(B4):
        import torch
        from transformers import AutoModelForSequenceClassification

        seed = int(entry["params"]["seed"])
        clf = DebertaClassifier(seed)
        clf.device = torch.device("cpu")
        clf.model_ = AutoModelForSequenceClassification.from_pretrained(
            path.parent, dtype=torch.float32).to(clf.device)  # fmt: skip
        return clf
    with path.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301 (our own file, hash-checked above)
    if name == baselines.EmbedLR.name:
        from reqbench.models.embeddings import CachedEmbedder

        model.embed = CachedEmbedder()
    return model
