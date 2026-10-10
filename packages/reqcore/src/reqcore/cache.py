"""SQLite cache of model inferences, keyed by a hash of everything that determines the input.

Jev and the LLMs are not deterministic, so a cached entry is one recorded draw. Deliberate
repeats use a different `sample_index`, which is part of the key.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reqcore.hashing import content_hash
from reqcore.paths import cache_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inference (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_requested TEXT NOT NULL,
    model_reported TEXT,
    question_set_hash TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    batch INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
)
"""


def cache_key(
    provider: str, model: str, question_set_hash: str, state: Any, sample_index: int
) -> str:
    return content_hash(
        {
            "provider": provider,
            "model": model,
            "question_set_hash": question_set_hash,
            "state": state,
            "sample_index": sample_index,
        }
    )


@dataclass(frozen=True)
class CachedInference:
    key: str
    model_reported: str | None
    response: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None


class InferenceCache:
    def __init__(self, path: Path | None = None):
        self.path = path or cache_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> CachedInference | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model_reported, response_json, input_tokens, output_tokens, cost_usd,"
                " latency_ms FROM inference WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CachedInference(key, row[0], json.loads(row[1]), *row[2:])

    def put(
        self,
        *,
        key: str,
        provider: str,
        model_requested: str,
        model_reported: str | None,
        question_set_hash: str,
        sample_index: int,
        request: dict[str, Any],
        response: dict[str, Any],
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
        latency_ms: float | None,
        batch: bool = False,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO inference VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    provider,
                    model_requested,
                    model_reported,
                    question_set_hash,
                    sample_index,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_ms,
                    int(batch),
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
