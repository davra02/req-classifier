"""Local annotation app for the fresh set: `uv run reqbench fresh annotate`.

A stdlib HTTP server bound to 127.0.0.1 serves one page and a small JSON API that writes the
chosen leaf into `annotator_a` or `annotator_b` of data/fresh/requirements.csv. To keep the
two annotations independent, the API only ever returns the caller's own label: never the
other annotator's column nor the adjudication.
"""

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from reqcore.paths import data_dir
from reqcore.questions import load_taxonomy

DISCARD = "discard"
COLUMNS = {w: f"annotator_{w}" for w in "abcde"}
ASSIGN_COLUMN = "second_annotator"
PAGE = Path(__file__).with_name("annotate.html")


class AnnotationStore:
    """Read and write one annotator column of the fresh requirements CSV."""

    def __init__(self, requirements: Path, documents: Path) -> None:
        self.path = requirements
        self.lock = threading.Lock()
        taxonomy = load_taxonomy()
        self.labels = [
            {"key": leaf, **taxonomy.classes.get(leaf, {})} for leaf in taxonomy.leaves
        ] + [{"key": DISCARD, "what": "Not a requirement (heading, note, process rule...)."}]
        self.allowed = {label["key"] for label in self.labels}
        docs = pd.read_csv(documents, dtype=str, keep_default_na=False)
        self.doc_urls = dict(zip(docs["doc_id"], docs["url"], strict=True))

    def _read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, dtype=str, keep_default_na=False)

    def _rows(self, annotator: str) -> pd.DataFrame:
        """All rows for A; for B and C only their assigned subsample."""
        df = self._read()
        if annotator == "a":
            return df.reset_index(drop=True)
        if COLUMNS[annotator] not in df or ASSIGN_COLUMN not in df:
            return df.iloc[0:0]
        return df[df[ASSIGN_COLUMN] == annotator].reset_index(drop=True)

    def _item(self, df: pd.DataFrame, index: int, column: str) -> dict:
        row = df.iloc[index]
        done = int((df[column] != "").sum())
        path, _, line = row["location"].partition("#")
        tree = self.doc_urls.get(row["doc_id"], "")
        source = f"{tree.replace('/tree/', '/blob/', 1)}/{path}#{line}" if tree else ""
        return {
            "index": index, "total": len(df), "done": done, "req_id": row["req_id"],
            "doc_id": row["doc_id"], "location": row["location"], "text": row["text"],
            "label": row[column], "source_url": source,
        }  # fmt: skip

    def item(self, annotator: str, index: int | None = None) -> dict:
        """The row at `index`, or the first row this annotator has not labelled yet."""
        column = COLUMNS[annotator]
        df = self._rows(annotator)
        if df.empty:
            raise ValueError(f"no rows assigned to annotator {annotator!r}")
        if index is None:
            pending = df.index[df[column] == ""]
            if len(pending) == 0:
                return {"complete": True, "total": len(df), "done": len(df)}
            index = int(pending[0])
        if not 0 <= index < len(df):
            raise ValueError("index out of range")
        return self._item(df, index, column)

    def save(self, annotator: str, req_id: str, label: str) -> None:
        if label not in self.allowed:
            raise ValueError(f"unknown label {label!r}")
        column = COLUMNS[annotator]
        with self.lock:
            df = self._read()
            allowed = self._rows(annotator)["req_id"]
            match = df.index[(df["req_id"] == req_id) & df["req_id"].isin(allowed)]
            if len(match) != 1:
                raise ValueError(f"unknown req_id {req_id!r} for annotator {annotator!r}")
            df.loc[match[0], column] = label
            tmp = self.path.with_suffix(".csv.part")
            df.to_csv(tmp, index=False)
            tmp.replace(self.path)


def _handler(store: AnnotationStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/labels":
                    self._json(store.labels)
                elif url.path == "/api/item":
                    index = int(query["index"]) if "index" in query else None
                    self._json(store.item(query["annotator"], index))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/label":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                store.save(body["annotator"], body["req_id"], body["label"])
                self._json(store.item(body["annotator"]))
            except (KeyError, ValueError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(port: int = 8765, open_browser: bool = True) -> None:
    fresh = data_dir() / "fresh"
    store = AnnotationStore(fresh / "requirements.csv", fresh / "documents.csv")
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(store))
    url = f"http://127.0.0.1:{port}/"
    print(f"Annotation app at {url} (Ctrl+C to stop). Writing to {store.path}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
