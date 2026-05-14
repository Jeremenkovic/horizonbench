"""Mock document server for research_synth task.

Serves plain-text documents from a directory over HTTP.
Intended to be started in a background thread or as a sidecar process.

Endpoints
---------
GET /docs                   → JSON list of available document IDs
GET /docs/{doc_id}          → plain-text document content
GET /health                 → {"status": "ok"}
"""

from __future__ import annotations

import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse


def create_app(docs_dir: Path) -> FastAPI:
    app = FastAPI(title="HorizonBench Mock Document Server")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/documents")
    def list_docs():
        if not docs_dir.exists():
            return JSONResponse([])
        doc_ids = sorted(p.stem for p in docs_dir.glob("*.txt"))
        return JSONResponse(doc_ids)

    @app.get("/documents/{doc_id}")
    def get_doc(doc_id: str):
        path = docs_dir / f"{doc_id}.txt"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Document {doc_id!r} not found")
        return PlainTextResponse(path.read_text())

    return app


class DocServerThread:
    """Run the document server in a daemon background thread."""

    def __init__(self, docs_dir: Path, host: str = "127.0.0.1", port: int = 8765):
        self.docs_dir = docs_dir
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        app = create_app(self.docs_dir)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait briefly for startup
        import time
        time.sleep(0.5)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True

    def __enter__(self) -> "DocServerThread":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
