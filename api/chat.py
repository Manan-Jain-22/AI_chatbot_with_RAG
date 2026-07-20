from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_TOP_K
from src.rag_chain import answer_question
from src.vector_store import index_exists


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))
            question = str(payload.get("question", "")).strip()
            top_k = int(payload.get("top_k", DEFAULT_TOP_K))

            if not question:
                self._send_json({"error": "Question is required."}, status=400)
                return

            if not index_exists():
                self._send_json(
                    {
                        "error": (
                            "FAISS index is not available. Build the index locally "
                            "or deploy a sanitized prebuilt index for the Vercel demo."
                        )
                    },
                    status=503,
                )
                return

            result = answer_question(question, top_k=top_k)
            self._send_json(
                {
                    "question": result["question"],
                    "topic": result["query_topic"],
                    "rewritten_question": result["rewritten_question"],
                    "answer": result["answer"],
                    "sources": result["sources"],
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
