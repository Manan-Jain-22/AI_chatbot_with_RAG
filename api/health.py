from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHAT_MODEL, EMBEDDING_MODEL, FAISS_INDEX_DIR
from src.vector_store import index_exists


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        payload = {
            "status": "ok",
            "index_ready": index_exists(),
            "index_dir": str(FAISS_INDEX_DIR),
            "chat_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
