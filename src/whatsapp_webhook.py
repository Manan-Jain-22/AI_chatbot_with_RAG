from __future__ import annotations

import logging

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from src.config import WHATSAPP_VERIFY_TOKEN
from src.rag_chain import answer_question
from src.vector_store import index_exists
from src.whatsapp_client import (
    extract_text_messages,
    format_whatsapp_rag_response,
    send_whatsapp_text_message_impl,
)


logger = logging.getLogger(__name__)


async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)

    return PlainTextResponse("Forbidden", status_code=403)


async def receive_message(request: Request):
    payload = await request.json()
    messages = extract_text_messages(payload)

    for message in messages:
        sender = message["from"]
        question = message["text"]

        try:
            if not index_exists():
                reply = "The document index is not ready yet. Build the FAISS index first."
            else:
                result = answer_question(question)
                reply = format_whatsapp_rag_response(result["answer"], result["sources"])

            send_whatsapp_text_message_impl(sender, reply)
        except Exception as exc:
            logger.exception("Failed to handle WhatsApp message")
            try:
                send_whatsapp_text_message_impl(sender, f"Sorry, I hit an error: {exc}")
            except Exception:
                logger.exception("Failed to send WhatsApp error reply")

    return JSONResponse({"status": "ok", "messages_received": len(messages)})


app = Starlette(
    debug=False,
    routes=[
        Route("/webhook/whatsapp", verify_webhook, methods=["GET"]),
        Route("/webhook/whatsapp", receive_message, methods=["POST"]),
    ],
)
