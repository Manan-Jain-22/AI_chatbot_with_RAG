from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from src.config import (
    WHATSAPP_ACCESS_TOKEN,
    WHATSAPP_API_VERSION,
    WHATSAPP_PHONE_NUMBER_ID,
    validate_whatsapp_config,
)


GRAPH_API_BASE_URL = "https://graph.facebook.com"


def send_whatsapp_text_message_impl(
    recipient_phone_number: str,
    message: str,
    preview_url: bool = False,
) -> Dict[str, Any]:
    """
    Send a WhatsApp text message through Meta's WhatsApp Cloud API.
    """
    validate_whatsapp_config()

    url = (
        f"{GRAPH_API_BASE_URL}/{WHATSAPP_API_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone_number,
        "type": "text",
        "text": {"body": message[:4096], "preview_url": preview_url},
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_text_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract inbound text messages from a WhatsApp webhook payload.
    """
    extracted: List[Dict[str, str]] = []

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                if message.get("type") != "text":
                    continue

                sender = message.get("from")
                text = message.get("text", {}).get("body")
                message_id = message.get("id", "")

                if sender and text:
                    extracted.append(
                        {
                            "from": sender,
                            "text": text,
                            "message_id": message_id,
                        }
                    )

    return extracted


def format_whatsapp_rag_response(answer: str, sources: Optional[List[str]] = None) -> str:
    source_text = ", ".join(sources or [])
    if source_text and "Sources:" not in answer:
        return f"{answer}\n\nSources: {source_text}"
    return answer
