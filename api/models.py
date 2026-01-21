from __future__ import annotations

"""Modelos de resposta da API."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResponse:
    """Resposta do endpoint /chat."""
    session_id: str
    answer: str
    sources: list[dict[str, Any]]


@dataclass
class ScrapeResponse:
    """Resposta do endpoint /scrape."""
    url: str
    chunks: int
