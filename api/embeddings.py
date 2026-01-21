from __future__ import annotations

"""Helpers para embeddings locais - usa HuggingFaceEmbeddings do rag.py."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    """Gera embeddings usando HuggingFaceEmbeddings (mesmo modelo da busca)."""
    if not texts:
        return []
    
    from .rag import get_embeddings
    embeddings = get_embeddings(model_name)
    return embeddings.embed_documents(texts)


async def embed_texts_async(model_name: str, texts: list[str]) -> list[list[float]]:
    """Gera embeddings de forma assincrona (nao bloqueia event loop)."""
    if not texts:
        return []
    
    # Executa em thread pool para nao bloquear
    return await asyncio.to_thread(embed_texts, model_name, texts)


def embed_query_cached(model_name: str, text: str) -> list[float]:
    """Gera embedding para query usando cache."""
    from .rag import get_embeddings
    embeddings = get_embeddings(model_name)
    return embeddings.embed_query(text)
