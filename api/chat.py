from __future__ import annotations

"""Orquestracao de RAG via LangChain com cache e retrievers ."""

import logging
import uuid
from typing import Any

from .config import Settings
from .db import insert_history
from .rag import (
    generate_answer as rag_generate_answer,
    retrieve_context as rag_retrieve_context,
    get_rag_chain,
)

logger = logging.getLogger(__name__)


async def generate_answer(
    settings: Settings,
    question: str,
    history: list[dict[str, Any]],
    docs: list[dict[str, Any]],
) -> str:
    """Gera resposta usando RAG com LangChain.
    
    Otimizacoes aplicadas:
    - Cache de respostas LLM em memoria
    - Prompt com LCEL
    - Historico limitado para evitar contexto muito grande
    """
    return await rag_generate_answer(settings, question, history, docs)


async def retrieve_context(
    pool,
    settings: Settings,
    question: str,
) -> list[dict[str, Any]]:
    """Busca documentos relevantes com retriever.
    
    Otimizacoes aplicadas:
    - Cache de embeddings de queries
    - Busca com score threshold
    - Reranking por relevancia
    """
    return await rag_retrieve_context(pool, settings, question)


async def run_rag_pipeline(
    pool,
    settings: Settings,
    question: str,
    history: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Executa pipeline RAG completo.
    
    Esta funcao combina retrieve + generate em uma unica chamada,
    evitando overhead de multiplas inicializacoes.
    
    Returns:
        Tupla com (resposta, documentos recuperados)
    """
    chain = get_rag_chain(settings)
    answer, docs = await chain.run(pool, question, history)
    
    # Converte Documents para dicts
    docs_dict = [
        {
            "source": doc.metadata.get("source", ""),
            "content": doc.page_content,
            "score": doc.metadata.get("score", 0),
            "metadata": doc.metadata,
        }
        for doc in docs
    ]
    
    return answer, docs_dict


async def persist_history(
    pool,
    session_id: str,
    question: str,
    answer: str,
) -> None:
    """Salva a conversa no banco para persistencia."""
    entries = [
        {
            "id": uuid.uuid4(),
            "session_id": session_id,
            "role": "user",
            "content": question,
        },
        {
            "id": uuid.uuid4(),
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
        },
    ]
    await insert_history(pool, entries)
