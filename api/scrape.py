from __future__ import annotations

"""Scraping e persistencia vetorial otimizada."""

import asyncio
import logging
import re
import uuid
from typing import Any

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright

from .config import Settings
from .db import insert_documents
from .embeddings import embed_texts_async

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Remove espacos extras e caracteres invalidos para UTF-8/PostgreSQL."""
    # Remove bytes nulos que causam erro no PostgreSQL
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Divide texto em chunks menores com overlap para melhor recuperacao.
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho maximo de cada chunk (default 500)
        chunk_overlap: Sobreposicao entre chunks para manter contexto (default 100)
    
    Note:
        Chunks menores (400-600) geralmente performam melhor em RAG para:
        - Textos densos com muita informacao
        - Buscas especificas
        
        Chunks maiores (800-1200) sao melhores para:
        - Textos narrativos
        - Perguntas que precisam de mais contexto
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


async def chunk_and_store(
    pool,
    settings: Settings,
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Gera embeddings e armazena chunks com processamento assincrono em lote.
    
    Otimizacoes:
    - Usa chunk_size e chunk_overlap das configuracoes
    - Embeddings gerados de forma assincrona (nao bloqueia event loop)
    - Processamento em lotes para controle de memoria
    - Insercao em batch no banco
    """
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    total_chunks = len(chunks)
    logger.info(f"Iniciando processamento de {total_chunks} chunks para fonte: {source}")
    
    # Lotes maiores para embeddings (mais eficiente)
    batch_size = 64
    stored_count = 0
    
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        try:
            # Usa versao assincrona para nao bloquear
            embeddings = await embed_texts_async(settings.embedding_model, batch_chunks)
            
            items = [
                {
                    "id": uuid.uuid4(),
                    "source": source,
                    "content": chunk,
                    "metadata": metadata or {},
                    "embedding": embedding,
                }
                for chunk, embedding in zip(batch_chunks, embeddings)
            ]
            
            await insert_documents(pool, items)
            stored_count += len(items)
            
            # Log de progresso a cada 25%
            progress = (i + len(batch_chunks)) / total_chunks * 100
            if progress % 25 < (batch_size / total_chunks * 100):
                logger.info(f"Progresso: {progress:.0f}% ({stored_count}/{total_chunks} chunks)")
                
        except Exception as e:
            logger.error(f"Erro ao processar lote {i//batch_size + 1}: {e}")
            # Continua para o proximo lote mesmo se um falhar
            continue
            
    logger.info(f"Finalizado: {stored_count}/{total_chunks} chunks armazenados com sucesso.")
    return stored_count


def _sync_scrape(url: str, user_agent: str) -> str:
    """Executa scraping síncrono com Playwright (para rodar em thread separada)."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent=user_agent)
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        html = page.content()
        browser.close()
    return html


async def scrape_and_store(pool, settings: Settings, url: str | None = None) -> int:
    """Executa scraping da URL fornecida ou da configurada."""
    target_url = url or settings.scrape_url
    if not target_url:
        raise ValueError("URL não fornecida e SCRAPE_URL não configurada.")
    
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    # Executa Playwright síncrono em thread separada para evitar problema do event loop no Windows
    html = await asyncio.to_thread(_sync_scrape, target_url, user_agent)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = _clean_text(soup.get_text(separator=" "))
    return await chunk_and_store(
        pool,
        settings,
        text,
        source="scrape",
        metadata={"url": target_url},
    )
