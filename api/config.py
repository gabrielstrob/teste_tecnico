from __future__ import annotations

"""Carregamento de configuracoes via variaveis de ambiente."""

import os

from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Config central da aplicacao.
    
    Parametros de RAG otimizados:
    - top_k: Numero de documentos a recuperar (5-10 recomendado)
    - similarity_threshold: Score minimo para considerar relevante (0.3-0.5)
    - chunk_size: Tamanho dos chunks de texto (400-600 para precisao, 800-1200 para contexto)
    - chunk_overlap: Sobreposicao entre chunks (10-20% do chunk_size)
    - ef_search: Precisao da busca HNSW (40-100, maior = mais preciso)
    - max_history: Limite de mensagens de historico (5-15)
    - temperature: Criatividade do LLM (0.1-0.9, menor = mais deterministico)
    """
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    scrape_url: str | None
    scrape_on_startup: bool
    ollama_url: str
    ollama_model: str
    embedding_model: str
    embedding_dim: int
    top_k: int
    similarity_threshold: float
    # Novos parametros de otimizacao
    chunk_size: int
    chunk_overlap: int
    ef_search: int
    max_history: int
    temperature: float
    enable_cache: bool


def _get_bool(value: str | None, default: bool) -> bool:
    """Converte string para boolean com fallback."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_settings() -> Settings:
    """Carrega .env e retorna Settings com valores otimizados para RAG."""
    load_dotenv()
    # Modelo multilíngue - melhor para português
    # Alternativas: "all-MiniLM-L6-v2" (mais rápido), "intfloat/multilingual-e5-base" (melhor qualidade)
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "384"))
    return Settings(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "rag_db"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        scrape_url=os.getenv("SCRAPE_URL"),
        scrape_on_startup=_get_bool(os.getenv("SCRAPE_ON_STARTUP"), True),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.2")),
        # Parametros de otimizacao RAG
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        ef_search=int(os.getenv("EF_SEARCH", "40")),
        max_history=int(os.getenv("MAX_HISTORY", "10")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        enable_cache=_get_bool(os.getenv("ENABLE_CACHE"), True),
    )
