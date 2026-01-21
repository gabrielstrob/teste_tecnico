from __future__ import annotations

"""Camada de acesso ao PostgreSQL + PGVector."""

import json
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from .config import Settings


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Cria pool e registra suporte a vetores."""
    pool = await asyncpg.create_pool(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        min_size=1,
        max_size=10,
    )
    async with pool.acquire() as conn:
        await register_vector(conn)
    return pool


async def init_db(pool: asyncpg.Pool, embedding_dim: int) -> None:
    """Cria tabelas e indices para RAG.
    
    Usa indice HNSW ao inves de IVFFlat por ser:
    - Mais rapido para buscas (O(log n) vs O(n/lists))
    - Nao precisa de treinamento
    - Melhor para datasets que crescem dinamicamente
    """
    async with pool.acquire() as conn:
        await register_vector(conn)
        
        # Tabela de documentos
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR(%s) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            % embedding_dim
        )
        
        # Indice HNSW para busca vetorial rapida
        # m=16: numero de conexoes por no (16-64, maior = mais preciso mas mais lento)
        # ef_construction=64: tamanho da lista de candidatos na construcao (64-200)
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
        
        # Indice para filtro por source (otimiza queries com WHERE source = ...)
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_source_idx
            ON documents (source);
            """
        )
        
        # Indice GIN para busca em metadata JSONB
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_metadata_idx
            ON documents USING GIN (metadata);
            """
        )
        
        # Tabela de historico de chat
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id UUID PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        
        # Indice composto para busca eficiente por sessao
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chat_history_session_idx
            ON chat_history (session_id, created_at DESC);
            """
        )


async def insert_documents(
    pool: asyncpg.Pool,
    items: list[dict[str, Any]],
) -> None:
    """Insere documentos e embeddings."""
    if not items:
        return
    records = [
        (
            item["id"],
            item["source"],
            item["content"],
            json.dumps(item.get("metadata") or {}),
            item["embedding"],
        )
        for item in items
    ]
    async with pool.acquire() as conn:
        await register_vector(conn)
        await conn.executemany(
            """
            INSERT INTO documents (id, source, content, metadata, embedding)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            """,
            records,
        )


async def search_documents(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int,
    ef_search: int = 40,
) -> list[dict[str, Any]]:
    """Busca documentos similares por distancia vetorial com HNSW otimizado.
    
    Args:
        pool: Pool de conexoes
        embedding: Vetor de embedding da query
        limit: Numero maximo de resultados
        ef_search: Tamanho da lista de candidatos na busca HNSW
                   Maior = mais preciso mas mais lento (default 40, range 10-200)
    
    O indice HNSW usa distancia cosseno (<=>) que retorna valores entre 0-2.
    Convertemos para score de similaridade (1 - distancia) onde:
    - 1.0 = identico
    - 0.0 = ortogonal
    - -1.0 = oposto
    """
    async with pool.acquire() as conn:
        await register_vector(conn)
        
        # Define ef_search para esta query (maior = mais preciso)
        await conn.execute(f"SET hnsw.ef_search = {ef_search};")
        
        rows = await conn.fetch(
            """
            SELECT id, source, content, metadata,
                   1 - (embedding <=> $1) AS score
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            limit,
        )
    return [dict(row) for row in rows]


async def search_documents_filtered(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int,
    source_filter: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Busca documentos com filtros opcionais de source e metadata.
    
    Util para buscar apenas em documentos de uma fonte especifica
    ou com determinados metadados.
    """
    async with pool.acquire() as conn:
        await register_vector(conn)
        await conn.execute("SET hnsw.ef_search = 40;")
        
        # Monta query dinamicamente baseado nos filtros
        query = """
            SELECT id, source, content, metadata,
                   1 - (embedding <=> $1) AS score
            FROM documents
            WHERE 1=1
        """
        params = [embedding]
        param_count = 1
        
        if source_filter:
            param_count += 1
            query += f" AND source = ${param_count}"
            params.append(source_filter)
        
        if metadata_filter:
            param_count += 1
            query += f" AND metadata @> ${param_count}::jsonb"
            params.append(json.dumps(metadata_filter))
        
        query += f" ORDER BY embedding <=> $1 LIMIT ${param_count + 1}"
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
    
    return [dict(row) for row in rows]


async def insert_history(
    pool: asyncpg.Pool,
    entries: list[dict[str, Any]],
) -> None:
    """Insere mensagens no historico."""
    if not entries:
        return
    records = [
        (
            entry["id"],
            entry["session_id"],
            entry["role"],
            entry["content"],
        )
        for entry in entries
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO chat_history (id, session_id, role, content)
            VALUES ($1, $2, $3, $4)
            """,
            records,
        )


async def fetch_history(
    pool: asyncpg.Pool,
    session_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Carrega historico ordenado para uma sessao."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content
            FROM chat_history
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return list(reversed([dict(row) for row in rows]))
