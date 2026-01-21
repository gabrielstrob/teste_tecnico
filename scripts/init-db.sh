#!/bin/bash
set -e

# Cria os usuários e bancos de dados (se não existirem)
# O banco 'n8n_db' é para o N8N
# O banco 'rag_db' é para a sua API Python + LangChain
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
	SELECT 'CREATE DATABASE n8n_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'n8n_db')\gexec
	SELECT 'CREATE DATABASE rag_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'rag_db')\gexec
EOSQL

# Conecta especificamente no banco 'rag_db' para habilitar a extensão pgvector
# O n8n NÃO precisa dessa extensão, apenas a API de RAG
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "rag_db" <<-EOSQL
	CREATE EXTENSION IF NOT EXISTS vector;
EOSQL