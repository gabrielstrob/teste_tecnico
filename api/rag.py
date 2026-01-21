from __future__ import annotations

"""RAG otimizado com LangChain: query preprocessing, hybrid search e cache."""

import asyncio
import hashlib
import logging
import re
import unicodedata
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

from .config import Settings

logger = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSAMENTO DE QUERY
# ============================================================================

def preprocess_query(query: str) -> str:
    """Preprocessa a query para melhorar matching.
    
    - Remove acentos para matching mais flexivel
    - Normaliza espacos
    - Mantem a query original tambem para nao perder contexto
    """
    # Normaliza espacos
    query = " ".join(query.split())
    return query


def expand_query(query: str) -> list[str]:
    """Expande a query em variações para melhorar recall.
    
    Retorna lista de queries para buscar (a original + variações).
    """
    queries = [query]
    
    # Versao sem acentos
    normalized = unicodedata.normalize('NFKD', query)
    without_accents = ''.join(c for c in normalized if not unicodedata.combining(c))
    if without_accents != query:
        queries.append(without_accents)
    
    return queries


def extract_keywords(text: str) -> list[str]:
    """Extrai keywords importantes do texto para busca hibrida."""
    # Remove pontuacao e converte para minusculas
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Remove stopwords comuns em portugues
    stopwords = {
        'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas',
        'um', 'uma', 'uns', 'umas', 'o', 'a', 'os', 'as', 'e', 'ou',
        'que', 'qual', 'quais', 'como', 'para', 'por', 'com', 'sem',
        'me', 'te', 'se', 'lhe', 'nos', 'vos', 'lhes', 'meu', 'minha',
        'seu', 'sua', 'esse', 'essa', 'este', 'esta', 'isso', 'isto',
        'aquele', 'aquela', 'onde', 'quando', 'porque', 'pois',
        'ser', 'estar', 'ter', 'haver', 'fazer', 'poder', 'dever',
        'quero', 'preciso', 'gostaria', 'poderia', 'favor', 'obrigado',
        'informacoes', 'informacao', 'sobre', 'dessa', 'desse', 'deste', 'desta',
    }
    
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


# ============================================================================
# CACHE GLOBAL
# ============================================================================

# Cache de embeddings de queries (evita recalcular para perguntas repetidas)
_query_embedding_cache: dict[str, list[float]] = {}
_cache_initialized = False


def init_cache(enable: bool = True) -> None:
    """Inicializa cache de respostas LLM."""
    global _cache_initialized
    if not _cache_initialized and enable:
        set_llm_cache(InMemoryCache())
        _cache_initialized = True
        logger.info("Cache LLM inicializado")


def _cache_key(text: str, model: str) -> str:
    """Gera chave de cache para embedding."""
    return hashlib.md5(f"{model}:{text}".encode()).hexdigest()


# ============================================================================
# EMBEDDINGS OTIMIZADOS
# ============================================================================

class CachedHuggingFaceEmbeddings(Embeddings):
    """Wrapper com cache para HuggingFaceEmbeddings."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Mude para "cuda" se tiver GPU
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,  # Processa em lotes
            },
        )
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para documentos (sem cache, usado na ingestao)."""
        return self._embedder.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        """Gera embedding para query com cache."""
        cache_key = _cache_key(text, self.model_name)
        if cache_key in _query_embedding_cache:
            logger.debug(f"Cache hit para query embedding")
            return _query_embedding_cache[cache_key]
        
        embedding = self._embedder.embed_query(text)
        _query_embedding_cache[cache_key] = embedding
        
        # Limita tamanho do cache (LRU simples)
        if len(_query_embedding_cache) > 1000:
            # Remove metade das entradas mais antigas
            keys = list(_query_embedding_cache.keys())[:500]
            for k in keys:
                del _query_embedding_cache[k]
        
        return embedding


@lru_cache(maxsize=1)
def get_embeddings(model_name: str) -> CachedHuggingFaceEmbeddings:
    """Singleton para embeddings."""
    return CachedHuggingFaceEmbeddings(model_name)


# ============================================================================
# RAG CHAIN OTIMIZADA
# ============================================================================

class RAGChain:
    """Chain RAG com todos os componentes otimizados."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = get_embeddings(settings.embedding_model)
        
        # Inicializa cache se habilitado
        init_cache(settings.enable_cache)
        
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_url,
            temperature=settings.temperature,
        )
        
        # Prompt otimizado com instrucoes claras
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Voce e um assistente util e preciso. 
Use APENAS as informacoes do contexto fornecido para responder.
Se a informacao nao estiver no contexto, diga claramente que nao encontrou.
Seja conciso e direto nas respostas.

Contexto:
{context}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
    
    def _format_docs(self, docs: list[Document]) -> str:
        """Formata documentos para o prompt."""
        if not docs:
            return "Nenhum contexto relevante encontrado."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconhecido")
            formatted.append(f"[{i}. {source}]\n{doc.page_content}")
        
        return "\n\n".join(formatted)
    
    def _format_history(self, history: list[dict[str, Any]]) -> list:
        """Converte historico do banco para mensagens LangChain."""
        messages = []
        # Limita historico para evitar contexto muito grande
        max_hist = self.settings.max_history
        recent_history = history[-max_hist:] if len(history) > max_hist else history
        
        for item in recent_history:
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))
        
        return messages
    
    async def retrieve(
        self,
        pool,
        question: str,
        source_filter: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Recupera documentos relevantes com busca hibrida otimizada.
        
        Melhorias aplicadas:
        - Preprocessamento da query
        - Keyword boost (aumenta score se keywords estão presentes)
        - Threshold adaptativo
        
        Args:
            pool: Pool de conexoes do banco
            question: Pergunta do usuario
            source_filter: Filtrar por source (ex: "upload", "scrape")
            metadata_filter: Filtrar por metadata (ex: {"filename": "doc.pdf"})
        """
        # Preprocessa a query
        processed_query = preprocess_query(question)
        keywords = extract_keywords(processed_query)
        
        logger.info(f"Query: '{processed_query[:50]}...' | Keywords: {keywords[:5]}")
        
        # Gera embedding da query (com cache)
        query_embedding = self.embeddings.embed_query(processed_query)
        
        # Busca direta no banco - pega mais documentos para filtrar depois
        from .db import search_documents, search_documents_filtered
        
        fetch_limit = max(self.settings.top_k * 4, 20)  # Pega mais para ter margem
        
        if source_filter or metadata_filter:
            docs = await search_documents_filtered(
                pool, query_embedding, fetch_limit,
                source_filter=source_filter,
                metadata_filter=metadata_filter,
            )
        else:
            docs = await search_documents(pool, query_embedding, fetch_limit)
        
        logger.info(f"Busca vetorial retornou {len(docs)} documentos")
        
        # Processa e rankeia documentos com score hibrido
        scored_docs = []
        for doc in docs:
            vector_score = float(doc.get("score") or 0)
            content = doc.get("content", "").lower()
            
            # Calcula keyword boost (0.0 a 0.3)
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
            keyword_boost = min(keyword_matches * 0.05, 0.3) if keywords else 0
            
            # Score final: vetorial + boost de keywords
            final_score = vector_score + keyword_boost
            
            # Log dos top scores para debug
            if len(scored_docs) < 5:
                logger.debug(f"  Doc: vec={vector_score:.4f} + kw_boost={keyword_boost:.4f} = {final_score:.4f}")
            
            # Threshold mais baixo (o boost pode compensar)
            if final_score < self.settings.similarity_threshold:
                continue
            
            # Metadata pode vir como string JSON do banco
            raw_metadata = doc.get("metadata") or {}
            if isinstance(raw_metadata, str):
                import json
                try:
                    raw_metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    raw_metadata = {}
            
            scored_docs.append(Document(
                page_content=doc["content"],
                metadata={
                    "source": doc.get("source", ""),
                    "score": final_score,
                    "vector_score": vector_score,
                    "keyword_boost": keyword_boost,
                    **raw_metadata,
                },
            ))
        
        logger.info(f"Apos filtro (threshold={self.settings.similarity_threshold}): {len(scored_docs)} documentos")
        
        # Ordena por score final e limita
        scored_docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        return scored_docs[:self.settings.top_k]
    
    async def generate(
        self,
        question: str,
        docs: list[Document],
        history: list[dict[str, Any]],
    ) -> str:
        """Gera resposta usando a chain otimizada."""
        context = self._format_docs(docs)
        history_messages = self._format_history(history)
        
        # Cria chain com LCEL
        chain = self.prompt | self.llm | StrOutputParser()
        
        # Executa de forma assincrona
        response = await chain.ainvoke({
            "context": context,
            "history": history_messages,
            "question": question,
        })
        
        return response.strip()
    
    async def run(
        self,
        pool,
        question: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, list[Document]]:
        """Executa o pipeline completo de RAG."""
        # Retrieve
        docs = await self.retrieve(pool, question)
        
        # Generate
        answer = await self.generate(question, docs, history)
        
        return answer, docs


# ============================================================================
# FUNCOES DE CONVENIENCIA
# ============================================================================

_rag_chain: RAGChain | None = None


def get_rag_chain(settings: Settings) -> RAGChain:
    """Singleton para a chain RAG."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain(settings)
    return _rag_chain


async def retrieve_context(
    pool,
    settings: Settings,
    question: str,
) -> list[dict[str, Any]]:
    """Interface compativel com o codigo existente."""
    chain = get_rag_chain(settings)
    docs = await chain.retrieve(pool, question)
    
    return [
        {
            "source": doc.metadata.get("source", ""),
            "content": doc.page_content,
            "score": doc.metadata.get("score", 0),
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


async def generate_answer(
    settings: Settings,
    question: str,
    history: list[dict[str, Any]],
    docs: list[dict[str, Any]],
) -> str:
    """Interface compativel com o codigo existente."""
    chain = get_rag_chain(settings)
    
    # Converte para Documents
    documents = [
        Document(
            page_content=doc.get("content", ""),
            metadata={
                "source": doc.get("source", ""),
                "score": doc.get("score", 0),
            },
        )
        for doc in docs
    ]
    
    return await chain.generate(question, documents, history)


async def embed_texts_async(model_name: str, texts: list[str]) -> list[list[float]]:
    """Gera embeddings de forma assincrona (em thread pool)."""
    embeddings = get_embeddings(model_name)
    
    # Executa em thread separada para nao bloquear event loop
    return await asyncio.to_thread(embeddings.embed_documents, texts)
