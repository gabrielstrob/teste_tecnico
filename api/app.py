from __future__ import annotations

"""API Litestar: endpoints e ciclo de vida."""

import uuid
from typing import Any

from litestar import Litestar, Request, Response, post
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException

from .chat import generate_answer, persist_history, retrieve_context, run_rag_pipeline
from .config import get_settings
from .db import create_pool, fetch_history, init_db
from .ingest import extract_text_from_file
from .models import ChatResponse, ScrapeResponse
from .scrape import chunk_and_store, scrape_and_store


@post("/scrape")
async def scrape_endpoint(request: Request) -> ScrapeResponse:
    """Realiza scraping da URL configurada ou fornecida no body (JSON ou Form)."""
    settings = request.app.state.settings
    pool = request.app.state.pool
    
    url: str | None = None
    
    # Detecta content-type
    raw_content_type = request.content_type
    if isinstance(raw_content_type, tuple):
        content_type = (raw_content_type[0] or "").lower()
    else:
        content_type = (raw_content_type or "").lower()

    # Tenta obter URL conforme o tipo de envio
    if "application/json" in content_type:
        try:
            data = await request.json()
            url = data.get("url")
        except Exception:
            pass
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            url = form.get("url")
        except Exception:
            pass

    try:
        chunks = await scrape_and_store(pool, settings, url=url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    return ScrapeResponse(url=url or settings.scrape_url or "", chunks=chunks)


@post("/chat")
async def chat_endpoint(request: Request) -> Response:
    """Recebe pergunta e/ou arquivo(s), processa e retorna resposta do LLM."""
    settings = request.app.state.settings
    pool = request.app.state.pool

    question: str | None = None
    session_id: str | None = None
    uploads: list[UploadFile] = []

    # Aceita JSON ou multipart (para upload de arquivo).
    raw_content_type = request.content_type
    if isinstance(raw_content_type, tuple):
        content_type = (raw_content_type[0] or "").lower()
    else:
        content_type = (raw_content_type or "").lower()

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        question = form.get("question")
        session_id = form.get("session_id")
        
        if "multipart/form-data" in content_type:
            # Aceita tanto "file" (singular) quanto "files" (multiplos)
            # para manter compatibilidade com integracoes existentes
            files_list = list(form.getall("files")) if "files" in form else []
            single_file = form.get("file")
            if single_file and hasattr(single_file, "read"):
                files_list.append(single_file)
            uploads = [f for f in files_list if hasattr(f, "read")]
    else:
        data = await request.json()
        question = data.get("question")
        session_id = data.get("session_id")

    # Pelo menos uma pergunta ou arquivo deve ser enviado.
    if not question and not uploads:
        raise HTTPException(
            status_code=400,
            detail="Envie 'question' e/ou 'file'/'files' na requisicao.",
        )

    # Se houver arquivo(s), extrai e grava embeddings para cada um.
    files_processed: list[str] = []
    files_errors: list[dict] = []
    
    for upload in uploads:
        filename = getattr(upload, "filename", None) or "arquivo"
        try:
            file_bytes = await upload.read()
            text = extract_text_from_file(filename, file_bytes)
            await chunk_and_store(
                pool,
                settings,
                text,
                source="upload",
                metadata={"filename": filename},
            )
            files_processed.append(filename)
        except ValueError as exc:
            files_errors.append({"filename": filename, "error": str(exc)})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            files_errors.append({"filename": filename, "error": str(exc)})

    # Se nao houver pergunta, retorna confirmacao com detalhes dos arquivos.
    if not question:
        total = len(files_processed)
        if total == 0 and files_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao processar arquivo(s): {files_errors}",
            )
        answer = f"{total} arquivo(s) processado(s) com sucesso."
        if files_errors:
            answer += f" {len(files_errors)} arquivo(s) com erro."
        response = {
            "session_id": session_id or str(uuid.uuid4()),
            "answer": answer,
            "files_processed": files_processed,
            "files_errors": files_errors if files_errors else None,
            "sources": [],
        }
        return Response(response, status_code=200)

    session_id = session_id or str(uuid.uuid4())
    history = await fetch_history(pool, session_id, limit=settings.max_history)
    
    # Usa pipeline otimizado que combina retrieve + generate
    answer, docs = await run_rag_pipeline(pool, settings, question, history)
    await persist_history(pool, session_id, question, answer)

    response = ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=[
            {
                "source": doc.get("source"),
                "score": float(doc.get("score") or 0),
                "metadata": doc.get("metadata") or {},
            }
            for doc in docs
        ],
    )
    return Response(response, status_code=200)


async def on_startup(app: Litestar) -> None:
    """Inicializa pool, schema e scraping opcional."""
    settings = get_settings()
    pool = await create_pool(settings)
    await init_db(pool, settings.embedding_dim)
    app.state.settings = settings
    app.state.pool = pool

    if settings.scrape_on_startup and settings.scrape_url:
        try:
            await scrape_and_store(pool, settings)
        except Exception:
            pass


async def on_shutdown(app: Litestar) -> None:
    """Encerra conexoes ao banco."""
    pool = getattr(app.state, "pool", None)
    if pool:
        await pool.close()


app = Litestar(
    route_handlers=[chat_endpoint, scrape_endpoint],
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
    debug=True,  # Mostra tracebacks completos no terminal
)
