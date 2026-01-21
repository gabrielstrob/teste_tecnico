# API RAG com LLM Local

API de chatbot com Retrieval-Augmented Generation (RAG) utilizando LLM local (Ollama), embeddings locais e PostgreSQL com PGVector para armazenamento vetorial.


## Tecnologias

- **Backend**: Python 3.10+ com Litestar
- **LLM Local**: Ollama (Phi3, Mistral, Llama3, etc.)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Banco de Dados**: PostgreSQL com extensão PGVector
- **Orquestração LLM**: LangChain
- **Web Scraping**: Playwright + BeautifulSoup4
- **OCR**: Tesseract + pdf2image (para PDFs escaneados)
- **Containerização**: Docker Compose

## Funcionalidades

### POST /chat
Recebe pergunta e/ou arquivo para processamento RAG.

**Funcionalidades:**
- Recebe perguntas em texto
- Aceita upload de arquivos (PDF, CSV, Excel)
- Extrai texto com OCR automático para PDFs escaneados
- Gera embeddings localmente
- Busca vetorial por documentos relevantes
- Gera resposta usando LLM local
- Mantém histórico de conversa por sessão

**Exemplo de requisição (JSON):**
```json
{
  "question": "O que é inteligência artificial?",
  "session_id": "opcional-uuid"
}
```

**Exemplo de requisição (Form-Data com arquivo):**
```
POST /chat
Content-Type: multipart/form-data

file: documento.pdf
question: Resuma o documento (opcional)
session_id: uuid (opcional)
```

### POST /scrape
Realiza scraping da URL configurada em `SCRAPE_URL`.

**Funcionalidades:**
- Coleta conteúdo da página web
- Limpa HTML e extrai texto
- Divide em chunks com overlap
- Gera embeddings e armazena no banco

## Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Banco de Dados
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres

# Ollama (LLM Local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3

# Scraping
SCRAPE_URL=https://pt.wikipedia.org/wiki/Inteligência_artificial
SCRAPE_ON_STARTUP=false

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

## Execução

### Opção 1: Docker Compose (Recomendado)

**1. Suba todos os containers:**
```bash
docker-compose up -d --build
```

**2. Baixe o modelo do Ollama:**
```bash
docker exec -it llm_service ollama pull phi3
```

**3. Acesse os serviços:**
- API RAG: http://localhost:8000
- n8n: http://localhost:5678
- Ollama: http://localhost:11434

### Opção 2: Execução Local

**1. Suba apenas os serviços de infraestrutura:**
```bash
docker-compose up -d postgres ollama
```

**2. Baixe o modelo do Ollama:**
```bash
docker exec -it llm_service ollama pull phi3
```

**3. Crie e ative o ambiente virtual:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

**4. Instale as dependências:**
```bash
pip install -r api/requirements.txt
playwright install chromium
```

**5. Execute a API:**
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Estrutura do Projeto

```
├── api/
│   ├── __init__.py
│   ├── app.py           # Aplicação Litestar e endpoints
│   ├── chat.py          # Lógica RAG e comunicação com LLM
│   ├── config.py        # Configurações e variáveis de ambiente
│   ├── db.py            # Conexão e operações no PostgreSQL
│   ├── embeddings.py    # Geração de embeddings locais
│   ├── models.py        # Modelos de resposta da API
│   ├── scrape.py        # Web scraping com Playwright
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── loaders.py   # Extração de texto (PDF, CSV, Excel, OCR)
│   └── requirements.txt
├── scripts/
│   └── init-db.sh       # Script de inicialização do banco
├── docker-compose.yml
├── Dockerfile
├── .env
└── README.md
```

## Uso com n8n

### Configuração do HTTP Request Node

**Para perguntas de texto:**
- Method: `POST`
- URL: `http://api:8000/chat` (dentro do Docker) ou `http://localhost:8000/chat` (local)
- Body Content Type: `JSON`
- Body:
```json
{
  "question": "{{ $json.pergunta }}",
  "session_id": "{{ $json.session_id }}"
}
```

**Para upload de arquivos:**
- Method: `POST`
- URL: `http://api:8000/chat`
- Body Content Type: `Form-Data Multipart`
- Body Parameters:
  - `file`: (tipo File) - arquivo binário
  - `session_id`: (tipo Text) - opcional

## Comandos Docker Úteis

```bash
# Ver logs da API
docker-compose logs -f api

# Reiniciar apenas a API
docker-compose restart api

# Verificar status dos containers
docker-compose ps

# Parar todos os serviços
docker-compose down

# Parar e limpar volumes (remove dados do banco)
docker-compose down -v

# Reconstruir imagem da API
docker-compose build api
```

## Formatos de Arquivo Suportados

| Formato | Extensão | Observação |
|---------|----------|------------|
| PDF | `.pdf` | Suporte a OCR para PDFs escaneados |
| CSV | `.csv` | Conversão automática para texto |
| Excel | `.xls`, `.xlsx` | Conversão automática para texto |

## Dependências do Sistema (para OCR local)

Se executar fora do Docker e precisar de OCR:

**Windows:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/osborn/poppler-windows/releases