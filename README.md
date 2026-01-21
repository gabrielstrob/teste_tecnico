# API RAG com LLM Local

API de chatbot com Retrieval-Augmented Generation (RAG) utilizando LLM local (Ollama), embeddings locais e PostgreSQL com PGVector para armazenamento vetorial.

## Tecnologias

- **Backend**: Python 3.10+ com Litestar
- **LLM Local**: Ollama (Phi3, Mistral, Llama3, etc.)
- **Embeddings**: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Banco de Dados**: PostgreSQL com extensão PGVector
- **Orquestração LLM**: LangChain
- **Web Scraping**: Playwright + BeautifulSoup4
- **OCR**: Tesseract + pdf2image (para PDFs escaneados)
- **Containerização**: Docker Compose

## Como Executar

O projeto é 100% containerizado.

### 1. Configuração

Crie um arquivo `.env` na raiz do projeto (use `env.example` como base):

```env
# Banco de Dados
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres

# Ollama (LLM Local)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3

# Scraping
SCRAPE_URL=http://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial
SCRAPE_ON_STARTUP=true

# Embeddings
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
TOP_K=10
```

### 2. Iniciar Aplicação

Execute o comando abaixo, na pasta raiz do projeto, para construir e subir todos os containers:

```bash
docker-compose up -d --build
```

### 3. Baixar Modelo LLM

Após os containers subirem, baixe o modelo escolhido no Ollama (ex: qwen3):

```bash
docker exec -it llm_service ollama pull qwen3
```

### 4. Acessar Serviços

| Serviço | URL Externa (Host) | URL Interna (Docker Network) |
|---------|-------------------|------------------------------|
| API RAG | http://localhost:8000 | http://api:8000 |
| n8n (Fluxo) | http://localhost:5678 | http://n8n:5678 |
| Ollama | http://localhost:11434 | http://ollama:11434 |
| PostgreSQL | localhost:5432 | postgres:5432 |

---
### 5. Configurar Credenciais do Postgres no n8n
- Configurar com as mesmas credenciais do .env

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

**Exemplo de Resposta:**
```json
{
  "session_id": "uuid-da-sessao",
  "answer": "A inteligência artificial é...",
  "sources": [
    {
      "source": "documento.pdf",
      "score": 0.85,
      "metadata": { "page": 1 }
    }
  ]
}
```

### POST /scrape
Realiza scraping da URL configurada em `SCRAPE_URL`. Também aceita uma URL específica no corpo da requisição para scraping sob demanda.

**Funcionalidades:**
- Coleta conteúdo da página web
- Limpa HTML e extrai texto
- Divide em chunks com overlap
- Gera embeddings e armazena no banco

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
- URL: `http://api:8000/chat` (dentro da rede Docker)
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