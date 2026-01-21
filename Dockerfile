# Dockerfile para a API RAG com suporte a OCR
FROM python:3.11-slim

# Define diretorio de trabalho
WORKDIR /app

# Instala dependencias do sistema para OCR e Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dependencias para pdf2image (poppler)
    poppler-utils \
    # Dependencias para Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-por \
    tesseract-ocr-eng \
    # Dependencias para Playwright
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # curl para healthcheck
    curl \
    # Limpeza
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependencias Python
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instala navegadores do Playwright
RUN playwright install chromium

# Copia codigo da aplicacao
COPY api/ ./api/

# Porta da API
EXPOSE 8000

# Comando para iniciar a aplicacao
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
