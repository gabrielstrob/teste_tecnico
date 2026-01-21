from __future__ import annotations

"""Extracao de texto de PDF/CSV/Excel com suporte a OCR."""

import io
import logging
from typing import Callable

import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

# Limite minimo de caracteres para considerar extracao valida
MIN_TEXT_LENGTH = 50


def _clean_text(text: str) -> str:
    """Normaliza espacos em branco e remove caracteres invalidos para UTF-8/PostgreSQL."""
    # Remove bytes nulos e caracteres de controle invalidos
    text = text.replace("\x00", "")
    # Remove outros caracteres de controle (exceto newline, tab, carriage return)
    cleaned = "".join(
        char for char in text
        if char in ("\n", "\t", "\r") or (ord(char) >= 32 or char.isprintable())
    )
    return " ".join(cleaned.split())


def _extract_pdf_with_pdfplumber(data: bytes) -> str:
    """Extrai PDF via pdfplumber."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _extract_pdf_with_pypdf2(data: bytes) -> str:
    """Extrai PDF via PyPDF2 (fallback)."""
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_pdf_with_ocr(data: bytes) -> str:
    """Extrai texto de PDF usando OCR (para PDFs escaneados/imagens).
    
    Requer Tesseract instalado no sistema:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Linux: sudo apt install tesseract-ocr tesseract-ocr-por
    - Mac: brew install tesseract tesseract-lang
    """
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError as e:
        logger.warning(f"OCR nao disponivel: {e}")
        return ""
    
    logger.info("Usando OCR para extrair texto do PDF...")
    try:
        # Converte paginas do PDF em imagens
        images = convert_from_bytes(data, dpi=300)
        pages_text = []
        for i, image in enumerate(images):
            # Extrai texto de cada imagem usando Tesseract
            # Tenta portugues primeiro, depois ingles como fallback
            try:
                text = pytesseract.image_to_string(image, lang="por+eng")
            except Exception:
                text = pytesseract.image_to_string(image)
            pages_text.append(text)
            logger.debug(f"OCR pagina {i+1}: {len(text)} caracteres")
        return "\n".join(pages_text)
    except Exception as e:
        logger.error(f"Erro no OCR: {e}")
        return ""


def _dataframe_to_semantic_text(df: pd.DataFrame) -> str:
    """Converte DataFrame para texto semantico otimizado para RAG.
    
    Em vez de exportar como CSV puro, formata cada linha como um
    "documento" legivel com os headers incluidos, facilitando a
    busca semantica.
    
    Exemplo de saida:
    ---
    Nome: João Silva
    Email: joao@email.com
    Valor: R$ 1.500,00
    ---
    Nome: Maria Santos
    ...
    """
    if df.empty:
        return ""
    
    # Remove colunas vazias
    df = df.dropna(axis=1, how='all')
    
    # Limpa nomes das colunas
    df.columns = [str(col).strip() for col in df.columns]
    
    documents = []
    headers = list(df.columns)
    
    for idx, row in df.iterrows():
        lines = []
        for header in headers:
            value = row[header]
            # Pula valores nulos/vazios
            if pd.isna(value) or str(value).strip() == '':
                continue
            # Formata como "Header: Valor"
            lines.append(f"{header}: {value}")
        
        if lines:
            documents.append("\n".join(lines))
    
    # Separa cada registro com delimitador claro
    return "\n---\n".join(documents)


def _extract_csv(data: bytes) -> str:
    """Converte CSV para texto semantico otimizado para RAG."""
    try:
        # Tenta detectar encoding e separador automaticamente
        df = pd.read_csv(io.BytesIO(data), encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(data), encoding='latin-1')
    except Exception:
        # Fallback com separador ponto-e-virgula (comum em CSVs brasileiros)
        try:
            df = pd.read_csv(io.BytesIO(data), sep=';', encoding='utf-8')
        except Exception:
            df = pd.read_csv(io.BytesIO(data), sep=';', encoding='latin-1')
    
    return _dataframe_to_semantic_text(df)


def _extract_excel(data: bytes) -> str:
    """Converte Excel para texto semantico otimizado para RAG.
    
    Processa todas as abas da planilha.
    """
    all_sheets_text = []
    
    try:
        # Le todas as abas
        xlsx = pd.ExcelFile(io.BytesIO(data))
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            sheet_text = _dataframe_to_semantic_text(df)
            if sheet_text:
                all_sheets_text.append(f"=== Aba: {sheet_name} ===\n{sheet_text}")
    except Exception as e:
        logger.warning(f"Erro ao ler Excel com multiplas abas: {e}")
        # Fallback para leitura simples
        df = pd.read_excel(io.BytesIO(data))
        return _dataframe_to_semantic_text(df)
    
    return "\n\n".join(all_sheets_text)


def _extract_pdf(data: bytes) -> str:
    """Extrai texto de PDF com fallback para OCR se necessario."""
    text = ""
    
    # Tenta pdfplumber primeiro
    try:
        text = _extract_pdf_with_pdfplumber(data)
        logger.debug(f"pdfplumber extraiu {len(text)} caracteres")
    except Exception as e:
        logger.debug(f"pdfplumber falhou: {e}")
    
    # Se falhou ou texto muito curto, tenta PyPDF2
    if len(text.strip()) < MIN_TEXT_LENGTH:
        try:
            text = _extract_pdf_with_pypdf2(data)
            logger.debug(f"PyPDF2 extraiu {len(text)} caracteres")
        except Exception as e:
            logger.debug(f"PyPDF2 falhou: {e}")
    
    # Se ainda nao tem texto suficiente, tenta OCR
    if len(text.strip()) < MIN_TEXT_LENGTH:
        logger.info("Extracao normal insuficiente, tentando OCR...")
        ocr_text = _extract_pdf_with_ocr(data)
        if len(ocr_text.strip()) > len(text.strip()):
            text = ocr_text
            logger.info(f"OCR extraiu {len(text)} caracteres")
    
    return text


def extract_text_from_file(filename: str, data: bytes) -> str:
    """Detecta o tipo e extrai o texto."""
    name = filename.lower()
    extractor: Callable[[bytes], str] | None = None
    
    if name.endswith(".pdf"):
        text = _extract_pdf(data)
        return _clean_text(text)
    
    if name.endswith(".csv"):
        extractor = _extract_csv
    elif name.endswith((".xls", ".xlsx")):
        extractor = _extract_excel
    
    if extractor is None:
        raise ValueError("Formato de arquivo nao suportado.")
    
    return _clean_text(extractor(data))
