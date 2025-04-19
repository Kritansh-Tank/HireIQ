"""
Utilities package for the AI-Powered Job Application Screening System.

This package contains utilities for various functions including:
- PDF extraction
- Text processing
- File handling
- Ollama LLM integration
- Embedding generation
- Custom tools
"""

# Import utilities to make them available when importing the package
from utils.pdf_extractor import PDFExtractor
from utils.ollama_client import OllamaClient
from utils.embeddings import EmbeddingUtility
from utils.web_scraper import WebScraper

__all__ = [
    'PDFExtractor',
    'OllamaClient',
    'EmbeddingUtility',
    'WebScraper'
] 