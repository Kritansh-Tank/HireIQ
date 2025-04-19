"""
Configuration settings for the AI-Powered Job Application Screening System.
"""

import os
from pathlib import Path

# Base directory paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "Dataset" / "[Usecase 5] AI-Powered Job Application Screening System"
JD_CSV_PATH = DATA_DIR / "job_description.csv"
CV_DIR = DATA_DIR / "CVs1"

# Database settings
DB_PATH = BASE_DIR / "database" / "job_screening.db"

# Agent settings
MATCH_THRESHOLD = 0.80

# Feature flags
USE_LLM = True
USE_EMBEDDINGS = True
USE_LOCAL_EMBEDDINGS = True

# NLP settings
NLP_MODEL = "en_core_web_md"

# Email settings
EMAIL_SENDER = "Sender's Email"
PASSWORD = "Your SMTP Password"
EMAIL_TEMPLATE_PATH = BASE_DIR / "utils" / "email_templates"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "logs" / "app.log"

# Initialize directories if they don't exist
os.makedirs(EMAIL_TEMPLATE_PATH, exist_ok=True)
os.makedirs(BASE_DIR / "logs", exist_ok=True)

# Ollama settings
OLLAMA_BASE_URL = "http://35.154.211.247:11434"
OLLAMA_LLM_MODEL = "qwen2.5:0.5b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_TIMEOUT = 300
OLLAMA_MAX_RETRIES = 2
OLLAMA_RETRY_DELAY = 2

# Caching settings
ENABLE_CACHING = True
CACHE_DIR = BASE_DIR / "cache"
EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"
LLM_CACHE_DIR = CACHE_DIR / "llm_responses"
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

# Performance optimization settings
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKERS = 8 
BATCH_SIZE = 20 
USE_LLM_FOR_TOP_K = 5

# Custom tools settings
TOOLS_DIR = BASE_DIR / "tools"
os.makedirs(TOOLS_DIR, exist_ok=True)

# Web scraping settings
SCRAPER_CACHE_DIR = TOOLS_DIR / "scraper_cache"
SCRAPER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
SCRAPER_TIMEOUT = 30
os.makedirs(SCRAPER_CACHE_DIR, exist_ok=True)

# ML model settings
ML_MODELS_DIR = TOOLS_DIR / "ml_models"
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# Multi-agent framework settings
AGENT_MAX_RETRIES = 3
AGENT_RETRY_DELAY = 2
AGENT_MESSAGE_QUEUE_SIZE = 100
