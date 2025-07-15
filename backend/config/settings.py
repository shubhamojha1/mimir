# backend/config/settings.py
"""
Configuration settings for Mimir backend.
This module centralizes all environment variables and configuration options,
making it easy to manage different environments (dev, staging, production).
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    Uses Pydantic for automatic validation and type checking.
    """
    
    # Application settings
    app_name: str = "Mimir Knowledge Management System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database settings
    # database_url: str = Field(..., env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Vector database settings (ChromaDB)
    vector_db_host: str = Field(default="localhost", env="VECTOR_DB_HOST")
    vector_db_port: int = Field(default=8000, env="VECTOR_DB_PORT")
    vector_collection_name: str = Field(default="mimir_documents", env="VECTOR_COLLECTION_NAME")
    
    # LLM API settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")  # openai, anthropic, ollama
    
    # Authentication settings
    # secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # File upload settings
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="./data/processed", env="PROCESSED_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_extensions: List[str] = Field(
        default=["pdf", "docx", "txt", "mp4", "mp3", "wav", "m4a"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Audio/Video processing settings
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")
    tts_provider: str = Field(default="openai", env="TTS_PROVIDER")  # openai, elevenlabs, google
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    
    # MCP (Model Context Protocol) settings
    mcp_server_port: int = Field(default=8001, env="MCP_SERVER_PORT")
    google_drive_credentials: Optional[str] = Field(default=None, env="GOOGLE_DRIVE_CREDENTIALS")
    dropbox_access_token: Optional[str] = Field(default=None, env="DROPBOX_ACCESS_TOKEN")
    
    # RAG settings - These control how retrieval-augmented generation works
    rag_chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")  # Characters per chunk
    rag_chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")  # Overlap between chunks
    rag_similarity_threshold: float = Field(default=0.7, env="RAG_SIMILARITY_THRESHOLD")
    rag_max_results: int = Field(default=5, env="RAG_MAX_RESULTS")  # Max chunks to retrieve
    
    # Redis settings for caching
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures we only create the settings object once,
    which is important for performance and consistency.
    """
    return Settings()

# Create directories if they don't exist
def ensure_directories():
    """
    Create necessary directories for file storage.
    This runs at startup to ensure the application has the required folders.
    """
    settings = get_settings()
    directories = [
        settings.upload_dir,
        settings.processed_dir,
        "./data/audio_summaries",
        "./data/video_summaries"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)