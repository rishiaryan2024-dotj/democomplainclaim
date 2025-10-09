# config_ec2.py
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class EC2Config:
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Processing Parameters
    batch_size: int = int(os.getenv("BATCH_SIZE", "20"))
    max_memory_usage: float = float(os.getenv("MAX_MEMORY_USAGE", "0.7"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Cache Configuration
    cache_expiry_days: int = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
    
    # Data Paths - Using ec2-user paths
    data_paths: List[str] = field(default_factory=lambda: [
        "/home/ec2-user/data/Extracted_Files",
        "/home/ec2-user/data/Air_purge_Reference_materials",
        "/home/ec2-user/data/Air_Purge_LG_Claim"
    ])
    
    # Model Configuration - REMOTE OLLAMA VIA NGROK
    text_model: str = os.getenv("TEXT_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-oss:20b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "https://tenantlike-nontyrannically-karla.ngrok-free.dev")
    
    # Security
    session_expiry_days: int = int(os.getenv("SESSION_EXPIRY_DAYS", "7"))
    max_login_attempts: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    
    # Performance
    max_workers: int = int(os.getenv("MAX_WORKERS", "2"))
    
    # UI Configuration
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "20"))
    
    # EC2 Specific
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8501"))
