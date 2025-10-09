# config_ec2.py
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class EC2Config:
    # API Keys (use environment variables for security)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Processing Parameters
    batch_size: int = int(os.getenv("BATCH_SIZE", "20"))  # Reduced for EC2
    max_memory_usage: float = float(os.getenv("MAX_MEMORY_USAGE", "0.7"))  # Reduced for EC2
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Cache Configuration
    cache_expiry_days: int = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
    
    # Data Paths - Updated for EC2
    data_paths: List[str] = field(default_factory=lambda: [
        "/home/ubuntu/data/Extracted_Files",
        "/home/ubuntu/data/Air_purge_Reference_materials",
        "/home/ubuntu/data/Air_Purge_LG_Claim"
    ])
    
    # Model Configuration
    text_model: str = os.getenv("TEXT_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-oss:20b")
    
    # Security
    session_expiry_days: int = int(os.getenv("SESSION_EXPIRY_DAYS", "7"))
    max_login_attempts: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    
    # Performance
    max_workers: int = int(os.getenv("MAX_WORKERS", "2"))  # Reduced for EC2
    
    # UI Configuration
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "20"))
    
    # EC2 Specific
    host: str = "0.0.0.0"  # Listen on all interfaces
    port: int = int(os.getenv("PORT", "8501"))
