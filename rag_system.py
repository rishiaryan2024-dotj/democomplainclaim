# ===== IMPORTS =====
# At the top of rag_system.py
import os
import sys

# Check if we're running on EC2 or locally
if os.path.exists("/home/ec2-user"):
    from config_ec2 import EC2Config as Config
    USE_LOCAL_OLLAMA = False
else:
    from config_local import LocalConfig as Config
    USE_LOCAL_OLLAMA = True

import re
import json
import time
import uuid
import pickle
import hashlib
import secrets
import logging
import threading
import queue
import asyncio
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import sqlite3
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False
# Third-party imports (made optional to avoid ModuleNotFoundError on deployment)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pd = None
    PANDAS_AVAILABLE = False

# bcrypt: optional (fallback hashing implemented later)
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    bcrypt = None
    BCRYPT_AVAILABLE = False

# bcrypt is imported earlier and handled with BCRYPT_AVAILABLE flag
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    px = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except Exception:
    RecursiveCharacterTextSplitter = None
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False

try:
    from typing import TypedDict
except Exception:
    TypedDict = dict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    TQDM_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except Exception:
    Translator = None
    GOOGLETRANS_AVAILABLE = False
# Optional imports (with fallbacks)
try:
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    partition_pdf = None
try:
    import pdf2image
except ImportError:
    pdf2image = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    import docx2txt
except ImportError:
    docx2txt = None
try:
    import markdown
except ImportError:
    markdown = None
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ===== SPEECH-TO-TEXT & TEXT-TO-SPEECH IMPORTS =====
# Handle the faster_whisper import with proper error handling
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import faster_whisper: {str(e)}")
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    gTTS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
@dataclass
class Config:
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "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")
    # Processing Parameters
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
    max_memory_usage: float = float(os.getenv("MAX_MEMORY_USAGE", "0.8"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    # Cache Configuration
    cache_expiry_days: int = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
    # Data Paths
    # Data Paths - Updated to your specified paths
    data_paths: List[str] = field(default_factory=lambda: [
        r"C:\Users\PCZ2109-001\Downloads\Extracted_Files",
        r"C:\Users\PCZ2109-001\Downloads\Air purge_Reference materials to use when handling complaints",
        r"C:\Users\PCZ2109-001\Downloads\Air Purge LG_Claim\Air Purge LG_Claim"
    ])
    # Model Configuration
    text_model: str = os.getenv("TEXT_MODEL", "nomic-embed-text")  #nomic-embed-text
    llm_model: str = os.getenv("LLM_MODEL", "gpt-oss:20b")  # Much faster model
    # Security
    session_expiry_days: int = int(os.getenv("SESSION_EXPIRY_DAYS", "7"))
    max_login_attempts: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    # Performance
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    # UI Configuration
    max_conversation_turns: int = int(os.getenv("MAX_CONVERSATION_TURNS", "20"))

# Initialize configuration
config = Config()

# ===== PATHS =====
# Create directories for processed documents
PROCESSED_DIR = Path("./processed_documents")
CATEGORIES_DIR = Path("./categorized_documents")
VECTOR_STORE_DIR = Path("./vector_store")
LOGS_DIR = Path("./logs")
UPLOADS_DIR = Path("./uploads")
USER_DATA_DIR = Path("./user_data")
CACHE_DIR = Path("./cache")

# Create directories if they don't exist
for dir_path in [PROCESSED_DIR, CATEGORIES_DIR, VECTOR_STORE_DIR, LOGS_DIR, UPLOADS_DIR, USER_DATA_DIR, CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize Groq client if available
if GROQ_AVAILABLE and Groq is not None and config.groq_api_key:
    try:
        groq_client = Groq(api_key=config.groq_api_key)
    except Exception as e:
        logger.warning(f"Failed to initialize Groq client: {e}")
        groq_client = None
        GROQ_AVAILABLE = False
else:
    groq_client = None

# ===== SECURITY =====
def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    if BCRYPT_AVAILABLE and bcrypt is not None:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    # Fallback: use PBKDF2-HMAC-SHA256
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
    # store as hex: iterations$salt_hex$dk_hex
    return f"100000${salt.hex()}${dk.hex()}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    if BCRYPT_AVAILABLE and bcrypt is not None:
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            return False
    # Fallback: expected format iterations$salt_hex$dk_hex
    try:
        parts = hashed_password.split('$')
        if len(parts) != 3:
            return False
        iterations = int(parts[0])
        salt = bytes.fromhex(parts[1])
        dk_hex = parts[2]
        new_dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
        return secrets.compare_digest(new_dk.hex(), dk_hex)
    except Exception:
        return False

def generate_session_token() -> str:
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not input_str:
        return ""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_str)
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized.strip()

def validate_username(username: str) -> Optional[str]:
    """Validate username format"""
    if not username or len(username) < 3:
        return "Username must be at least 3 characters"
    if len(username) > 20:
        return "Username must be less than 20 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return "Username can only contain letters, numbers, and underscores"
    return None

def validate_password(password: str) -> Optional[str]:
    """Validate password strength"""
    if not password or len(password) < 8:
        return "Password must be at least 8 characters"
    if len(password) > 128:
        return "Password must be less than 128 characters"
    # Check for at least one uppercase, one lowercase, and one digit
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return "Password must contain at least one digit"
    return None

# ===== CUSTOM EXCEPTIONS =====
class RAGError(Exception):
    """Base exception for RAG system"""
    pass

class DatabaseError(RAGError):
    """Database operation error"""
    pass

class AuthenticationError(RAGError):
    """Authentication error"""
    pass

class DocumentProcessingError(RAGError):
    """Document processing error"""
    pass

class ValidationError(RAGError):
    """Input validation error"""
    pass

# ===== DATABASE =====
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect('rag_system.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class DatabaseManager:
    """Manages database operations"""
    def __init__(self):
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        with get_db_connection() as conn:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            ''')
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            # Search history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            # Feedback table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            # Document versions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            conn.commit()

    def execute_query(self, query: str, params: Tuple = (), fetch_one: bool = False, fetch_all: bool = False):
        """Execute a database query with error handling"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Database error: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}")

# ===== USER AUTHENTICATION SYSTEM =====
class UserAuthSystem:
    """Secure user authentication system"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def register_user(self, username: str, password: str, email: str = "") -> bool:
        """Register a new user"""
        # Validate inputs
        username_error = validate_username(username)
        if username_error:
            raise ValidationError(username_error)
        password_error = validate_password(password)
        if password_error:
            raise ValidationError(password_error)

        # Check if username already exists
        existing_user = self.db.execute_query(
            "SELECT id FROM users WHERE username = ?",
            (username,),
            fetch_one=True
        )
        if existing_user:
            raise ValidationError("Username already exists")

        # Hash password
        password_hash = hash_password(password)

        # Insert user
        try:
            user_id = self.db.execute_query(
                "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                (username, password_hash, email)
            )
            return user_id is not None
        except DatabaseError:
            raise DatabaseError("Failed to register user")

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return session token"""
        # Validate inputs
        username_error = validate_username(username)
        if username_error:
            raise ValidationError(username_error)
        if not password:
            raise ValidationError("Password is required")

        # Get user
        user = self.db.execute_query(
            "SELECT id, password_hash, login_attempts, locked_until FROM users WHERE username = ?",
            (username,),
            fetch_one=True
        )
        if not user:
            raise AuthenticationError("Invalid username or password")

        # Check if account is locked
        if user['locked_until'] and datetime.fromisoformat(user['locked_until']) > datetime.now():
            raise AuthenticationError("Account is temporarily locked due to too many failed attempts")

        # Verify password
        if not verify_password(password, user['password_hash']):
            # Increment login attempts
            attempts = user['login_attempts'] + 1
            if attempts >= config.max_login_attempts:
                # Lock account
                locked_until = (datetime.now() + timedelta(minutes=30)).isoformat()
                self.db.execute_query(
                    "UPDATE users SET login_attempts = ?, locked_until = ? WHERE id = ?",
                    (attempts, locked_until, user['id'])
                )
                raise AuthenticationError("Account locked due to too many failed attempts")
            else:
                self.db.execute_query(
                    "UPDATE users SET login_attempts = ? WHERE id = ?",
                    (attempts, user['id'])
                )
                raise AuthenticationError("Invalid username or password")

        # Reset login attempts and update last login
        self.db.execute_query(
            "UPDATE users SET login_attempts = 0, locked_until = NULL, last_login = CURRENT_TIMESTAMP WHERE id = ?",
            (user['id'],)
        )

        # Create session token
        session_token = generate_session_token()
        expires_at = (datetime.now() + timedelta(days=config.session_expiry_days)).isoformat()
        try:
            self.db.execute_query(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (session_token, user['id'], expires_at)
            )
            return session_token
        except DatabaseError:
            raise DatabaseError("Failed to create session")

    def validate_session(self, session_token: str) -> Optional[int]:
        """Validate a session token and return user ID"""
        if not session_token:
            return None

        session = self.db.execute_query(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?",
            (session_token,),
            fetch_one=True
        )
        if not session:
            return None

        # Check if session is expired
        if datetime.fromisoformat(session['expires_at']) < datetime.now():
            # Delete expired session
            self.db.execute_query("DELETE FROM sessions WHERE token = ?", (session_token,))
            return None

        return session['user_id']

    def logout(self, session_token: str) -> bool:
        """Logout a user by invalidating their session"""
        if not session_token:
            return False

        try:
            self.db.execute_query("DELETE FROM sessions WHERE token = ?", (session_token,))
            return True
        except DatabaseError:
            return False

    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information"""
        user = self.db.execute_query(
            "SELECT id, username, email, role, created_at, last_login FROM users WHERE id = ?",
            (user_id,),
            fetch_one=True
        )
        if user:
            return dict(user)
        return None

    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user password"""
        # Validate new password
        password_error = validate_password(new_password)
        if password_error:
            raise ValidationError(password_error)

        # Get current password hash
        user = self.db.execute_query(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,),
            fetch_one=True
        )
        if not user:
            raise AuthenticationError("User not found")

        # Verify current password
        if not verify_password(current_password, user['password_hash']):
            raise AuthenticationError("Current password is incorrect")

        # Hash new password
        new_password_hash = hash_password(new_password)

        # Update password
        try:
            self.db.execute_query(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_password_hash, user_id)
            )
            return True
        except DatabaseError:
            raise DatabaseError("Failed to update password")

# ===== SEARCH HISTORY SYSTEM =====
class SearchHistorySystem:
    """System to store and retrieve search history"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_search(self, user_id: int, query: str, response: str) -> bool:
        """Add a search to the history"""
        try:
            self.db.execute_query(
                "INSERT INTO search_history (user_id, query, response) VALUES (?, ?, ?)",
                (user_id, query, response)
            )
            return True
        except DatabaseError:
            return False

    def get_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get search history for a user"""
        history = self.db.execute_query(
            "SELECT id, query, response, created_at FROM search_history "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
            fetch_all=True
        )
        return [dict(item) for item in history] if history else []

    def clear_history(self, user_id: int) -> bool:
        """Clear search history for a user"""
        try:
            self.db.execute_query("DELETE FROM search_history WHERE user_id = ?", (user_id,))
            return True
        except DatabaseError:
            return False

# ===== FEEDBACK SYSTEM =====
class FeedbackSystem:
    """System to collect and analyze user feedback"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_feedback(self, user_id: int, query: str, response: str, rating: int, comment: str = "") -> bool:
        """Add feedback for a response"""
        # Validate rating
        if not 1 <= rating <= 5:
            raise ValidationError("Rating must be between 1 and 5")

        try:
            self.db.execute_query(
                "INSERT INTO feedback (user_id, query, response, rating, comment) VALUES (?, ?, ?, ?, ?)",
                (user_id, query, response, rating, comment)
            )
            return True
        except DatabaseError:
            return False

    def get_feedback(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get feedback, optionally filtered by user"""
        if user_id:
            feedback = self.db.execute_query(
                "SELECT id, query, response, rating, comment, created_at FROM feedback "
                "WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
                fetch_all=True
            )
        else:
            feedback = self.db.execute_query(
                "SELECT id, query, response, rating, comment, created_at FROM feedback "
                "ORDER BY created_at DESC",
                fetch_all=True
            )
        return [dict(item) for item in feedback] if feedback else []

    def get_average_rating(self, user_id: Optional[int] = None) -> float:
        """Get average rating, optionally filtered by user"""
        if user_id:
            result = self.db.execute_query(
                "SELECT AVG(rating) as avg_rating FROM feedback WHERE user_id = ?",
                (user_id,),
                fetch_one=True
            )
        else:
            result = self.db.execute_query(
                "SELECT AVG(rating) as avg_rating FROM feedback",
                fetch_one=True
            )
        if result and result['avg_rating'] is not None:
            return float(result['avg_rating'])
        return 0.0

# ===== DOCUMENT VERSIONING SYSTEM =====
class DocumentVersioningSystem:
    """Document versioning system"""
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.uploads_dir = Path("./uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)

    def add_version(self, file_path: Path, user_id: int, comment: str = "") -> str:
        """Add a new version of a document"""
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")

        filename = file_path.name
        file_hash = self._calculate_file_hash(file_path)

        # Generate version ID
        version_id = str(uuid.uuid4())
        version_path = self.uploads_dir / f"{version_id}_{filename}"

        # Copy file to versioned location
        shutil.copy2(file_path, version_path)

        try:
            self.db.execute_query(
                "INSERT INTO document_versions (filename, version_id, file_path, file_hash, user_id, comment) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (filename, version_id, str(version_path), file_hash, user_id, comment)
            )
            return version_id
        except DatabaseError:
            # Clean up copied file if database operation fails
            if version_path.exists():
                version_path.unlink()
            raise DatabaseError("Failed to add document version")

    def get_versions(self, filename: str) -> List[Dict[str, Any]]:
        """Get all versions of a document"""
        versions = self.db.execute_query(
            "SELECT dv.id, dv.version_id, dv.file_path, dv.file_hash, dv.comment, dv.created_at, u.username "
            "FROM document_versions dv "
            "JOIN users u ON dv.user_id = u.id "
            "WHERE dv.filename = ? "
            "ORDER BY dv.created_at DESC",
            (filename,),
            fetch_all=True
        )
        return [dict(item) for item in versions] if versions else []

    def get_latest_version(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a document"""
        version = self.db.execute_query(
            "SELECT dv.id, dv.version_id, dv.file_path, dv.file_hash, dv.comment, dv.created_at, u.username "
            "FROM document_versions dv "
            "JOIN users u ON dv.user_id = u.id "
            "WHERE dv.filename = ? "
            "ORDER BY dv.created_at DESC "
            "LIMIT 1",
            (filename,),
            fetch_one=True
        )
        return dict(version) if version else None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

# ===== MEMORY MONITOR =====
class MemoryMonitor:
    """Monitor memory usage and provide warnings"""
    @staticmethod
    def get_memory_usage():
        """Get current memory usage as a fraction of total memory"""
        if not PSUTIL_AVAILABLE or psutil is None:
            # psutil not available on the deployment environment (e.g., Streamlit cloud)
            # Return a conservative low usage value and log a debug message.
            logger.debug("psutil not available; returning 0 memory usage fraction")
            return 0.0

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / psutil.virtual_memory().total

    @staticmethod
    def check_memory():
        """Check if memory usage is within limits"""
        usage = MemoryMonitor.get_memory_usage()
        if usage > config.max_memory_usage:
            logger.warning(f"High memory usage: {usage:.2%}")
            return False
        return True

    @staticmethod
    def gc_collect():
        """Force garbage collection"""
        import gc
        gc.collect()

# ===== CACHE FUNCTIONS =====
def cache_result(cache_key: str, result: Any, expiry_days: int = None):
    """Cache a result with expiry"""
    if expiry_days is None:
        expiry_days = config.cache_expiry_days

    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    expiry_time = datetime.now() + timedelta(days=expiry_days)

    with open(cache_file, 'wb') as f:
        pickle.dump({
            'result': result,
            'expiry': expiry_time
        }, f)

def get_cached_result(cache_key: str) -> Any:
    """Get a cached result if it exists and is not expired"""
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        if datetime.now() < cached_data['expiry']:
            return cached_data['result']
        else:
            # Cache expired, remove it
            cache_file.unlink()
            return None
    except Exception as e:
        logger.error(f"Error retrieving cached result: {str(e)}")
        return None

# ===== DOCUMENT PROCESSOR =====
class DocumentProcessor:
    """Document processor with semi-structured RAG support"""
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.png': self._process_image,
            '.jpg': self._process_image,
            '.jpeg': self._process_image
        }
        self.processing_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'by_type': {},
            'tables_extracted': 0,
            'table_summaries_generated': 0
        }

    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a document based on its file type"""
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()
        logger.info(f"Processing document: {file_path.name}")

        self.processing_stats['total_files'] += 1

        if file_extension not in self.processing_stats['by_type']:
            self.processing_stats['by_type'][file_extension] = {'total': 0, 'success': 0, 'failed': 0}

        self.processing_stats['by_type'][file_extension]['total'] += 1

        if file_extension not in self.supported_formats:
            error_msg = f"Unsupported file type: {file_extension}"
            logger.error(error_msg)
            self.processing_stats['by_type'][file_extension]['failed'] += 1
            self.processing_stats['failed'] += 1
            return {
                "path": str(file_path),
                "filename": file_path.name,
                "file_type": file_extension,
                "content": "",
                "metadata": {},
                "error": error_msg,
                "success": False
            }

        try:
            result = self.supported_formats[file_extension](file_path)
            result["success"] = True
            self.processing_stats['by_type'][file_extension]['success'] += 1
            self.processing_stats['successful'] += 1
            logger.info(f"Successfully processed {file_path.name}")
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing {file_path.name}: {error_msg}")
            self.processing_stats['by_type'][file_extension]['failed'] += 1
            self.processing_stats['failed'] += 1
            return {
                "path": str(file_path),
                "filename": file_path.name,
                "file_type": file_extension,
                "content": "",
                "metadata": {},
                "error": error_msg,
                "success": False
            }

    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF files with advanced text extraction"""
        logger.info(f"Processing PDF: {file_path.name}")
        try:
            # Try to use Unstructured for better table extraction
            return self._process_pdf_with_unstructured(file_path)
        except ImportError:
            logger.warning("Unstructured not available, using standard PDF processing")
            return self._process_pdf_standard(file_path)
        except Exception as e:
            logger.error(f"Error processing PDF with Unstructured {file_path.name}: {str(e)}")
            # Fallback to standard PDF processing
            return self._process_pdf_standard(file_path)

    def _process_pdf_with_unstructured(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF files using Unstructured for better table extraction"""
        from unstructured.partition.pdf import partition_pdf

        # Process PDF with Unstructured
        elements = partition_pdf(
            filename=str(file_path),
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )

        # Separate elements into text and tables
        text_elements = []
        table_elements = []
        for element in elements:
            element_type = str(type(element))
            if "Table" in element_type:
                table_elements.append(str(element))
                self.processing_stats['tables_extracted'] += 1
            elif "CompositeElement" in element_type or "Text" in element_type:
                text_elements.append(str(element))

        # Combine text elements
        text_content = "\n".join(text_elements)

        # Extract metadata
        metadata = {
            "pages": len(elements),
            "tables_extracted": len(table_elements),
            "text_chunks": len(text_elements)
        }

        return {
            "path": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "content": text_content,
            "tables": table_elements,
            "metadata": metadata,
            "error": None,
            "success": True
        }

    def _process_pdf_standard(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF files using standard methods"""
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        text = ""
        metadata = {
            "pages": len(reader.pages),
            "title": reader.metadata.get('/Title', ''),
            "author": reader.metadata.get('/Author', ''),
            "creator": reader.metadata.get('/Creator', '')
        }

        # Extract text from each page
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"--- Page {i+1} ---\n"
            text += page_text
            text += "\n"

        # If text extraction is poor, try OCR
        if len(text.strip()) < 100:
            logger.info(f"Using OCR for PDF: {file_path.name}")
            text = self._extract_text_with_ocr(file_path)
            metadata["ocr_used"] = True

        return {
            "path": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "content": text,
            "metadata": metadata,
            "error": None,
            "success": True
        }

    def _extract_text_with_ocr(self, file_path: Path) -> str:
        """Extract text from PDF using OCR"""
        try:
            import pdf2image
            import pytesseract

            # Convert PDF to images
            images = pdf2image.convert_from_path(str(file_path))

            # Extract text from each image
            text = ""
            for i, img in enumerate(images):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    img.save(tmp.name)
                    text += f"--- Page {i+1} ---\n"
                    text += pytesseract.image_to_string(Image.open(tmp.name))
                    text += "\n"
                    os.unlink(tmp.name)

            return text
        except Exception as e:
            logger.error(f"OCR extraction failed for {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"OCR extraction failed: {str(e)}")

    def _process_docx(self, file_path: Path) -> Dict[str, Any]:
        """Process DOCX files"""
        try:
            import docx2txt
            text = docx2txt.process(str(file_path))

            # Extract metadata if possible
            metadata = {
                "word_count": len(text.split()),
                "char_count": len(text)
            }

            return {
                "path": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "content": text,
                "metadata": metadata,
                "error": None,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"Error processing DOCX: {str(e)}")

    def _process_text(self, file_path: Path) -> Dict[str, Any]:
        """Process plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            metadata = {
                "word_count": len(text.split()),
                "char_count": len(text)
            }

            return {
                "path": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "content": text,
                "metadata": metadata,
                "error": None,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing text file {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"Error processing text file: {str(e)}")

    def _process_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Process Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # Convert markdown to plain text for better search
            import markdown
            html = markdown.markdown(text)
            plain_text = re.sub(r'<[^>]+>', '', html)

            metadata = {
                "word_count": len(text.split()),
                "char_count": len(text)
            }

            return {
                "path": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "content": plain_text,
                "metadata": metadata,
                "error": None,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing Markdown {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"Error processing Markdown: {str(e)}")

    def _process_image(self, file_path: Path) -> Dict[str, Any]:
        """Process image files with OCR and feature extraction"""
        try:
            import pytesseract
            img = Image.open(file_path)

            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(img)

            # Extract image features
            img_features = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }

            return {
                "path": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
                "content": ocr_text,
                "metadata": img_features,
                "error": None,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing image {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"Error processing image: {str(e)}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats

    def visualize_processing_stats(self):
        """Visualize processing statistics"""
        stats = self.get_processing_stats()

        # Handle empty stats
        if not stats or stats.get('total_files', 0) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No processing statistics available.<br>Please process documents first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title_text="Document Processing Statistics",
                height=800
            )
            return fig

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Processing Results', 'Files by Type', 'Success Rate by Type', 'Processing Summary'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )

        # Processing Results pie chart
        fig.add_trace(
            go.Pie(
                labels=['Successful', 'Failed'],
                values=[stats.get('successful', 0), stats.get('failed', 0)],
                name="Processing Results"
            ),
            row=1, col=1
        )

        # Files by Type bar chart
        by_type = stats.get('by_type', {})
        file_types = list(by_type.keys())
        file_counts = [by_type.get(ft, {}).get('total', 0) for ft in file_types]
        fig.add_trace(
            go.Bar(
                x=file_types,
                y=file_counts,
                name="Files by Type",
                marker_color='rgb(55, 83, 109)'
            ),
            row=1, col=2
        )

        # Success Rate by Type
        success_rates = []
        for ft in file_types:
            type_stats = by_type.get(ft, {})
            total = type_stats.get('total', 0)
            if total > 0:
                success_rates.append((type_stats.get('success', 0) / total) * 100)
            else:
                success_rates.append(0)

        fig.add_trace(
            go.Bar(
                x=file_types,
                y=success_rates,
                name="Success Rate (%)",
                marker_color='rgb(26, 118, 255)'
            ),
            row=2, col=1
        )

        # Processing Summary
        total_files = stats.get('total_files', 0)
        successful = stats.get('successful', 0)
        success_rate = (successful / max(1, total_files)) * 100

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=success_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Success Rate"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Document Processing Statistics",
            showlegend=False
        )

        return fig

# ===== DOCUMENT CATEGORIZER =====
class DocumentCategorizer:
    """Categorize documents using Groq's model"""
    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.categories = [
            "Technical Manuals",
            "Troubleshooting Guides",
            "Installation Procedures",
            "Maintenance Procedures",
            "Safety Guidelines",
            "Parts Diagrams",
            "Claim Cases",
            "Uncategorized"
        ]
        self.categorization_stats = {
            'total': 0,
            'by_category': {cat: 0 for cat in self.categories},
            'errors': 0
        }

    def categorize_document(self, document: Dict[str, Any]) -> str:
        """Categorize a document based on its content"""
        logger.info(f"Categorizing document: {document['filename']}")

        self.categorization_stats['total'] += 1

        if not document.get("content"):
            logger.warning(f"No content to categorize for {document['filename']}")
            self.categorization_stats['by_category']['Uncategorized'] += 1
            return "Uncategorized"

        # Prepare content for categorization
        content = document.get("content", "")

        # Truncate if too long
        if len(content) > 3000:
            content = content[:3000] + "..."

        # Create prompt for categorization
        prompt = f"""
        You are an expert in Nakakita air purge level gauge systems. Please categorize the following document content into one of these categories:

        1. Technical Manuals - Detailed technical specifications, operating principles, and system descriptions
        2. Troubleshooting Guides - Step-by-step instructions for diagnosing and fixing problems
        3. Installation Procedures - Instructions for installing the equipment
        4. Maintenance Procedures - Routine maintenance tasks and schedules
        5. Safety Guidelines - Safety precautions and warnings
        6. Parts Diagrams - Diagrams showing components and part numbers
        7. Claim Cases - Documentation of previous claims or issues
        8. Uncategorized - If none of the above categories fit

        Document content:
        {content}

        Only respond with the category name and nothing else.
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20
            )
            category = response.choices[0].message.content.strip()

            # Validate category
            if category in self.categories:
                logger.info(f"Categorized {document['filename']} as {category}")
                self.categorization_stats['by_category'][category] += 1
                return category
            else:
                # Try to match partial category names
                for valid_category in self.categories:
                    if valid_category.lower() in category.lower():
                        logger.info(f"Partially matched {document['filename']} as {valid_category}")
                        self.categorization_stats['by_category'][valid_category] += 1
                        return valid_category

                logger.warning(f"Could not categorize {document['filename']}, response: {category}")
                self.categorization_stats['by_category']['Uncategorized'] += 1
                return "Uncategorized"

        except Exception as e:
            logger.error(f"Error categorizing document {document['filename']}: {str(e)}")
            self.categorization_stats['errors'] += 1
            self.categorization_stats['by_category']['Uncategorized'] += 1
            return "Uncategorized"

    def categorize_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize a list of documents"""
        logger.info(f"Categorizing {len(documents)} documents")

        categorized_docs = []
        for doc in documents:
            category = self.categorize_document(doc)
            doc["category"] = category
            categorized_docs.append(doc)

            # Copy file to category directory
            try:
                category_dir = CATEGORIES_DIR / category
                os.makedirs(category_dir, exist_ok=True)
                shutil.copy2(doc["path"], category_dir / doc["filename"])
                logger.info(f"Copied {doc['filename']} to {category_dir}")
            except Exception as e:
                logger.error(f"Error copying file to category directory: {str(e)}")

        logger.info(f"Categorized {len(categorized_docs)} documents")
        return categorized_docs

    def get_categorization_stats(self) -> Dict[str, Any]:
        """Get categorization statistics"""
        return self.categorization_stats

    def visualize_categorization_stats(self):
        """Visualize categorization statistics"""
        stats = self.get_categorization_stats()

        # Handle empty stats
        if not stats or stats.get('total', 0) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No categorization statistics available.<br>Please process documents first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title_text="Document Categorization Statistics",
                height=600
            )
            return fig

        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Documents by Category', 'Categorization Summary'),
            specs=[[{"type": "pie"}, {"type": "indicator"}]]
        )

        # Documents by Category pie chart
        by_category = stats.get('by_category', {})
        categories = list(by_category.keys())
        counts = [by_category.get(cat, 0) for cat in categories]

        fig.add_trace(
            go.Pie(
                labels=categories,
                values=counts,
                name="Documents by Category"
            ),
            row=1, col=1
        )

        # Categorization Summary
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=stats.get('total', 0),
                title={'text': "Total Documents"},
                delta={'reference': len(categories)},
                number={'suffix': " docs"}
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=600,
            title_text="Document Categorization Statistics",
            showlegend=False
        )

        return fig

# ===== TEXT CHUNKER =====
class AdvancedTextChunker:
    """Advanced text chunking with context preservation"""
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.overlap = overlap or config.chunk_overlap
        # Use LangChain splitter when available, otherwise use a simple fallback
        if LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter is not None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            # Fallback splitter: naive paragraph/sentence based splitter
            class SimpleSplitter:
                def __init__(self, chunk_size, overlap):
                    self.chunk_size = chunk_size
                    self.overlap = overlap

                def split_text(self, text):
                    # Split into paragraphs first
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    chunks = []
                    for p in paragraphs:
                        if len(p) <= self.chunk_size:
                            chunks.append(p)
                        else:
                            # split into sliding windows of words
                            words = p.split()
                            i = 0
                            while i < len(words):
                                chunk = ' '.join(words[i:i+self.chunk_size//5])
                                chunks.append(chunk)
                                i += max(1, (self.chunk_size//5) - (self.overlap//5))
                    return chunks

            self.text_splitter = SimpleSplitter(self.chunk_size, self.overlap)
        self.chunking_stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunks_per_doc': 0,
            'by_category': {},
            'tables_processed': 0
        }

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces while preserving context"""
        logger.info(f"Chunking document: {document['filename']}")

        content = document.get("content", "")
        tables = document.get("tables", [])
        category = document.get("category", "Uncategorized")

        if not content.strip() and not tables:
            logger.warning(f"No content to chunk for {document['filename']}")
            return []

        # Update stats
        self.chunking_stats['total_documents'] += 1

        if category not in self.chunking_stats['by_category']:
            self.chunking_stats['by_category'][category] = {'documents': 0, 'chunks': 0, 'tables': 0}

        self.chunking_stats['by_category'][category]['documents'] += 1

        # Process text content
        text_chunks = []
        if content.strip():
            text_chunks = self.text_splitter.split_text(content)

        # Create chunk objects
        chunk_objects = []

        # Add text chunks
        for i, chunk in enumerate(text_chunks):
            chunk_obj = {
                "doc_id": document["path"],
                "filename": document["filename"],
                "category": category,
                "chunk_index": i,
                "text": chunk,
                "content_type": "text",
                "metadata": {
                    "source": document["path"],
                    "file_type": document.get("file_type", "")
                }
            }
            chunk_objects.append(chunk_obj)

        # Add tables as separate chunks
        for i, table in enumerate(tables):
            chunk_obj = {
                "doc_id": document["path"],
                "filename": document["filename"],
                "category": category,
                "chunk_index": len(text_chunks) + i,
                "text": table,  # Raw table content
                "content_type": "table",
                "metadata": {
                    "source": document["path"],
                    "file_type": document.get("file_type", ""),
                    "table_index": i
                }
            }
            chunk_objects.append(chunk_obj)
            self.chunking_stats['tables_processed'] += 1
            self.chunking_stats['by_category'][category]['tables'] += 1

        # Update stats
        self.chunking_stats['total_chunks'] += len(chunk_objects)
        self.chunking_stats['by_category'][category]['chunks'] += len(chunk_objects)
        self.chunking_stats['avg_chunks_per_doc'] = self.chunking_stats['total_chunks'] / self.chunking_stats['total_documents']

        logger.info(f"Chunked {document['filename']} into {len(chunk_objects)} chunks (text: {len(text_chunks)}, tables: {len(tables)})")

        return chunk_objects

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents"""
        logger.info(f"Chunking {len(documents)} documents")

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} documents")

        return all_chunks

    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return self.chunking_stats

    def visualize_chunking_stats(self):
        """Visualize chunking statistics"""
        stats = self.get_chunking_stats()

        # Handle empty stats
        if not stats or stats.get('total_documents', 0) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No chunking statistics available.<br>Please process documents first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title_text="Document Chunking Statistics",
                height=800
            )
            return fig

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Chunks by Category', 'Avg Chunks per Document', 'Chunking Summary', 'Documents vs Chunks'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Chunks by Category bar chart
        by_category = stats.get('by_category', {})
        categories = list(by_category.keys())
        chunk_counts = [by_category.get(cat, {}).get('chunks', 0) for cat in categories]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=chunk_counts,
                name="Chunks by Category",
                marker_color='rgb(55, 83, 109)'
            ),
            row=1, col=1
        )

        # Avg Chunks per Document
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=stats.get('avg_chunks_per_doc', 0),
                title={'text': "Avg Chunks per Document"},
                number={'valueformat': ".2f"}
            ),
            row=1, col=2
        )

        # Documents vs Chunks scatter
        doc_counts = [by_category.get(cat, {}).get('documents', 0) for cat in categories]

        fig.add_trace(
            go.Scatter(
                x=doc_counts,
                y=chunk_counts,
                mode='markers',
                marker=dict(size=10, color=chunk_counts, colorscale='Viridis', showscale=True),
                text=categories,
                name="Documents vs Chunks"
            ),
            row=2, col=1
        )

        # Chunking Summary
        fig.add_trace(
            go.Bar(
                x=['Total Documents', 'Total Chunks'],
                y=[stats.get('total_documents', 0), stats.get('total_chunks', 0)],
                name="Chunking Summary",
                marker_color='rgb(26, 118, 255)'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Document Chunking Statistics",
            showlegend=False
        )

        return fig

# ===== EMBEDDING GENERATOR =====
class MultiModalEmbeddingGenerator:
    """Generate embeddings for text and images using Ollama"""
    def __init__(self):
        self.text_model = config.text_model
        self.embedding_stats = {
            'total_text_embeddings': 0,
            'table_summaries_generated': 0,
            'errors': 0
        }
        
        # Configure Ollama client for remote connection
        import ollama
        self.ollama_client = ollama.Client(host=config.ollama_base_url)
        logger.info(f"Using Ollama at {config.ollama_base_url}")
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Ollama"""
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            if NUMPY_AVAILABLE:
                return np.zeros(768, dtype=np.float32)
            else:
                return []

        # If Ollama is not available, return a zero vector (graceful degradation)
        if not OLLAMA_AVAILABLE or ollama is None:
            logger.warning("Ollama not available; returning zero-vector embedding")
            if NUMPY_AVAILABLE:
                self.embedding_stats['errors'] += 1
                return np.zeros(768, dtype=np.float32)
            else:
                self.embedding_stats['errors'] += 1
                return []

        try:
            response = ollama.embeddings(model=self.text_model, prompt=text)
            embedding = np.array(response["embedding"], dtype=np.float32)
            self.embedding_stats['total_text_embeddings'] += 1
            logger.debug(f"Generated text embedding of shape {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            self.embedding_stats['errors'] += 1
            if NUMPY_AVAILABLE:
                return np.zeros(768, dtype=np.float32)
            return []

    def generate_table_summary(self, table: str) -> str:
        """Generate a summary for a table"""
        if not table.strip():
            return ""

        # Create prompt for table summarization
        prompt = f"""
        You are an assistant tasked with summarizing tables.
        Give a concise summary of the table that captures its key information and structure.
        Table: {table}
        """

        try:
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Using Groq model for summarization
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            summary = response.choices[0].message.content.strip()
            self.embedding_stats['table_summaries_generated'] += 1
            return summary
        except Exception as e:
            logger.error(f"Error generating table summary: {str(e)}")
            return table[:500] + "..." if len(table) > 500 else table

    def generate_document_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate embeddings for document chunks with table summarization"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embeddings = []
        enhanced_chunks = []

        for chunk in chunks:
            # Determine what to embed based on content type
            if chunk.get("content_type") == "table":
                # Generate summary for table and embed the summary
                summary = self.generate_table_summary(chunk["text"])
                text_to_embed = summary
                enhanced_chunk = chunk.copy()
                enhanced_chunk["summary"] = summary
            else:
                # For text chunks, embed the text directly
                text_to_embed = chunk["text"]
                enhanced_chunk = chunk.copy()

            # Generate embedding
            embedding = self.generate_text_embedding(text_to_embed)
            embeddings.append(embedding)
            enhanced_chunk["has_embedding"] = True
            enhanced_chunks.append(enhanced_chunk)

        logger.info(f"Generated {len(embeddings)} embeddings")

        return np.array(embeddings, dtype=np.float32), enhanced_chunks

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return self.embedding_stats

    def visualize_embedding_stats(self):
        """Visualize embedding statistics"""
        stats = self.get_embedding_stats()

        # Handle empty stats
        if not stats or stats.get('total_text_embeddings', 0) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No embedding statistics available.<br>Please process documents first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title_text="Embedding Generation Statistics",
                height=400
            )
            return fig

        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Text Embeddings', 'Table Summaries'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}]]
        )

        # Text Embeddings
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=stats.get('total_text_embeddings', 0),
                title={'text': "Text Embeddings"},
                number={'suffix': " embeddings"}
            ),
            row=1, col=1
        )

        # Table Summaries
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=stats.get('table_summaries_generated', 0),
                title={'text': "Table Summaries"},
                number={'suffix': " summaries"}
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text="Embedding Generation Statistics",
            showlegend=False
        )

        return fig

# ===== VECTOR STORE =====
class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.is_built = False
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0,
            'avg_results': 0
        }

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Build FAISS index from embeddings and metadata"""
        logger.info(f"Building FAISS index with {len(embeddings)} embeddings")

        if len(embeddings) == 0:
            logger.error("Cannot build index with empty embeddings")
            raise ValueError("Cannot build index with empty embeddings")

        # Check embeddings shape and type
        logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(normalized_embeddings)

        # Add embeddings to index
        self.index.add(normalized_embeddings)

        # Store metadata
        self.metadata = metadata
        self.is_built = True

        # Save index and metadata
        self._save_index()

        logger.info(f"Built FAISS index with {len(embeddings)} embeddings")

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings to an existing index"""
        logger.info(f"Adding {len(embeddings)} embeddings to existing index")

        if not self.is_built:
            logger.error("Index has not been built yet")
            raise ValueError("Index has not been built yet")

        if len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(normalized_embeddings)

        # Add embeddings to index
        self.index.add(normalized_embeddings)

        # Add metadata
        self.metadata.extend(metadata)

        # Save index and metadata
        self._save_index()

        logger.info(f"Added {len(embeddings)} embeddings to index")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        start_time = time.time()

        if not self.is_built:
            logger.error("Index has not been built yet")
            raise ValueError("Index has not been built yet")

        # Normalize query embedding
        normalized_query = query_embedding.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(normalized_query)

        # Search index
        distances, indices = self.index.search(normalized_query, k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["relevance_score"] = float(distances[0][i])
                results.append(result)

        # Update search stats
        search_time = time.time() - start_time
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
            self.search_stats['total_searches']
        )
        self.search_stats['avg_results'] = (
            (self.search_stats['avg_results'] * (self.search_stats['total_searches'] - 1) + len(results)) /
            self.search_stats['total_searches']
        )

        logger.debug(f"Search completed in {search_time:.4f}s, found {len(results)} results")

        return results

    def _save_index(self) -> None:
        """Save index and metadata to disk"""
        if not self.is_built:
            return

        # Save FAISS index
        faiss.write_index(self.index, str(VECTOR_STORE_DIR / "faiss_index.bin"))

        # Save metadata
        with open(VECTOR_STORE_DIR / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Saved FAISS index and metadata to disk")

    def load_index(self) -> bool:
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            index_path = VECTOR_STORE_DIR / "faiss_index.bin"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info("Loaded FAISS index from disk")
            else:
                logger.info("FAISS index file not found - will need to process documents first")
                return False

            # Load metadata
            metadata_path = VECTOR_STORE_DIR / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info("Loaded metadata from disk")
            else:
                logger.warning("Metadata file not found - starting with empty metadata")
                self.metadata = []

            self.is_built = True
            logger.info(f"Successfully loaded FAISS index with {len(self.metadata)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            logger.info("Resetting vector store state due to loading error")
            self.index = None
            self.metadata = []
            self.is_built = False
            return False

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return self.search_stats

    def visualize_search_stats(self):
        """Visualize search statistics"""
        stats = self.get_search_stats()

        # Handle empty stats
        if not stats or stats.get('total_searches', 0) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No search statistics available.<br>Please perform a search first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title_text="Vector Store Search Statistics",
                height=400
            )
            return fig

        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Searches', 'Average Search Time'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}]]
        )

        # Total Searches
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=stats.get('total_searches', 0),
                title={'text': "Total Searches"},
                number={'suffix': " searches"}
            ),
            row=1, col=1
        )

        # Average Search Time
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=stats.get('avg_search_time', 0),
                title={'text': "Average Search Time"},
                number={'suffix': " seconds", 'valueformat': ".4f"}
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text="Vector Store Search Statistics",
            showlegend=False
        )

        return fig

# ===== RAG STATE =====
class RAGState(TypedDict):
    """State for the RAG workflow"""
    query: str
    relevant_chunks: List[Dict[str, Any]]
    response: str
    processing_stats: Dict[str, Any]
    error: Optional[str]

# ===== RAG SYSTEM =====
class EnhancedRAGSystem:
    """Complete multi-modal RAG system with LangGraph workflow"""
    def __init__(self):
        # Initialize database manager
        self.db_manager = DatabaseManager()

        # Initialize systems
        self.auth_system = UserAuthSystem(self.db_manager)
        self.search_history = SearchHistorySystem(self.db_manager)
        self.feedback_system = FeedbackSystem(self.db_manager)
        self.versioning_system = DocumentVersioningSystem(self.db_manager)

        # Initialize processors
        self.document_processor = DocumentProcessor()
        self.categorizer = DocumentCategorizer(groq_client)
        self.chunker = AdvancedTextChunker()
        self.embedding_generator = MultiModalEmbeddingGenerator()
        self.vector_store = FAISSVectorStore()

        # Create workflow
        self.workflow = self._create_workflow()
        self.initialized = False

        # Try to load existing vector store
        if self.vector_store.load_index():
            self.initialized = True
            logger.info("Loaded existing vector store")

        # Initialize task queue for async processing
        self.task_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.processing_thread.start()

        # Initialize progress tracking
        self.progress = {
            'current': 0,
            'total': 0,
            'status': 'Idle',
            'errors': []
        }

    def _create_workflow(self):
        """Create the LangGraph workflow for answering queries"""
        # Use LangGraph StateGraph when available; otherwise provide a minimal fallback
        if LANGGRAPH_AVAILABLE and StateGraph is not None:
            workflow = StateGraph(RAGState)

            # Add nodes
            workflow.add_node("retrieve_relevant_chunks", self._retrieve_relevant_chunks_node)
            workflow.add_node("generate_response", self._generate_response_node)

            # Set the entry point
            workflow.set_entry_point("retrieve_relevant_chunks")

            # Add edges
            workflow.add_edge("retrieve_relevant_chunks", "generate_response")
            workflow.add_edge("generate_response", END)

            # Compile the workflow
            return workflow.compile()

        # Minimal fallback workflow when langgraph isn't available
        class MinimalWorkflow:
            def __init__(self):
                self.nodes = []
                self.entry = None

            def add_node(self, name, func):
                self.nodes.append((name, func))

            def set_entry_point(self, name):
                self.entry = name

            def add_edge(self, a, b):
                # edges are implicit in node order for this minimal workflow
                pass

            def compile(self):
                nodes = list(self.nodes)

                class Runner:
                    def __init__(self, nodes):
                        self.nodes = nodes

                    def invoke(self, state):
                        for name, func in self.nodes:
                            try:
                                state = func(state)
                            except Exception as e:
                                logger.error(f"Workflow node '{name}' failed: {e}")
                                state['error'] = str(e)
                                return state
                        return state

                return Runner(nodes)

        wf = MinimalWorkflow()
        wf.add_node("retrieve_relevant_chunks", self._retrieve_relevant_chunks_node)
        wf.add_node("generate_response", self._generate_response_node)
        wf.set_entry_point("retrieve_relevant_chunks")
        return wf.compile()

    def _retrieve_relevant_chunks_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant chunks node"""
        logger.info("Retrieving relevant chunks node")

        try:
            # Check if query is empty
            if not state["query"].strip():
                logger.warning("Empty query provided")
                state["relevant_chunks"] = []
                return state

            # Check if index is built
            if not self.vector_store.is_built:
                logger.error("Index has not been built yet")
                state["error"] = "Index has not been built yet"
                return state

            # Generate query embedding
            query_embedding = self.embedding_generator.generate_text_embedding(state["query"])

            # Search vector store
            relevant_chunks = self.vector_store.search(query_embedding)

            state["relevant_chunks"] = relevant_chunks

            return state

        except Exception as e:
            logger.error(f"Error in retrieve_relevant_chunks_node: {str(e)}")
            state["error"] = str(e)
            return state

    def _generate_response_node(self, state: RAGState) -> RAGState:
        """Generate response node"""
        logger.info("Generating response node")

        try:
            response = self._generate_response(
                state["query"],
                state["relevant_chunks"]
            )
            state["response"] = response
            return state
        except Exception as e:
            logger.error(f"Error in generate_response_node: {str(e)}")
            state["error"] = str(e)
            return state

    def _generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]],
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a response using Ollama with conversation history"""
        # Prepare context from relevant chunks
        context = "RELEVANT INFORMATION:\n"
        for chunk in relevant_chunks:
            content_type = chunk.get("content_type", "text")
            if content_type == "table":
                # For tables, include both summary and raw table
                context += f"From {chunk['filename']} (Category: {chunk['category']}, Table):\n"
                if "summary" in chunk:
                    context += f"Summary: {chunk['summary']}\n"
                context += f"Table: {chunk['text']}\n"
            else:
                context += f"From {chunk['filename']} (Category: {chunk['category']}):\n"
                context += f"{chunk['text']}\n"

        # Add conversation history if available
        history_text = ""
        if conversation_history:
            history_text = "CONVERSATION HISTORY:\n"
            for turn in conversation_history[-config.max_conversation_turns:]:
                history_text += f"User: {turn['query']}\n"
                history_text += f"Assistant: {turn['response']}\n"

        # Create prompt for LLM
        prompt = f"""
        You are an expert technician specializing in Nakakita air purge level gauge systems. You are helpful, knowledgeable, and conversational.

        {history_text}

        CURRENT USER QUERY: {query}

        {context}

        INSTRUCTIONS:
        1. Answer the user's query in the most detailed way possible based ONLY on the provided information from the data source.
        2. If the information is insufficient, clearly state what additional information is needed.
        3. Include specific steps, technical details, and references to the documentation when possible.
        4. If you don't know the answer based on the provided context, say "I don't have enough information to answer this question."
        5. Do not make up information or hallucinate.
        6. Keep your response conversational and engaging.
        7. Remember the conversation history and use it to provide context-aware responses.
        8. If the user refers to something mentioned earlier in the conversation, acknowledge it and build upon it.
        9. DO REMEMBER THE CONTEXT OF THE CONVERSATION.
        """

        # If Ollama is not available, return a helpful message instead of raising
        if not OLLAMA_AVAILABLE or ollama is None:
            logger.warning("OLLAMA not available; returning placeholder response")
            return "LLM backend is not available in this environment. Please configure Ollama or another LLM provider."

        try:
            response = ollama.chat(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a user query and return a response"""
        logger.info(f"Processing query: {query}")

        # Check if the vector store is built
        if not self.vector_store.is_built:
            logger.warning("Vector store not built, attempting to process documents")
            success = self.process_documents()
            if not success:
                return {
                    "query": query,
                    "relevant_chunks": [],
                    "response": "Error: The document index has not been built. Please process documents first.",
                    "processing_stats": {},
                    "error": "Index not built and document processing failed"
                }

        # Initialize state
        state = {
            "query": query,
            "relevant_chunks": [],
            "response": "",
            "processing_stats": {},
            "error": None
        }

        # Run the workflow
        result = self.workflow.invoke(state)

        # Generate response with conversation history
        if result.get("relevant_chunks"):
            response = self._generate_response(
                query,
                result["relevant_chunks"],
                conversation_history
            )
            result["response"] = response

        return result

    def process_documents(self, force_reprocess: bool = False) -> bool:
        """Process all documents in the specified paths using batch processing"""
        logger.info("Processing documents with batch processing")

        # Check if we can load existing index
        if not force_reprocess and self.vector_store.load_index():
            logger.info("Loaded existing vector store")
            self.initialized = True
            return True

        # Check if data paths exist and contain files (including subfolders)
        found_files = False
        all_files = []
        for data_path in config.data_paths:
            path = Path(data_path)
            if not path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                continue

            # Search recursively through all subfolders
            files = []
            for file_path in path.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # Check if it's a supported file type
                    if file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md', '.png', '.jpg', '.jpeg']:
                        files.append(file_path)

            if not files:
                logger.warning(f"No supported files found in path: {data_path}")
                continue

            logger.info(f"Found {len(files)} supported files in path: {data_path} (including subfolders)")
            all_files.extend(files)
            found_files = True

        if not found_files:
            logger.error("No supported files found in any of the specified paths")
            return False

        # Process files in batches
        total_files = len(all_files)
        logger.info(f"Processing {total_files} files in batches of {config.batch_size}")

        # Initialize progress tracking
        self.progress = {
            'current': 0,
            'total': total_files,
            'status': 'Processing',
            'errors': []
        }

        # Initialize vector store with first batch
        first_batch = all_files[:config.batch_size]
        if not self._process_batch(first_batch, is_first_batch=True):
            logger.error("Failed to process first batch")
            return False

        # Process remaining batches
        for i in range(config.batch_size, total_files, config.batch_size):
            batch = all_files[i:i+config.batch_size]

            # Check memory usage
            if not MemoryMonitor.check_memory():
                logger.warning("High memory usage, forcing garbage collection")
                MemoryMonitor.gc_collect()

            # Process batch
            if not self._process_batch(batch, is_first_batch=False):
                logger.error(f"Failed to process batch starting at index {i}")
                continue

            # Update progress
            self.progress['current'] = min(i + config.batch_size, total_files)

        self.initialized = True
        self.progress['status'] = 'Completed'

        logger.info("All documents processed successfully")

        return True

    def _process_batch(self, file_paths: List[Path], is_first_batch: bool) -> bool:
        """Process a batch of files"""
        logger.info(f"Processing batch of {len(file_paths)} files")

        try:
            # Process documents
            all_documents = []
            for file_path in file_paths:
                if file_path.is_file():
                    doc = self.document_processor.process_document(file_path)
                    all_documents.append(doc)

            if not all_documents:
                logger.warning("No documents processed in this batch")
                return True

            # Categorize documents
            categorized_docs = self.categorizer.categorize_documents(all_documents)

            # Save processed documents
            with open(PROCESSED_DIR / "processed_documents.json", "a") as f:
                for doc in categorized_docs:
                    f.write(json.dumps(doc) + "\n")

            # Chunk documents
            chunks = self.chunker.chunk_documents(categorized_docs)
            if not chunks:
                logger.warning("No chunks generated from documents in this batch")
                return True

            # Generate embeddings
            embeddings, enhanced_chunks = self.embedding_generator.generate_document_embeddings(chunks)
            if len(embeddings) == 0:
                logger.warning("No embeddings generated in this batch")
                return True

            # Add to vector store
            if is_first_batch:
                # Build new index with first batch
                self.vector_store.build_index(embeddings, enhanced_chunks)
            else:
                # Add to existing index
                self.vector_store.add_embeddings(embeddings, enhanced_chunks)

            # Clear memory
            del all_documents
            del categorized_docs
            del chunks
            del embeddings
            del enhanced_chunks
            MemoryMonitor.gc_collect()

            logger.info(f"Successfully processed batch of {len(file_paths)} files")

            return True

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self.progress['errors'].append(str(e))
            return False

    def _process_tasks(self):
        """Process tasks from the queue in a separate thread"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break

                task_type, task_data, callback = task
                result = None

                if task_type == "process_document":
                    result = self._process_document_task(task_data)
                elif task_type == "generate_embedding":
                    result = self._generate_embedding_task(task_data)

                if callback:
                    callback(result)

                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")

    def _process_document_task(self, file_path: str) -> Dict[str, Any]:
        """Process a document task"""
        try:
            return self.document_processor.process_document(Path(file_path))
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {"error": str(e), "success": False}

    def _generate_embedding_task(self, text: str) -> np.ndarray:
        """Generate embedding task"""
        try:
            return self.embedding_generator.generate_text_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(768, dtype=np.float32)

    def add_task(self, task_type: str, task_data: Any, callback: Callable = None):
        """Add a task to the processing queue"""
        self.task_queue.put((task_type, task_data, callback))

    def process_documents_parallel(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process documents in parallel"""
        logger.info(f"Processing {len(file_paths)} documents in parallel with {config.max_workers} workers")

        results = []
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.document_processor.process_document, path): path
                for path in file_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    results.append({
                        "path": str(path),
                        "filename": path.name,
                        "error": str(e),
                        "success": False
                    })

        return results

    def process_query_with_cache(self, query: str, user_id: int, use_cache: bool = True) -> Dict[str, Any]:
        """Process a query with caching support"""
        # Sanitize query
        sanitized_query = sanitize_input(query)

        # Check cache if enabled
        cache_key = hashlib.md5(f"{sanitized_query}_{user_id}".encode()).hexdigest()
        if use_cache:
            cached_result = get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Returning cached result for query: {sanitized_query}")
                return cached_result

        # Process the query
        result = self.process_query(sanitized_query)

        # Add to search history
        self.search_history.add_search(
            user_id,
            sanitized_query,
            result.get("response", "")
        )

        # Cache the result
        if use_cache:
            cache_result(cache_key, result)

        return result

    def add_feedback(self, user_id: int, query: str, response: str, rating: int, comment: str = "") -> bool:
        """Add feedback for a response"""
        try:
            return self.feedback_system.add_feedback(user_id, query, response, rating, comment)
        except ValidationError as e:
            logger.error(f"Validation error adding feedback: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False

    def upload_document(self, file_data, filename: str, user_id: int, comment: str = "") -> Dict[str, Any]:
        """Upload and process a document"""
        # Save uploaded file
        upload_path = UPLOADS_DIR / filename
        with open(upload_path, 'wb') as f:
            f.write(file_data.getbuffer())

        # Add to versioning system
        try:
            version_id = self.versioning_system.add_version(upload_path, user_id, comment)
        except Exception as e:
            logger.error(f"Error adding document version: {str(e)}")
            return {"success": False, "error": str(e)}

        # Process the document
        try:
            doc = self.document_processor.process_document(upload_path)
            # Categorize document
            categorized_doc = self.categorizer.categorize_document(doc)
            # Chunk document
            chunks = self.chunker.chunk_document(categorized_doc)
            # Generate embeddings
            embeddings, enhanced_chunks = self.embedding_generator.generate_document_embeddings(chunks)
            # Add to vector store
            if self.vector_store.is_built:
                self.vector_store.add_embeddings(embeddings, enhanced_chunks)
            else:
                self.vector_store.build_index(embeddings, enhanced_chunks)

            return {
                "success": True,
                "document": categorized_doc,
                "chunks_count": len(chunks),
                "version_id": version_id
            }
        except Exception as e:
            logger.error(f"Error processing uploaded document: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_user_search_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get search history for a user"""
        return self.search_history.get_history(user_id, limit)

    def clear_user_search_history(self, user_id: int) -> bool:
        """Clear search history for a user"""
        return self.search_history.clear_history(user_id)

    def get_user_feedback(self, user_id: int) -> List[Dict[str, Any]]:
        """Get feedback for a user"""
        return self.feedback_system.get_feedback(user_id)

    def get_document_versions(self, filename: str) -> List[Dict[str, Any]]:
        """Get versions of a document"""
        return self.versioning_system.get_versions(filename)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return self.progress

# ===== TRANSLATION FUNCTION =====
def translate_text(text: str, target_lang: str, translator) -> str:
    """Translate text to target language"""
    if not text or target_lang == "English":
        return text

    try:
        # Limit text length to avoid API limits
        if len(text) > 5000:
            text = text[:5000] + "..."

        if not translator:
            logger.debug("No translator available; returning original text")
            return text

        res = translator.translate(text, dest='ja' if target_lang == "Japanese" else 'en')

        # Handle coroutine/awaitable translators (some implementations may be async)
        if asyncio.iscoroutine(res):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Running inside an event loop (Streamlit); run synchronously
                res = asyncio.get_event_loop().run_until_complete(res)
            else:
                res = asyncio.run(res)

        # Some translators return object with .text, others return string
        if hasattr(res, 'text'):
            return res.text
        if isinstance(res, str):
            return res
        return text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

# ===== STREAMLIT APP =====
# Initialize the RAG system
rag_system = EnhancedRAGSystem()

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main container */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
    }
    /* Header */
    .header {
        padding: 1rem;
        background-color: #f8f9fa;
        border-bottom: 1px solid #e9ecef;
    }
    /* Chat container */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f7f7f8;
    }
    /* Message bubbles */
    .message {
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        max-width: 80%;
    }
    .user-message {
        align-self: flex-end;
    }
    .assistant-message {
        align-self: flex-start;
    }
    .message-bubble {
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin-bottom: 0.25rem;
        word-wrap: break-word;
    }
    .user-message .message-bubble {
        background-color: #007bff;
        color: white;
        border-bottom-right-radius: 4px;
    }
    .assistant-message .message-bubble {
        background-color: #e9e9eb;
        color: #000;
        border-bottom-left-radius: 4px;
    }
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.25rem;
        font-weight: bold;
        color: white;
    }
    .user-message .message-avatar {
        background-color: #007bff;
        align-self: flex-end;
    }
    .assistant-message .message-avatar {
        background-color: #6c757d;
        align-self: flex-start;
    }
    .message-time {
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    .user-message .message-time {
        text-align: right;
    }
    /* Input area */
    .input-container {
        padding: 1rem;
        background-color: white;
        border-top: 1px solid #e9ecef;
    }
    /* Sidebar */
    .sidebar {
        background-color: #f8f9fa;
        border-left: 1px solid #e9ecef;
        padding: 1rem;
        height: 100vh;
        overflow-y: auto;
    }
    .sidebar-section {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .source-document {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-radius: 4px;
    }
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background-color: #e9e9eb;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 80px;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6c757d;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
    /* Feedback buttons */
    .feedback-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .feedback-button {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        font-size: 0.8rem;
    }
    .feedback-positive {
        background-color: #28a745;
        color: white;
    }
    .feedback-negative {
        background-color: #dc3545;
        color: white;
    }
    /* Authentication form */
    .auth-form {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    /* Document upload */
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Error message */
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    /* Success message */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    /* Progress bar */
    .progress-container {
        width: 100%;
        background-color: #f1f1f1;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 20px;
        background-color: #4CAF50;
        border-radius: 4px;
        text-align: center;
        line-height: 20px;
        color: white;
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-container {
            flex-direction: column;
        }
        .sidebar {
            height: auto;
            border-left: none;
            border-top: 1px solid #e9ecef;
        }
    }
    /* Document preview and download buttons */
    .doc-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .doc-action-btn {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        font-size: 0.8rem;
        background-color: #f8f9fa;
        color: #212529;
        border: 1px solid #dee2e6;
    }
    .doc-action-btn:hover {
        background-color: #e9ecef;
    }
    /* Relevance score badge */
    .relevance-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #007bff;
        color: white;
    }
    /* Language toggle */
    .language-toggle {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .language-toggle label {
        margin-bottom: 0;
        font-weight: 500;
    }
    /* Document section at bottom of response */
    .documents-section {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .documents-section h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        color: #495057;
    }
    .document-item {
        margin-bottom: 0.75rem;
        padding: 0.5rem;
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .document-item-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
    }
    .document-item-title {
        font-weight: 500;
        margin: 0;
    }
    .document-item-meta {
        font-size: 0.75rem;
        color: #6c757d;
    }
    .document-item-content {
        font-size: 0.875rem;
        color: #495057;
        margin: 0.25rem 0;
    }
    /* Loading indicator */
    .loading-container {
        display: flex;
        justify-content: center;
        padding: 1rem;
    }
    .loading-spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-radius: 50%;
        border-top: 4px solid #007bff;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Nakakita Air Purge Assistant",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}
    if 'typing_indicator' not in st.session_state:
        st.session_state.typing_indicator = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'success_message' not in st.session_state:
        st.session_state.success_message = None
    if 'translator' not in st.session_state:
        if GOOGLETRANS_AVAILABLE and Translator is not None:
            try:
                st.session_state.translator = Translator()
            except Exception as e:
                logger.warning(f"Failed to initialize Translator: {e}")
                st.session_state.translator = None
        else:
            st.session_state.translator = None
    if 'language' not in st.session_state:
        st.session_state.language = "English"
    # Initialize TTS file state
    if 'tts_file' not in st.session_state:
        st.session_state.tts_file = None
    # Initialize transcribed text state
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = None

    # Check if user is logged in
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        show_auth_ui()
        return

    # Now safely get user info
    user_info = rag_system.auth_system.get_user_info(st.session_state.user_id)
    if not user_info:
        # Session expired
        st.session_state.user_id = None
        st.session_state.error_message = "Your session has expired. Please log in again."
        st.rerun()

    # Main application UI
    show_main_ui(user_info)

def show_auth_ui():
    """Show authentication UI"""
    st.markdown("""
    <div class="auth-form">
        <h1 style="text-align: center;">🔧 Nakakita Air Purge Assistant</h1>
        <p style="text-align: center; color: #6c757d;">Please log in to continue</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login-form"):
            st.text_input("Username", key="login_username")
            st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
            if submitted:
                username = st.session_state.login_username
                password = st.session_state.login_password
                if not username or not password:
                    st.error("Please enter both username and password")
                    return

                try:
                    session_token = rag_system.auth_system.authenticate_user(username, password)
                    if session_token:
                        # Validate session and get user_id
                        user_id = rag_system.auth_system.validate_session(session_token)
                        if user_id is not None:
                            st.session_state.user_id = user_id  # Set user_id explicitly
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Session validation failed.")
                    else:
                        st.error("Invalid username or password")
                except AuthenticationError as e:
                    st.error(str(e))
                except ValidationError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

    with tab2:
        with st.form("register-form"):
            st.text_input("Username", key="register_username")
            st.text_input("Password", type="password", key="register_password")
            st.text_input("Confirm Password", type="password", key="register_confirm_password")
            st.text_input("Email (optional)", key="register_email")
            submitted = st.form_submit_button("Register")
            if submitted:
                username = st.session_state.register_username
                password = st.session_state.register_password
                confirm_password = st.session_state.register_confirm_password
                email = st.session_state.register_email
                if not username or not password:
                    st.error("Please enter both username and password")
                    return
                if password != confirm_password:
                    st.error("Passwords do not match")
                    return
                try:
                    if rag_system.auth_system.register_user(username, password, email):
                        st.success("Registration successful! Please log in.")
                    else:
                        st.error("Username already exists")
                except ValidationError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

def show_main_ui(user_info):
    """Show main application UI"""
    # Create two columns
    col1, col2 = st.columns([3, 1])

    with col1:
        # Header with language toggle
        st.markdown(f"""
        <div class="header">
            <h1>🔧 NS Air Purge GPT</h1>
            <p>Welcome, {user_info['username']} | <a href="#" onclick="logout()">Logout</a></p>
        </div>
        """, unsafe_allow_html=True)

        # Language toggle - improved for one-click switching
        st.markdown("""
        <div class="language-toggle">
            <label for="language_selector">Response Language:</label>
        </div>
        """, unsafe_allow_html=True)

        col_lang1, col_lang2 = st.columns([1, 1])
        with col_lang1:
            if st.button("English", key="lang_en", type="primary" if st.session_state.language == "English" else "secondary"):
                st.session_state.language = "English"
                st.rerun()
        with col_lang2:
            if st.button("日本語 (Japanese)", key="lang_ja", type="primary" if st.session_state.language == "Japanese" else "secondary"):
                st.session_state.language = "Japanese"
                st.rerun()

        # Show error/success messages
        if st.session_state.error_message:
            st.markdown(f"""
            <div class="error-message">
                {st.session_state.error_message}
            </div>
            """, unsafe_allow_html=True)
            st.session_state.error_message = None

        if st.session_state.success_message:
            st.markdown(f"""
            <div class="success-message">
                {st.session_state.success_message}
            </div>
            """, unsafe_allow_html=True)
            st.session_state.success_message = None

        # Navigation
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

        # Define navigation labels based on language
        if st.session_state.language == "Japanese":
            nav_labels = ["チャット", "ドキュメントアップロード", "履歴", "⚙️ システム"]
        else:
            nav_labels = ["Chat", "Upload Documents", "History", "⚙️ System"]

        with nav_col1:
            if st.button(nav_labels[0], key="nav_chat", type="primary" if st.session_state.current_page == "chat" else "secondary"):
                st.session_state.current_page = "chat"
                st.rerun()
        with nav_col2:
            if st.button(nav_labels[1], key="nav_upload", type="primary" if st.session_state.current_page == "upload" else "secondary"):
                st.session_state.current_page = "upload"
                st.rerun()
        with nav_col3:
            if st.button(nav_labels[2], key="nav_history", type="primary" if st.session_state.current_page == "history" else "secondary"):
                st.session_state.current_page = "history"
                st.rerun()
        with nav_col4:
            if st.button(nav_labels[3], key="nav_system", type="primary" if st.session_state.current_page == "system" else "secondary"):
                st.session_state.current_page = "system"
                st.rerun()

        # Page content
        if st.session_state.current_page == "chat":
            show_chat_page()
        elif st.session_state.current_page == "upload":
            show_upload_page()
        elif st.session_state.current_page == "history":
            show_history_page()
        elif st.session_state.current_page == "system":
            show_system_page()

    with col2:
        # Sidebar
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)

        # System Status
        st.markdown("### 📊 System Status")
        if rag_system.initialized:
            st.success("✅ Ready")
            st.caption(f"{len(rag_system.vector_store.metadata)} chunks indexed")
        else:
            st.warning("⚠️ Not Ready")
            st.caption("Go to ⚙️ System to process documents.")

        # Quick Access Button
        if st.button("⚙️ System Settings", key="sidebar_system_btn"):
            st.session_state.current_page = "system"
            st.rerun()

        # Show relevant documents for last query
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_message = st.session_state.messages[-1]
            if "relevant_chunks" in last_message and last_message["relevant_chunks"]:
                st.markdown("### 📑 Sources")
                for s_idx, chunk in enumerate(last_message["relevant_chunks"][:2]):  # Show only top 2
                    with st.expander(f"{chunk['filename'][:30]}... ({chunk['category']})"):
                        st.caption(f"Type: {chunk.get('content_type', 'text')}")
                        # Show relevance score
                        if "relevance_score" in chunk:
                            st.markdown(f"<span class='relevance-badge'>Relevance: {chunk['relevance_score']:.2f}</span>", unsafe_allow_html=True)
                        st.write(chunk['text'][:200] + "...")
                        # Add document action buttons
                        doc_path = chunk.get("doc_id", "")
                        if doc_path and os.path.exists(doc_path):
                            st.markdown('<div class="doc-actions">', unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                # Quick look button
                                quick_key = f"sidebar_quick_{len(st.session_state.messages)-1}_{s_idx}_{chunk.get('chunk_index','0')}_{abs(hash(chunk['filename'] + str(chunk['text'][:50])))}"
                                if st.button("👁️ Quick Look", key=quick_key):
                                    with open(doc_path, 'rb') as f:
                                        st.download_button(
                                            label="📥 Download",
                                            data=f,
                                            file_name=chunk['filename'],
                                            mime="application/octet-stream",
                                            key=f"sidebar_download_{len(st.session_state.messages)-1}_{s_idx}_{chunk.get('chunk_index','0')}_{abs(hash(chunk['filename'] + str(chunk['text'][:50])))}"
                                        )
                            with col2:
                                # Download button
                                dl_key = f"sidebar_dl_{len(st.session_state.messages)-1}_{s_idx}_{chunk.get('chunk_index','0')}_{abs(hash(chunk['filename'] + str(chunk['text'][:50])))}"
                                if st.button("📥 Download", key=dl_key):
                                    with open(doc_path, 'rb') as f:
                                        st.download_button(
                                            label="Download File",
                                            data=f,
                                            file_name=chunk['filename'],
                                            mime="application/octet-stream"
                                        )
                            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

def show_system_page():
    """Show system administration and processing page"""
    st.header("⚙️ System Administration")

    # Process Documents Button
    # Translate process documents button
    if st.session_state.language == "Japanese":
        process_docs_text = "ドキュメントを処理"
    else:
        process_docs_text = "Process Documents"

    if st.button(process_docs_text, type="primary", key="process_docs_btn_system"):
        with st.spinner("Processing documents... This may take a while."):
            success = rag_system.process_documents(force_reprocess=True)
            if success:
                st.success("✅ All documents processed successfully!")
                # Optionally clear chat to avoid confusion with old context
                # st.session_state.messages = []
            else:
                st.error("❌ Failed to process documents.")

    st.subheader("📊 System Statistics")

    # Collapsible sections for stats — only load when expanded
    # Document Processing Stats
    with st.expander("📄 Document Processing Stats", expanded=False):
        doc_stats = rag_system.document_processor.get_processing_stats()
        if doc_stats.get('total_files', 0) > 0:
            st.plotly_chart(rag_system.document_processor.visualize_processing_stats(), use_container_width=True)
        else:
            st.info("No document processing stats available. Process documents first.")

    # Categorization Stats
    with st.expander("🏷️ Document Categorization", expanded=False):
        cat_stats = rag_system.categorizer.get_categorization_stats()
        if cat_stats.get('total', 0) > 0:
            st.plotly_chart(rag_system.categorizer.visualize_categorization_stats(), use_container_width=True)
        else:
            st.info("No categorization stats available.")

    # Chunking Stats
    with st.expander("✂️ Text Chunking Stats", expanded=False):
        chunk_stats = rag_system.chunker.get_chunking_stats()
        if chunk_stats.get('total_documents', 0) > 0:
            st.plotly_chart(rag_system.chunker.visualize_chunking_stats(), use_container_width=True)
        else:
            st.info("No chunking stats available.")

    # Embedding & Vector Store Stats
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🧠 Embedding Stats", expanded=False):
            embed_stats = rag_system.embedding_generator.get_embedding_stats()
            st.plotly_chart(rag_system.embedding_generator.visualize_embedding_stats(), use_container_width=True)

    with col2:
        with st.expander("🗄️ Vector Store Stats", expanded=False):
            vs_stats = rag_system.vector_store.get_search_stats()
            st.plotly_chart(rag_system.vector_store.visualize_search_stats(), use_container_width=True)

    # Show current progress if any
    progress = rag_system.get_progress()
    if progress['status'] != 'Idle':
        st.info(f"Status: {progress['status']}")
        st.progress(progress['current'] / max(1, progress['total']))
        if progress['errors']:
            with st.expander("⚠️ Errors"):
                for err in progress['errors']:
                    st.error(err)

def show_chat_page():
    """Show chat page"""
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            # Translate user message if needed
            display_text = message["content"]
            if st.session_state.language == "Japanese":
                display_text = translate_text(message["content"], "Japanese", st.session_state.translator)
            st.markdown(f"""
            <div class="message user-message">
                <div class="message-avatar">U</div>
                <div class="message-bubble">{display_text}</div>
                <div class="message-time">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Translate assistant message if needed
            if st.session_state.language == "Japanese" and "content_ja" in message and message["content_ja"]:
                display_text = message["content_ja"]
            else:
                display_text = message["content"]

            st.markdown(f"""
            <div class="message assistant-message" data-message-id="{i}">
                <div class="message-avatar">AI</div>
                <div class="message-bubble">{display_text}</div>
                <div class="feedback-buttons">
                    <button class="feedback-button feedback-positive" onclick="sendFeedback('{i}', 5)">👍 Helpful</button>
                    <button class="feedback-button feedback-negative" onclick="sendFeedback('{i}', 1)">👎 Not Helpful</button>
                </div>
                <div class="message-time">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Store feedback in session state
            feedback_id = f"feedback_{i}"
            if feedback_id not in st.session_state:
                st.session_state[feedback_id] = None

            # Show relevant documents at the bottom of the response
            if "relevant_chunks" in message and message["relevant_chunks"]:
                st.markdown('<div class="documents-section">', unsafe_allow_html=True)
                if st.session_state.language == "Japanese":
                    st.markdown("<h4>関連ドキュメント</h4>", unsafe_allow_html=True)
                else:
                    st.markdown("<h4>Relevant Documents</h4>", unsafe_allow_html=True)

                for chunk_idx, chunk in enumerate(message["relevant_chunks"][:3]):  # Show top 3 documents
                    st.markdown('<div class="document-item">', unsafe_allow_html=True)

                    # Document header with filename and relevance score
                    st.markdown('<div class="document-item-header">', unsafe_allow_html=True)
                    st.markdown(f"<h5 class='document-item-title'>{chunk['filename']}</h5>", unsafe_allow_html=True)
                    if "relevance_score" in chunk:
                        st.markdown(f"<span class='relevance-badge'>Relevance: {chunk['relevance_score']:.2f}</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Document metadata
                    st.markdown(f"<div class='document-item-meta'>{chunk['category']} | {chunk.get('content_type', 'text')}</div>", unsafe_allow_html=True)

                    # Document content preview
                    st.markdown(f"<div class='document-item-content'>{chunk['text'][:200]}...</div>", unsafe_allow_html=True)

                    # Document actions
                    st.markdown('<div class="doc-actions">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        # Quick look button
                        quicklook_key = f"msg_quicklook_{i}_{chunk_idx}_{chunk.get('chunk_index','0')}_{chunk['doc_id']}_{abs(hash(chunk['filename'] + str(chunk['text'][:50]))) }"
                        if st.button("👁️ Quick Look", key=quicklook_key):
                            doc_path = chunk.get("doc_id", "")
                            if doc_path and os.path.exists(doc_path):
                                try:
                                    with open(doc_path, 'rb') as f:
                                        st.download_button(
                                            label="📥 Download",
                                            data=f,
                                            file_name=chunk['filename'],
                                            mime="application/octet-stream",
                                            key=f"msg_download_{i}_{chunk_idx}_{chunk.get('chunk_index','0')}_{chunk['doc_id']}_{abs(hash(chunk['filename'] + str(chunk['text'][:50])))}"
                                        )
                                except Exception as e:
                                    st.error(f"Could not load document: {str(e)}")
                    with col2:
                        # Download button
                        if st.button("📥 Download", key=f"msg_dl_{i}_{chunk_idx}_{chunk.get('chunk_index','0')}_{chunk['doc_id']}_{abs(hash(chunk['filename'] + str(chunk['text'][:50])))}"):
                            doc_path = chunk.get("doc_id", "")
                            if doc_path and os.path.exists(doc_path):
                                try:
                                    with open(doc_path, 'rb') as f:
                                        st.download_button(
                                            label="Download File",
                                            data=f,
                                            file_name=chunk['filename'],
                                            mime="application/octet-stream"
                                        )
                                except Exception as e:
                                    st.error(f"Could not load document: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Play TTS audio if this is the latest message
            if i == len(st.session_state.messages) - 1 and st.session_state.tts_file:
                st.audio(st.session_state.tts_file, format="audio/mp3", autoplay=True)
                # Optionally clear the TTS file after playing
                # st.session_state.tts_file = None

    # Show typing indicator if needed
    if st.session_state.typing_indicator:
        st.markdown("""
        <div class="message assistant-message">
            <div class="message-avatar">AI</div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Translate clear conversation button
    if st.session_state.language == "Japanese":
        clear_button_text = "会話をクリア"
    else:
        clear_button_text = "Clear Conversation"

    if st.button(clear_button_text, key="clear_conversation_btn"):
        st.session_state.messages = []
        st.session_state.typing_indicator = False
        st.session_state.tts_file = None
        st.success("Conversation cleared.")
        st.rerun()

    # Input form
    with st.form("chat-form", clear_on_submit=True):
        # Create two columns for text input and mic button
        col_input, col_mic = st.columns([4, 1])

        with col_input:
            # Translate input placeholder
            if st.session_state.language == "Japanese":
                input_placeholder = "メッセージを入力してください..."
            else:
                input_placeholder = "Type your message here..."
            user_input = st.text_area(input_placeholder, height=80, key="user_input")

        with col_mic:
            # Speech-to-Text Input - Fixed: using st.audio_input instead of deprecated st.experimental_audio_input
            audio_bytes = st.audio_input("🎤", key="audio_input")

        # Translate send button
        if st.session_state.language == "Japanese":
            send_button_text = "送信"
        else:
            send_button_text = "Send"

        submitted = st.form_submit_button(send_button_text)

        if submitted:
            final_input = user_input

            # If audio is recorded, transcribe it
            if audio_bytes:
                if not FASTER_WHISPER_AVAILABLE:
                    st.error("Speech-to-text is not available. Please install 'faster-whisper'.")
                else:
                    with st.spinner("Transcribing audio..."):
                        # Save audio bytes to file
                        audio_file = "temp_audio.wav"
                        with open(audio_file, "wb") as f:
                            f.write(audio_bytes.getvalue())

                        # Transcribe using faster-whisper
                        try:
                            model = WhisperModel("base", device="cpu", compute_type="int8")
                            segments, info = model.transcribe(audio_file, beam_size=5)
                            transcribed_text = " ".join([segment.text for segment in segments])
                            final_input = transcribed_text
                            st.session_state.transcribed_text = transcribed_text  # Store for potential display
                        except Exception as e:
                            logger.error(f"Transcription Error: {e}")
                            st.error(f"Failed to transcribe audio: {str(e)}")
                            final_input = user_input  # Fallback to text input

            if final_input.strip():
                # Add user message
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.messages.append({
                    "role": "user",
                    "content": final_input,
                    "timestamp": timestamp
                })

                # Show typing indicator
                st.session_state.typing_indicator = True
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Process typing indicator
    if st.session_state.typing_indicator:
       # Get the last user message
       if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
           last_user_message = st.session_state.messages[-1]["content"]
           # Get conversation history
           conversation_history = [
            {"query": msg["content"], "response": st.session_state.messages[i+1]["content"]}
            for i, msg in enumerate(st.session_state.messages[:-1])
            if msg["role"] == "user" and i+1 < len(st.session_state.messages)
           ]

           # Process the query
           result = rag_system.process_query_with_cache(
               query=last_user_message,
               user_id=st.session_state.user_id
          )

           # Translate the response based on current language selection
           response_en = result["response"]
           response_display = response_en
           if st.session_state.language == "Japanese":
               response_display = translate_text(response_en, "Japanese", st.session_state.translator)

           # Add assistant response to chat history (store both English and translated version)
           response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

           # Generate TTS audio
           if GTTS_AVAILABLE:
               try:
                   tts_lang = 'ja' if st.session_state.language == "Japanese" else 'en'
                   tts = gTTS(text=response_display, lang=tts_lang)
                   with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                       tts.save(fp.name)
                       st.session_state.tts_file = fp.name
               except Exception as e:
                   logger.error(f"TTS Generation Error: {e}")
                   st.session_state.tts_file = None
           else:
               st.session_state.tts_file = None

           st.session_state.messages.append({
               "role": "assistant",
               "content": response_en,  # Store English version for consistency
               "content_ja": translate_text(response_en, "Japanese", st.session_state.translator) if st.session_state.language == "Japanese" else None,  # Store Japanese version if needed
               "timestamp": response_timestamp,
               "relevant_chunks": result.get("relevant_chunks", [])
           })

           # Hide typing indicator
           st.session_state.typing_indicator = False

           # Rerun to update the chat and play audio
           st.rerun()

def show_upload_page():
    """Show document upload page"""
    # Translate header
    if st.session_state.language == "Japanese":
        st.header("ドキュメントをアップロード")
        upload_button_text = "アップロードしたファイルを処理"
    else:
        st.header("Upload Documents")
        upload_button_text = "Process Uploaded Files"

    # Document upload
    st.markdown("""
    <div class="upload-container">
        <p>Drag and drop files here or click to browse</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload documents to add to the knowledge base"
    )

    # Comment for versioning
    comment = st.text_area("Comment (optional)", help="Add a comment to describe this version of the document")

    if uploaded_files and st.button(upload_button_text, type="primary"):
        with st.spinner("Processing uploaded documents..."):
            success_count = 0
            error_count = 0
            for uploaded_file in uploaded_files:
                try:
                    result = rag_system.upload_document(
                        uploaded_file,
                        uploaded_file.name,
                        st.session_state.user_id,
                        comment
                    )
                    if result["success"]:
                        success_count += 1
                    else:
                        error_count += 1
                        st.error(f"Error processing {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    error_count += 1
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            if success_count > 0:
                if st.session_state.language == "Japanese":
                    success_msg = f"{success_count} ドキュメントが正常に処理されました。"
                else:
                    success_msg = f"Successfully processed {success_count} document(s)."
                st.session_state.success_message = success_msg

            if error_count > 0:
                if st.session_state.language == "Japanese":
                    error_msg = f"{error_count} ドキュメントの処理に失敗しました。"
                else:
                    error_msg = f"Failed to process {error_count} document(s)."
                st.session_state.error_message = error_msg

            st.rerun()

    # Document versions
    if st.session_state.language == "Japanese":
        st.subheader("ドキュメントバージョン")
    else:
        st.subheader("Document Versions")

    # Get list of documents
    documents = []
    for filename in os.listdir(UPLOADS_DIR):
        if os.path.isfile(os.path.join(UPLOADS_DIR, filename)):
            documents.append(filename)

    if documents:
        selected_doc = st.selectbox("Select a document to view versions", documents)
        if selected_doc:
            versions = rag_system.get_document_versions(selected_doc)
            if versions:
                if st.session_state.language == "Japanese":
                    st.write(f"{selected_doc} の {len(versions)} バージョンが見つかりました")
                else:
                    st.write(f"Found {len(versions)} version(s) of {selected_doc}")
                for version in versions:
                    with st.expander(f"Version {version['version_id'][:8]}... - {version['created_at']}"):
                        st.write(f"Uploaded by: {version['username']}")
                        st.write(f"Comment: {version['comment']}")
                        st.write(f"File hash: {version['file_hash'][:16]}...")
            else:
                if st.session_state.language == "Japanese":
                    st.write("このドキュメントのバージョンは見つかりませんでした。")
                else:
                    st.write("No versions found for this document.")
    else:
        if st.session_state.language == "Japanese":
            st.write("まだドキュメントがアップロードされていません。")
        else:
            st.write("No documents uploaded yet.")

def show_history_page():
    """Show search history page"""
    # Translate header
    if st.session_state.language == "Japanese":
        st.header("検索履歴")
        clear_button_text = "検索履歴をクリア"
    else:
        st.header("Search History")
        clear_button_text = "Clear Search History"

    # Get search history
    history = rag_system.get_user_search_history(st.session_state.user_id)

    if history:
        # Display history in reverse chronological order
        for i, item in enumerate(reversed(history)):
            with st.expander(f"Query: {item['query']} - {item['created_at']}"):
                st.markdown(f"**Response:** {item['response']}")
                # Show relevant chunks
                if item.get("relevant_chunks"):
                    if st.session_state.language == "Japanese":
                        st.markdown("**関連ドキュメント:**")
                    else:
                        st.markdown("**Relevant Documents:**")
                    for hist_chunk_idx, chunk in enumerate(item["relevant_chunks"][:3]):
                        content_type = chunk.get("content_type", "text")
                        content_label = "Table" if content_type == "table" else "Text"
                        # Get document path
                        doc_path = chunk.get("doc_id") or chunk.get("metadata", {}).get("source", "")
                        st.markdown(f"""
                        <div class="source-document">
                            <strong>{chunk['filename']}</strong>
                            <span style="font-size: 0.8em; color: #6c757d;">({chunk['category']}, {content_label})</span>
                            <br>
                            <small>{chunk['text'][:150]}...</small>
                        """, unsafe_allow_html=True)
                        # Add download button if document exists
                        if doc_path and os.path.exists(doc_path):
                            try:
                                with open(doc_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Download",
                                        data=file,
                                        file_name=chunk['filename'],
                                        mime="application/octet-stream",
                                        key=f"hist_download_{item['id']}_{hist_chunk_idx}_{chunk.get('chunk_index','0')}_{chunk['doc_id']}_{abs(hash(chunk['filename'] + str(chunk['text'][:50])))}"
                                    )
                            except Exception as e:
                                st.error(f"Could not load document: {str(e)}")
                        st.markdown("</div>", unsafe_allow_html=True)

        # Clear history button
        if st.button(clear_button_text, type="secondary"):
            if rag_system.clear_user_search_history(st.session_state.user_id):
                if st.session_state.language == "Japanese":
                    st.session_state.success_message = "検索履歴がクリアされました。"
                else:
                    st.session_state.success_message = "Search history cleared."
                st.rerun()
            else:
                if st.session_state.language == "Japanese":
                    st.session_state.error_message = "検索履歴のクリアに失敗しました。"
                else:
                    st.session_state.error_message = "Failed to clear search history."
                st.rerun()
    else:
        if st.session_state.language == "Japanese":
            st.write("検索履歴が見つかりませんでした。")
        else:
            st.write("No search history found.")

# JavaScript for interactivity
st.markdown("""
<script>
function logout() {
    // Clear session token and reload
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: {key: 'logout_clicked', value: true}}, '*');
}
function sendFeedback(messageId, rating) {
    // Send feedback to the server
    const messageElement = document.querySelector(`[data-message-id="${messageId}"] .message-bubble`);
    const message = messageElement ? messageElement.textContent : '';
    window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: {
            key: `feedback_${messageId}`,
            value: {
                message: message,
                rating: rating
            }
        }
    }, '*');
    // Update UI to show feedback was received
    const feedbackButtons = document.querySelector(`[data-message-id="${messageId}"] .feedback-buttons`);
    feedbackButtons.innerHTML = rating >= 4 ?
        '<span style="color: #28a745;">✓ Thank you for your feedback!</span>' :
        '<span style="color: #dc3545;">✓ Thank you for your feedback!</span>';
}
</script>
""", unsafe_allow_html=True)

# Handle logout
if 'logout_clicked' in st.session_state and st.session_state.logout_clicked:
    rag_system.auth_system.logout(st.session_state.user_id)
    st.session_state.user_id = None
    st.session_state.logout_clicked = False
    st.rerun()

# Handle feedback
for key in st.session_state:
    if key.startswith('feedback_') and st.session_state[key] is not None:
        feedback_data = st.session_state[key]
        if isinstance(feedback_data, dict):
            rag_system.add_feedback(
                st.session_state.user_id,
                feedback_data.get('message', ''),
                feedback_data.get('response', ''),
                feedback_data.get('rating', 0),
                ""
            )
        st.session_state[key] = None

if __name__ == "__main__":
    main()



