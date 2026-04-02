"""
Configuration Management for Grocera Application

This module handles all application configuration including:
- Flask settings
- Database connections
- API keys
- File upload settings
- Business logic thresholds
- Security settings

Environment variables are loaded from .env file.
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """
    Application configuration class.
    
    All settings can be overridden via environment variables.
    Sensitive data (API keys, database URLs) should be stored in .env file.
    """
    
    # ============= Flask Configuration =============
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-prod')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    
    # Session Configuration
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'False') == 'True'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # ============= Database Configuration =============
    _is_vercel = os.getenv('VERCEL') == '1'

    _database_url = os.getenv('DATABASE_URL', '').strip()
    if _database_url.startswith('postgres://'):
        _database_url = _database_url.replace('postgres://', 'postgresql://', 1)
    if _database_url.startswith('mysql://'):
        _database_url = _database_url.replace('mysql://', 'mysql+pymysql://', 1)
    if _database_url.startswith('mariadb://'):
        _database_url = _database_url.replace('mariadb://', 'mysql+pymysql://', 1)

    SQLALCHEMY_DATABASE_URI = (
        _database_url
        if _database_url
        else ('sqlite:////tmp/grocera.db' if _is_vercel else 'mysql+pymysql://root:@localhost/grocera_db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set to True for SQL debugging
    if SQLALCHEMY_DATABASE_URI.startswith('sqlite:'):
        SQLALCHEMY_ENGINE_OPTIONS = {}
    elif SQLALCHEMY_DATABASE_URI.startswith('mysql+'):
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '5')),
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '280')),
            'pool_pre_ping': True,
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'connect_args': {
                'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10')),
                'read_timeout': int(os.getenv('DB_READ_TIMEOUT', '30')),
                'write_timeout': int(os.getenv('DB_WRITE_TIMEOUT', '30')),
            }
        }
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '5')),
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '280')),
            'pool_pre_ping': True,
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
        }
    
    # ============= AI Configuration =============
    # Supports both Grok (xAI) and Groq.
    # Priority:
    # 1) Explicit provider via AI_PROVIDER
    # 2) GROQ_API_KEY
    # 3) GROK_API_KEY
    AI_PROVIDER = os.getenv('AI_PROVIDER', '').strip().lower()
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '').strip()
    GROK_API_KEY = os.getenv('GROK_API_KEY', '').strip()

    if not AI_PROVIDER:
        # Auto-detect based on which key is present.
        # If a Groq-style key is stored in GROK_API_KEY, treat provider as groq.
        if GROQ_API_KEY:
            AI_PROVIDER = 'groq'
        elif GROK_API_KEY.startswith('gsk_'):
            AI_PROVIDER = 'groq'
        else:
            AI_PROVIDER = 'grok'

    AI_API_KEY = GROQ_API_KEY if (AI_PROVIDER == 'groq' and GROQ_API_KEY) else GROK_API_KEY
    AI_API_BASE_URL = os.getenv(
        'AI_API_BASE_URL',
        'https://api.groq.com/openai/v1' if AI_PROVIDER == 'groq' else 'https://api.x.ai/v1'
    )
    AI_MODEL = os.getenv(
        'AI_MODEL',
        'llama-3.1-8b-instant' if AI_PROVIDER == 'groq' else 'grok-beta'
    )

    # Backward compatibility aliases
    GROK_API_BASE_URL = AI_API_BASE_URL
    GROK_MODEL = AI_MODEL
    AI_TIMEOUT = int(os.getenv('AI_TIMEOUT', '30'))  # seconds
    
    # ============= File Upload Configuration =============
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/uploads' if _is_vercel else 'uploads')
    PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', '/tmp/processed' if _is_vercel else 'processed')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', str(16 * 1024 * 1024)))  # 16MB default
    
    # ============= Business Logic Thresholds =============
    STOCK_ALERT_THRESHOLD = int(os.getenv('STOCK_ALERT_THRESHOLD', '5'))
    EXPIRY_ALERT_DAYS = int(os.getenv('EXPIRY_ALERT_DAYS', '7'))
    REORDER_THRESHOLD = int(os.getenv('REORDER_THRESHOLD', '10'))
    
    # ============= Pagination & Display =============
    ITEMS_PER_PAGE = int(os.getenv('ITEMS_PER_PAGE', '20'))
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '100'))
    
    # ============= Security Settings =============
    PASSWORD_MIN_LENGTH = 6
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 300  # 5 minutes in seconds
    
    # ============= Logging Configuration =============
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/tmp/grocera.log' if _is_vercel else 'logs/grocera.log')
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate critical configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        required_settings = {
            'SECRET_KEY': cls.SECRET_KEY,
            'SQLALCHEMY_DATABASE_URI': cls.SQLALCHEMY_DATABASE_URI,
        }
        
        for setting, value in required_settings.items():
            if not value or value == 'dev-secret-key-change-in-prod':
                print(f"Warning: {setting} is not properly configured!")
                if cls.FLASK_ENV == 'production':
                    return False
        
        return True
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create required directories if they don't exist."""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.PROCESSED_FOLDER,
        ]

        # In non-serverless environments we keep local app directories.
        if not cls._is_vercel:
            directories.extend(['logs', 'models'])

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                # Vercel serverless filesystem is read-only except /tmp.
                if cls._is_vercel and 'Read-only file system' in str(exc):
                    continue
                raise
    
    @classmethod
    def get_database_name(cls) -> Optional[str]:
        """Extract database name from DATABASE_URI."""
        try:
            return cls.SQLALCHEMY_DATABASE_URI.split('/')[-1].split('?')[0]
        except:
            return None
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return cls.FLASK_ENV == 'production'
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get summary of current configuration (excluding sensitive data)."""
        return {
            'environment': cls.FLASK_ENV,
            'debug': cls.DEBUG,
            'database': cls.get_database_name(),
            'upload_folder': cls.UPLOAD_FOLDER,
            'items_per_page': cls.ITEMS_PER_PAGE,
            'stock_alert_threshold': cls.STOCK_ALERT_THRESHOLD,
            'expiry_alert_days': cls.EXPIRY_ALERT_DAYS,
            'ai_provider': cls.AI_PROVIDER,
            'has_ai_api_key': bool(cls.AI_API_KEY),
        }

