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
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL', 
        'mysql+pymysql://root:@localhost/grocera_db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set to True for SQL debugging
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,  # Verify connections before using
    }
    
    # ============= AI Configuration =============
    GROK_API_KEY = os.getenv('GROK_API_KEY', '')
    GROK_API_BASE_URL = os.getenv('GROK_API_BASE_URL', 'https://api.x.ai/v1')
    GROK_MODEL = os.getenv('GROK_MODEL', 'grok-beta')
    AI_TIMEOUT = int(os.getenv('AI_TIMEOUT', '30'))  # seconds
    
    # ============= File Upload Configuration =============
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'processed')
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
    LOG_FILE = os.getenv('LOG_FILE', 'logs/grocera.log')
    
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
            'logs',
            'models'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
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
            'has_grok_api_key': bool(cls.GROK_API_KEY),
        }

