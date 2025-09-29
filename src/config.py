"""
Configuration module for CLM automation system.
Loads environment variables and provides centralized configuration.
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables with override=True to ensure fresh reload
load_dotenv(override=True)

class Config:
    """Configuration class for CLM system"""
    
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Email Configuration
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_USERNAME: str = os.getenv("EMAIL_USERNAME", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    REPORT_EMAIL: str = os.getenv("REPORT_EMAIL", "")
    
    # Application Configuration
    DOCUMENTS_FOLDER: str = os.getenv("DOCUMENTS_FOLDER", "./documents")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536
    
    # Contract Analysis Configuration
    EXPIRATION_WARNING_DAYS: int = 30
    
    @classmethod
    def validate_config(cls, raise_on_missing: bool = False) -> tuple:
        """Validate that all required configuration is present
        
        Args:
            raise_on_missing: Whether to raise exception on missing fields
            
        Returns:
            Tuple of (is_valid, missing_fields)
        """
        required_fields = {
            "SUPABASE_URL": cls.SUPABASE_URL,
            "SUPABASE_KEY": cls.SUPABASE_KEY, 
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
        }
        
        missing_fields = [name for name, value in required_fields.items() if not value]
        
        if missing_fields and raise_on_missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")
        
        return len(missing_fields) == 0, missing_fields
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if minimum configuration is available"""
        is_valid, _ = cls.validate_config(raise_on_missing=False)
        return is_valid
    
    @classmethod
    def reload_config(cls) -> bool:
        """Reload configuration from .env file
        
        Returns:
            bool: True if configuration is valid after reload
        """
        # Reload environment variables
        load_dotenv(override=True)
        
        # Update class variables with new values
        cls.SUPABASE_URL = os.getenv("SUPABASE_URL", "")
        cls.SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        cls.SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        cls.SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
        cls.EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
        cls.EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
        cls.REPORT_EMAIL = os.getenv("REPORT_EMAIL", "")
        cls.DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "./documents")
        cls.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        cls.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        cls.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.9"))
        
        # Return validation status
        return cls.is_configured()

# Don't validate on import - let applications handle it
