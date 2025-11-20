import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Text Evaluator"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite:///./ai_text_evaluator.db"
    
    # HuggingFace
    HUGGINGFACE_HUB_TOKEN: Optional[str] = None
    
    # Model Settings
    TOXICITY_MODEL: str = "facebook/bart-large-mnli"
    PII_MODEL: str = "dslim/bert-base-NER"
    BIAS_MODEL: str = "d4data/bias-detection-model"
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Global settings instance
settings = Settings()
