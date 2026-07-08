import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load standard .env if present
load_dotenv()

class IngestionSettings(BaseSettings):
    """
    Configuration settings loaded from environment variables or .env file.
    Utilizes Pydantic Settings validation and types.
    """
    # Salesforce Credentials
    SALESFORCE_USERNAME: Optional[str] = None
    SALESFORCE_PASSWORD: Optional[str] = None
    SALESFORCE_SECURITY_TOKEN: Optional[str] = None

    # Zendesk Credentials
    ZENDESK_EMAIL: Optional[str] = None
    ZENDESK_API_TOKEN: Optional[str] = None
    ZENDESK_SUBDOMAIN: Optional[str] = None

    # Freshdesk Credentials
    FRESHDESK_API_KEY: Optional[str] = None
    FRESHDESK_DOMAIN: Optional[str] = None

    # Google Gmail API Credentials
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REFRESH_TOKEN: Optional[str] = None

    # Ingestion logs configurations
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = IngestionSettings()

# Logging Configuration
log_level_numeric = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level_numeric,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingestion")
logger.setLevel(log_level_numeric)
