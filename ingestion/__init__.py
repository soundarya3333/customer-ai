"""
Customer Support Ingestion Package.
Provides production-ready data connectors and schema normalization for:
- Salesforce Cases
- Zendesk Tickets
- Freshdesk Tickets
- Gmail Support Emails
- Website Forms (training datasets)
"""

from ingestion.config import settings, logger
from ingestion.exceptions import (
    IngestionError,
    AuthenticationError,
    RateLimitError,
    IngestionTimeoutError,
    APIFailureError,
    MissingFieldError,
    ValidationError
)
from ingestion.normalizer import Normalizer, UnifiedTicket
from ingestion.base_connector import BaseConnector
from ingestion.connector_factory import ConnectorFactory

from ingestion.salesforce_connector import SalesforceConnector
from ingestion.zendesk_connector import ZendeskConnector
from ingestion.freshdesk_connector import FreshdeskConnector
from ingestion.gmail_connector import GmailConnector
from ingestion.website_connector import WebsiteConnector

__all__ = [
    "settings",
    "logger",
    "IngestionError",
    "AuthenticationError",
    "RateLimitError",
    "IngestionTimeoutError",
    "APIFailureError",
    "MissingFieldError",
    "ValidationError",
    "Normalizer",
    "UnifiedTicket",
    "BaseConnector",
    "ConnectorFactory",
    "SalesforceConnector",
    "ZendeskConnector",
    "FreshdeskConnector",
    "GmailConnector",
    "WebsiteConnector"
]
