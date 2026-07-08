import logging
from typing import List, Dict, Any, Optional
from ingestion.base_connector import BaseConnector
from ingestion.config import settings, logger
from ingestion.normalizer import Normalizer
from ingestion.exceptions import (
    AuthenticationError,
    APIFailureError,
    IngestionTimeoutError,
    IngestionError
)

# Optional import to prevent immediate crash if dependency is missing
try:
    from simple_salesforce import Salesforce
    from simple_salesforce.exceptions import SalesforceAuthenticationFailed, SalesforceError
    SIMPLE_SF_AVAILABLE = True
except ImportError:
    SIMPLE_SF_AVAILABLE = False
    logger.warning("simple-salesforce library not installed. Salesforce connector will only operate in mock mode.")


class SalesforceConnector(BaseConnector):
    """
    Salesforce Connector (Temporary Empty Stub).
    Returns empty list for now. Mock / Real implementation can be done later.
    """
    def __init__(self, mock: bool = False):
        super().__init__(source_name="salesforce")

    def authenticate(self) -> None:
        """Stateless / empty pass for now."""
        logger.info("Salesforce Authenticating: Pass (stub mode).")

    def fetch(self, **kwargs) -> List[Dict[str, Any]]:
        """Returns empty list for now."""
        logger.info("Salesforce Fetching: Returning empty list (stub mode).")
        return []

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns empty list for now."""
        return []
