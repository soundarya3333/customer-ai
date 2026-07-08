import logging
import requests
from typing import List, Dict, Any, Optional
from ingestion.base_connector import BaseConnector
from ingestion.config import settings, logger
from ingestion.normalizer import Normalizer
from ingestion.exceptions import (
    AuthenticationError,
    RateLimitError,
    IngestionTimeoutError,
    APIFailureError,
    IngestionError
)

class FreshdeskConnector(BaseConnector):
    """
    Freshdesk Connector for fetching support tickets.
    Uses requests for API key connection, maps numeric values, and supports mock mode.
    """
    def __init__(self, mock: bool = False):
        super().__init__(source_name="freshdesk")
        self.mock = mock
        self.session = None

        # Mock mode is only active if explicitly requested via mock=True
        pass

    def authenticate(self) -> None:
        """Sets up the requests HTTP session with Freshdesk API Key (Basic Auth)."""
        if self.mock:
            logger.info("Freshdesk Authenticating: Mock connection session created.")
            self.session = "mock_session"
            return

        logger.info(f"Freshdesk Authenticating: Initializing API session for domain '{settings.FRESHDESK_DOMAIN}'...")
        # Freshdesk basic auth: API key as username, dummy 'X' as password
        self.session = requests.Session()
        self.session.auth = (settings.FRESHDESK_API_KEY, "X")
        self.session.headers.update({"Accept": "application/json"})

    def fetch(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetches tickets from Freshdesk REST API."""
        limit = kwargs.get("limit", 50)

        if not self.session:
            raise IngestionError("Freshdesk session not authenticated. Call authenticate() first.")

        # API endpoint: https://domain.freshdesk.com/api/v2/tickets
        url = f"https://{settings.FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets"
        params = {"per_page": min(limit, 100)}
        
        logger.info(f"Freshdesk Fetching: Querying tickets API URL '{url}'...")
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            # Map common HTTP status failures to custom exceptions
            if response.status_code == 401 or response.status_code == 403:
                raise AuthenticationError(f"Freshdesk auth failed: {response.text}")
            elif response.status_code == 429:
                raise RateLimitError(f"Freshdesk rate limit hit: {response.text}")
            
            response.raise_for_status()
            
            # Parse payload
            tickets = response.json()
            logger.info(f"Freshdesk Fetching: Successfully retrieved {len(tickets)} tickets.")
            return tickets

        except requests.exceptions.Timeout as e:
            err_msg = f"Freshdesk request timed out: {str(e)}"
            logger.error(err_msg)
            raise IngestionTimeoutError(err_msg)
        except requests.exceptions.HTTPError as e:
            err_msg = f"Freshdesk API returned HTTP error: {str(e)}"
            logger.error(err_msg)
            raise APIFailureError(err_msg)
        except Exception as e:
            err_msg = f"Freshdesk unexpected fetch failure: {str(e)}"
            logger.error(err_msg)
            raise APIFailureError(err_msg)

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardizes raw Freshdesk ticket dicts to Unified Schema."""
        logger.info(f"Freshdesk Normalizing: Standardizing {len(raw_data)} tickets...")
        normalized = []
        for raw_ticket in raw_data:
            normalized_ticket = Normalizer.normalize_freshdesk(raw_ticket)
            normalized.append(normalized_ticket)
        logger.info("Freshdesk Normalizing: Success.")
        return normalized
