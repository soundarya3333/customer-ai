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

class ZendeskConnector(BaseConnector):
    """
    Zendesk Connector for fetching support tickets.
    Uses requests for basic auth token connection, handles HTTP codes, and supports mock mode.
    """
    def __init__(self, mock: bool = False):
        super().__init__(source_name="zendesk")
        self.mock = mock
        self.session = None

        # Mock mode is only active if explicitly requested via mock=True
        pass

    def authenticate(self) -> None:
        """Sets up the requests HTTP session with Zendesk Token Basic Auth."""
        if self.mock:
            logger.info("Zendesk Authenticating: Mock connection session created.")
            self.session = "mock_session"
            return

        logger.info(f"Zendesk Authenticating: Initializing API session for subdomain '{settings.ZENDESK_SUBDOMAIN}'...")
        # Zendesk API Token Auth format: email_address/token : api_token
        auth_username = f"{settings.ZENDESK_EMAIL}/token"
        self.session = requests.Session()
        self.session.auth = (auth_username, settings.ZENDESK_API_TOKEN)
        self.session.headers.update({"Accept": "application/json"})

    def fetch(self, **kwargs) -> List[Dict[str, Any]]:
        """Fetches tickets from Zendesk REST API."""
        limit = kwargs.get("limit", 50)

        if not self.session:
            raise IngestionError("Zendesk session not authenticated. Call authenticate() first.")

        url = f"https://{settings.ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
        params = {"per_page": min(limit, 100)}
        
        logger.info(f"Zendesk Fetching: Querying tickets API URL '{url}'...")
        try:
            # 10s default timeout
            response = self.session.get(url, params=params, timeout=10)
            
            # Map common HTTP status failures to custom exceptions
            if response.status_code == 401 or response.status_code == 403:
                raise AuthenticationError(f"Zendesk auth failed: {response.text}")
            elif response.status_code == 429:
                raise RateLimitError(f"Zendesk rate limit hit: {response.text}")
            
            response.raise_for_status()
            
            # Parse payload
            data = response.json()
            tickets = data.get("tickets", [])
            logger.info(f"Zendesk Fetching: Successfully retrieved {len(tickets)} tickets.")
            return tickets

        except requests.exceptions.Timeout as e:
            err_msg = f"Zendesk request timed out: {str(e)}"
            logger.error(err_msg)
            raise IngestionTimeoutError(err_msg)
        except requests.exceptions.HTTPError as e:
            err_msg = f"Zendesk API returned HTTP error: {str(e)}"
            logger.error(err_msg)
            raise APIFailureError(err_msg)
        except Exception as e:
            err_msg = f"Zendesk unexpected fetch failure: {str(e)}"
            logger.error(err_msg)
            raise APIFailureError(err_msg)

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardizes raw Zendesk ticket dicts to Unified Schema."""
        logger.info(f"Zendesk Normalizing: Standardizing {len(raw_data)} tickets...")
        normalized = []
        for raw_ticket in raw_data:
            normalized_ticket = Normalizer.normalize_zendesk(raw_ticket)
            normalized.append(normalized_ticket)
        logger.info("Zendesk Normalizing: Success.")
        return normalized
