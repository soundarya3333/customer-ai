from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseConnector(ABC):
    """
    Abstract Base Class for all Customer Support Data Ingestion Connectors.
    Follows Interface Segregation and Single Responsibility principles.
    """
    def __init__(self, source_name: str):
        self.source_name = source_name

    @abstractmethod
    def authenticate(self) -> None:
        """
        Establish connection or authenticate client with the data source.
        Should raise AuthenticationError on failure.
        """
        pass

    @abstractmethod
    def fetch(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve raw data payloads from the source.
        Should raise APIFailureError, RateLimitError, IngestionTimeoutError, etc. on failure.
        """
        pass

    @abstractmethod
    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert the raw data payload list into standardized unified schema format.
        Should raise MissingFieldError or ValidationError on failure.
        """
        pass

    def get_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Orchestrates authentication, fetching, and normalization.
        This provides a single template method execution flow for any connector.
        """
        self.authenticate()
        raw_data = self.fetch(**kwargs)
        normalized_data = self.normalize(raw_data)
        return normalized_data
