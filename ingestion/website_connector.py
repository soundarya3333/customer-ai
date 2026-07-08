import logging
from typing import List, Dict, Any
from ingestion.base_connector import BaseConnector
from ingestion.config import logger


class WebsiteConnector(BaseConnector):
    def __init__(self, mock: bool = False):
        super().__init__(source_name="website")

    def authenticate(self) -> None:
        logger.info("Website Authenticating: Pass (stub mode).")

    def fetch(self, **kwargs) -> List[Dict[str, Any]]:
        logger.info("Website Fetching: Returning empty list (stub mode).")
        return []

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []
