from ingestion.base_connector import BaseConnector
from ingestion.salesforce_connector import SalesforceConnector
from ingestion.zendesk_connector import ZendeskConnector
from ingestion.freshdesk_connector import FreshdeskConnector
from ingestion.gmail_connector import GmailConnector
from ingestion.website_connector import WebsiteConnector
from ingestion.exceptions import ValidationError

class ConnectorFactory:
    """
    Factory Method Pattern registry for Support Data Ingestion Connectors.
    Allows dynamic retrieval of the correct connector class instance by name.
    """
    
    # Map sources to their corresponding connector class definitions
    _registry = {
        "salesforce": SalesforceConnector,
        "zendesk": ZendeskConnector,
        "freshdesk": FreshdeskConnector,
        "gmail": GmailConnector,
        "website": WebsiteConnector
    }

    @classmethod
    def get_connector(cls, source_name: str, mock: bool = False, **kwargs) -> BaseConnector:
        """
        Resolves, instantiates, and returns the requested BaseConnector implementation.
        
        :param source_name: Name of the platform to ingest from (case-insensitive).
        :param mock: Flag to force mock mode in connector.
        :param kwargs: Additional arguments passed to the connector constructor.
        :return: An initialized instance of a BaseConnector subclass.
        """
        clean_name = source_name.strip().lower()
        connector_class = cls._registry.get(clean_name)
        
        if not connector_class:
            raise ValidationError(
                f"Unsupported connector source name: '{source_name}'. "
                f"Supported platforms are: {list(cls._registry.keys())}"
            )
            
        return connector_class(mock=mock, **kwargs)

    @classmethod
    def register_connector(cls, source_name: str, connector_class: type) -> None:
        """
        Programmatically registers a new connector type.
        Supports OCP (Open-Closed Principle) by allowing third-party connector registration 
        without modifying this module.
        """
        clean_name = source_name.strip().lower()
        if not issubclass(connector_class, BaseConnector):
            raise ValidationError("Registered connector class must inherit from BaseConnector.")
        cls._registry[clean_name] = connector_class
