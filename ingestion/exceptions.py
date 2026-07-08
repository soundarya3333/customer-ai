"""
Custom exception hierarchy for the data ingestion layer.
All exceptions are structured to categorize failures during authentication,
fetching, rate limiting, timeouts, or normalization.
"""

class IngestionError(Exception):
    """Base exception for all ingestion errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(IngestionError):
    """Raised when authentication with a connector's source fails."""
    pass


class RateLimitError(IngestionError):
    """Raised when request rate limits are hit (e.g. HTTP 429)."""
    pass


class IngestionTimeoutError(IngestionError):
    """Raised when a request to a remote source times out."""
    pass


class APIFailureError(IngestionError):
    """Raised when an API returns an error or unexpected status code."""
    pass


class MissingFieldError(IngestionError):
    """Raised when a raw payload lacks fields necessary for normalization."""
    pass


class ValidationError(IngestionError):
    """Raised when normalized data does not conform to the Unified Schema."""
    pass
