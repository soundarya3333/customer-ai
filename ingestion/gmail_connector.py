import logging
import base64
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

# Optional import to prevent crash if google-api-python-client is not installed
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.exceptions import GoogleAuthError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("google-api-python-client library not installed. Gmail connector will only operate in mock mode.")


class GmailConnector(BaseConnector):
    """
    Gmail Connector for fetching support emails.
    Connects to the Google Gmail API using OAuth credentials, decodes and parses emails,
    and supports mock mode.
    """
    def __init__(self, mock: bool = False):
        super().__init__(source_name="gmail")
        self.mock = mock or not GOOGLE_API_AVAILABLE
        self.service = None

        # Check if we have default placeholders, force mock mode if so
        if not self.mock:
            is_placeholder = (
                settings.GOOGLE_CLIENT_ID == "317830822492-52khtdf4lr28o2nth72dgqiotbp3tumu.apps.googleusercontent.com" or
                settings.GOOGLE_CLIENT_SECRET == "your_google_client_secret" or
                not settings.GOOGLE_CLIENT_ID or
                not settings.GOOGLE_CLIENT_SECRET or
                not settings.GOOGLE_REFRESH_TOKEN
            )
            if is_placeholder:
                logger.info("Placeholder credentials detected. Operating GmailConnector in Mock Mode.")
                self.mock = True

    def authenticate(self) -> None:
        """Authenticates with the Google OAuth endpoint and builds the Gmail client."""
        if self.mock:
            logger.info("Gmail Authenticating: Mock connection session established.")
            self.service = "mock_gmail_service"
            return

        logger.info("Gmail Authenticating: Building credentials from OAuth settings...")
        try:
            # We construct Credentials using the client_id, client_secret, refresh_token
            # In a production backend, token_uri is typically google OAuth token endpoint
            creds = Credentials(
                token=None,  # Will refresh token on first call
                refresh_token=settings.GOOGLE_REFRESH_TOKEN,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET
            )
            self.service = build("gmail", "v1", credentials=creds)
            logger.info("Gmail Authentication: Success.")
        except GoogleAuthError as e:
            err_msg = f"Gmail OAuth authentication failed: {str(e)}"
            logger.error(err_msg)
            raise AuthenticationError(err_msg)
        except Exception as e:
            err_msg = f"Gmail client build failure: {str(e)}"
            logger.error(err_msg)
            raise AuthenticationError(err_msg)

    def fetch(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieves unread emails from the Gmail Inbox."""
        limit = kwargs.get("limit", 20)

        if not self.service:
            raise IngestionError("Gmail service not authenticated. Call authenticate() first.")

        logger.info("Gmail Fetching: Querying unread messages from Inbox...")
        try:
            # Query for unread emails in the inbox
            # is:unread label:INBOX
            results = self.service.users().messages().list(
                userId="me",
                q="label:UNREAD label:INBOX",
                maxResults=limit
            ).execute()
            
            messages_list = results.get("messages", [])
            logger.info(f"Gmail Fetching: Found {len(messages_list)} unread messages.")
            
            raw_emails = []
            for msg_summary in messages_list:
                msg_id = msg_summary["id"]
                # Get complete message details
                msg_detail = self.service.users().messages().get(
                    userId="me",
                    id=msg_id,
                    format="full"
                ).execute()
                
                parsed_msg = self._parse_gmail_payload(msg_detail)
                raw_emails.append(parsed_msg)

            return raw_emails
        except Exception as e:
            err_msg = f"Gmail API fetch failure: {str(e)}"
            logger.error(err_msg)
            raise APIFailureError(err_msg)

    def _parse_gmail_payload(self, msg_detail: Dict[str, Any]) -> Dict[str, Any]:
        """Parses the raw JSON payload returned by Google API into a clean dict."""
        headers = msg_detail.get("payload", {}).get("headers", [])
        headers_dict = {h["name"].lower(): h["value"] for h in headers}

        # Extract basic headers
        from_val = headers_dict.get("from", "")
        subject_val = headers_dict.get("subject", "")
        date_val = headers_dict.get("date", "")

        # Extract text body and attachments
        body = self._extract_body(msg_detail.get("payload", {}))
        attachments = self._extract_attachments_metadata(msg_detail.get("payload", {}))

        return {
            "id": msg_detail.get("id"),
            "threadId": msg_detail.get("threadId"),
            "labelIds": msg_detail.get("labelIds", []),
            "from": from_val,
            "subject": subject_val,
            "body": body,
            "date": date_val,
            "attachments": attachments
        }

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Recursively parses email body parts to extract plain text."""
        body = ""
        # If payload is multipart, iterate over parts
        if "parts" in payload:
            for part in payload["parts"]:
                mime_type = part.get("mimeType")
                # Look for plain text first
                if mime_type == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        body += self._decode_base64(data)
                elif mime_type == "text/html" and not body:
                    # Fallback to HTML if plain text isn't found yet
                    data = part.get("body", {}).get("data", "")
                    if data:
                        body += self._decode_base64(data)
                elif "parts" in part:
                    # Recursive search
                    body += self._extract_body(part)
        else:
            # Single-part message
            data = payload.get("body", {}).get("data", "")
            if data:
                body = self._decode_base64(data)
        return body

    def _extract_attachments_metadata(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scans parts of the payload to extract attachment metadata."""
        attachments = []
        if "parts" in payload:
            for part in payload["parts"]:
                filename = part.get("filename")
                body = part.get("body", {})
                attachment_id = body.get("attachmentId")
                
                if filename and attachment_id:
                    attachments.append({
                        "attachmentId": attachment_id,
                        "filename": filename,
                        "mimeType": part.get("mimeType"),
                        "size": body.get("size", 0)
                    })
                elif "parts" in part:
                    attachments.extend(self._extract_attachments_metadata(part))
        return attachments

    def _decode_base64(self, encoded_data: str) -> str:
        """Decodes URL-safe base64 encoding from Gmail bodies."""
        try:
            decoded_bytes = base64.urlsafe_b64decode(encoded_data.encode("ASCII"))
            return decoded_bytes.decode("UTF-8", errors="ignore")
        except Exception:
            return ""

    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardizes raw parsed Gmail dicts to Unified Schema."""
        logger.info(f"Gmail Normalizing: Standardizing {len(raw_data)} emails...")
        normalized = []
        for raw_email in raw_data:
            normalized_email = Normalizer.normalize_gmail(raw_email)
            normalized.append(normalized_email)
        logger.info("Gmail Normalizing: Success.")
        return normalized
