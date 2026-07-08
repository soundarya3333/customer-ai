import re
import html
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from ingestion.exceptions import ValidationError, MissingFieldError

logger = logging.getLogger("ingestion.normalizer")

# Standard Priority and Status validation sets
VALID_PRIORITIES = {"Low", "Medium", "High", "Urgent"}
VALID_STATUSES = {"New", "Open", "Pending", "Hold", "Resolved", "Solved", "Closed"}

class UnifiedTicket(BaseModel):
    """
    Pydantic Model defining the exact Unified Ticket Schema.
    Downstream AI pipelines rely on this structure.
    """
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_id: str = Field(..., description="Customer identifier or email")
    source: str = Field(..., description="Source system: salesforce, zendesk, freshdesk, gmail, website")
    subject: str = Field(..., description="Ticket subject line")
    customer_message: str = Field(..., description="Plain text customer message body")
    priority: str = Field(..., description="Standardized priority: Low, Medium, High, Urgent")
    status: str = Field(..., description="Standardized status: Open, Pending, Resolved, Closed, etc.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp string")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="List of attachment metadata dicts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific additional metadata")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        sources = {"salesforce", "zendesk", "freshdesk", "gmail", "website"}
        if v.lower() not in sources:
            raise ValueError(f"Source must be one of {sources}")
        return v.lower()

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        # Standardize capitalization
        val = v.strip().capitalize()
        if val not in VALID_PRIORITIES:
            return "Medium"  # Default fallback for safety
        return val

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        val = v.strip().capitalize()
        # Map Solved to Resolved if necessary, or keep standard
        if val == "Solved":
            val = "Resolved"
        if val not in VALID_STATUSES:
            return "Open"  # Default fallback
        return val


def strip_html(text: str) -> str:
    """Helper utility to strip HTML tags and decode HTML entities from text."""
    if not text:
        return ""
    # Strip HTML tags
    clean_text = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities
    clean_text = html.unescape(clean_text)
    # Collapse multiple whitespaces
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.strip()


def parse_email_sender(sender_str: str) -> str:
    """Extract clean email address from sender string (e.g. 'John Doe <john@example.com>' -> 'john@example.com')."""
    if not sender_str:
        return "unknown_sender"
    match = re.search(r"<([^>]+)>", sender_str)
    if match:
        return match.group(1).strip()
    return sender_str.strip()


def format_iso_timestamp(date_str: str) -> str:
    """Format diverse date formats into clean ISO 8601 UTC timestamp format."""
    if not date_str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        # Try common patterns or use dateutil/pandas if needed, but standard datetime is safer
        # Salesforce: '2026-07-01T10:00:00.000+0000' -> remove milliseconds and timezones for standard parse
        clean_date = date_str.replace("Z", "+00:00")
        if "." in clean_date:
            clean_date = clean_date.split(".")[0]
        # Parse and format back to UTC ISO
        dt = datetime.fromisoformat(clean_date)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        logger.warning(f"Could not parse timestamp '{date_str}': {e}. Using current time.")
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Normalizer:
    """
    Central Normalizer class that houses the normalization mapping logic
    for all five connectors (Salesforce, Zendesk, Freshdesk, Gmail, Website).
    """

    @staticmethod
    def normalize_salesforce(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Maps a raw Salesforce Case record to the Unified Schema."""
        try:
            # Check for critical fields
            if "Id" not in raw:
                raise MissingFieldError("Salesforce raw payload is missing case 'Id'")
            
            # Map values
            ticket_id = str(raw.get("Id"))
            customer_id = raw.get("ContactId") or raw.get("ContactEmail") or raw.get("SuppliedEmail") or "sf_contact_unknown"
            subject = raw.get("Subject") or "No Subject"
            customer_message = raw.get("Description") or ""
            priority = raw.get("Priority") or "Medium"
            status = raw.get("Status") or "Open"
            timestamp = format_iso_timestamp(raw.get("CreatedDate"))

            # Build metadata from remaining keys
            metadata_keys = {"CaseNumber", "Origin", "Reason", "Type", "AccountId", "SuppliedName"}
            metadata = {k: raw[k] for k in metadata_keys if k in raw}

            unified_dict = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "source": "salesforce",
                "subject": subject,
                "customer_message": customer_message,
                "priority": priority,
                "status": status,
                "timestamp": timestamp,
                "attachments": [],  # Extensible for future SF attachment sub-queries
                "metadata": metadata
            }

            # Validate using Pydantic model
            validated = UnifiedTicket(**unified_dict)
            logger.debug(f"Successfully normalized Salesforce case {ticket_id}")
            return validated.model_dump()
        except MissingFieldError:
            raise
        except Exception as e:
            raise ValidationError(f"Salesforce normalization validation failed: {str(e)}")

    @staticmethod
    def normalize_zendesk(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Maps a raw Zendesk ticket record to the Unified Schema."""
        try:
            if "id" not in raw:
                raise MissingFieldError("Zendesk raw payload is missing ticket 'id'")

            ticket_id = str(raw.get("id"))
            customer_id = str(raw.get("requester_id") or raw.get("submitter_id") or "zd_customer_unknown")
            subject = raw.get("subject") or "No Subject"
            customer_message = raw.get("description") or ""
            priority = raw.get("priority") or "Medium"
            status = raw.get("status") or "Open"
            timestamp = format_iso_timestamp(raw.get("created_at"))

            # Extract attachments if comment attachments exist
            attachments = []
            if "attachments" in raw:
                for att in raw["attachments"]:
                    attachments.append({
                        "id": str(att.get("id")),
                        "filename": att.get("file_name"),
                        "content_url": att.get("content_url"),
                        "content_type": att.get("content_type")
                    })

            # Custom metadata
            metadata_keys = {"tags", "custom_fields", "recipient", "group_id", "assignee_id", "url"}
            metadata = {k: raw[k] for k in metadata_keys if k in raw}

            unified_dict = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "source": "zendesk",
                "subject": subject,
                "customer_message": customer_message,
                "priority": priority,
                "status": status,
                "timestamp": timestamp,
                "attachments": attachments,
                "metadata": metadata
            }

            validated = UnifiedTicket(**unified_dict)
            logger.debug(f"Successfully normalized Zendesk ticket {ticket_id}")
            return validated.model_dump()
        except MissingFieldError:
            raise
        except Exception as e:
            raise ValidationError(f"Zendesk normalization validation failed: {str(e)}")

    @staticmethod
    def normalize_freshdesk(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Maps a raw Freshdesk ticket record to the Unified Schema."""
        try:
            if "id" not in raw:
                raise MissingFieldError("Freshdesk raw payload is missing ticket 'id'")

            ticket_id = str(raw.get("id"))
            customer_id = str(raw.get("requester_id") or "fd_customer_unknown")
            subject = raw.get("subject") or "No Subject"
            
            # Freshdesk description is typically HTML, strip it for plain text
            raw_description = raw.get("description") or ""
            customer_message = strip_html(raw_description)

            # Map Freshdesk numeric priority: 1=Low, 2=Medium, 3=High, 4=Urgent
            priority_map = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}
            priority_num = raw.get("priority")
            priority = priority_map.get(priority_num, "Medium")

            # Map Freshdesk numeric status: 2=Open, 3=Pending, 4=Resolved, 5=Closed
            status_map = {2: "Open", 3: "Pending", 4: "Resolved", 5: "Closed"}
            status_num = raw.get("status")
            status = status_map.get(status_num, "Open")

            timestamp = format_iso_timestamp(raw.get("created_at"))

            # Parse attachments
            attachments = []
            if "attachments" in raw:
                for att in raw["attachments"]:
                    attachments.append({
                        "id": str(att.get("id")),
                        "filename": att.get("name"),
                        "content_url": att.get("attachment_url"),
                        "content_type": att.get("content_type"),
                        "size": att.get("size")
                    })

            # Custom metadata (store original HTML description for downstream preservation)
            metadata_keys = {"responder_id", "company_id", "tags", "due_by", "fr_due_by", "type"}
            metadata = {k: raw[k] for k in metadata_keys if k in raw}
            metadata["raw_html_description"] = raw_description

            unified_dict = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "source": "freshdesk",
                "subject": subject,
                "customer_message": customer_message,
                "priority": priority,
                "status": status,
                "timestamp": timestamp,
                "attachments": attachments,
                "metadata": metadata
            }

            validated = UnifiedTicket(**unified_dict)
            logger.debug(f"Successfully normalized Freshdesk ticket {ticket_id}")
            return validated.model_dump()
        except MissingFieldError:
            raise
        except Exception as e:
            raise ValidationError(f"Freshdesk normalization validation failed: {str(e)}")

    @staticmethod
    def normalize_gmail(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Maps a raw Gmail message object to the Unified Schema."""
        try:
            if "id" not in raw:
                raise MissingFieldError("Gmail raw payload is missing message 'id'")

            ticket_id = str(raw.get("id"))
            customer_id = parse_email_sender(raw.get("from"))
            subject = raw.get("subject") or "No Subject"
            customer_message = raw.get("body") or ""
            priority = raw.get("priority") or "Medium"
            status = raw.get("status") or "Open"
            timestamp = format_iso_timestamp(raw.get("date"))

            # Custom metadata
            metadata_keys = {"threadId", "labelIds", "to", "cc", "historyId"}
            metadata = {k: raw[k] for k in metadata_keys if k in raw}

            # Map attachments if metadata is present
            attachments = []
            if "attachments" in raw:
                for att in raw["attachments"]:
                    attachments.append({
                        "id": att.get("attachmentId"),
                        "filename": att.get("filename"),
                        "content_type": att.get("mimeType"),
                        "size": att.get("size")
                    })

            unified_dict = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "source": "gmail",
                "subject": subject,
                "customer_message": customer_message,
                "priority": priority,
                "status": status,
                "timestamp": timestamp,
                "attachments": attachments,
                "metadata": metadata
            }

            validated = UnifiedTicket(**unified_dict)
            logger.debug(f"Successfully normalized Gmail message {ticket_id}")
            return validated.model_dump()
        except MissingFieldError:
            raise
        except Exception as e:
            raise ValidationError(f"Gmail normalization validation failed: {str(e)}")

    @staticmethod
    def normalize_website(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps a raw Website payload (which mirrors the Hugging Face fine-tuning dataset)
        to the Unified Schema.
        """
        try:
            # Check for Hugging Face or direct properties
            # Fine-tuning dataset uses 'instruction' for user query, and 'category' / 'intent' for classification
            customer_message = raw.get("instruction") or raw.get("customer_message")
            if not customer_message:
                raise MissingFieldError("Website/Dataset payload is missing 'instruction' or 'customer_message'")

            # Use provided or generate missing IDs
            ticket_id = raw.get("ticket_id") or f"web_{uuid.uuid4().hex[:8]}"
            customer_id = raw.get("customer_id") or "web_customer_anonymous"
            
            # Map subject from intent or category if available
            intent = raw.get("intent") or ""
            category = raw.get("category") or ""
            
            subject = raw.get("subject") or f"Website Support Query: {intent or category or 'General Inquiry'}"
            
            # Priority can be inferred from category (e.g. billing/payment is high priority)
            priority = raw.get("priority")
            if not priority:
                high_priority_categories = {"payment", "refund", "billing", "security", "urgent", "account", "login"}
                if any(c in category.lower() or c in intent.lower() for c in high_priority_categories):
                    priority = "High"
                else:
                    priority = "Medium"

            status = raw.get("status") or "Open"
            timestamp = format_iso_timestamp(raw.get("timestamp"))

            # Keep HF details in metadata
            metadata = {
                "category": category,
                "intent": intent,
                "flags": raw.get("flags") or "",
                "hf_response": raw.get("response") or "",
                "ip_address": raw.get("ip_address") or "unknown"
            }
            # Copy other arbitrary input keys
            for k, v in raw.items():
                if k not in {"instruction", "customer_message", "ticket_id", "customer_id", "subject", 
                             "priority", "status", "timestamp", "category", "intent", "flags", "response"}:
                    metadata[k] = v

            unified_dict = {
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "source": "website",
                "subject": subject,
                "customer_message": customer_message,
                "priority": priority,
                "status": status,
                "timestamp": timestamp,
                "attachments": [],
                "metadata": metadata
            }

            validated = UnifiedTicket(**unified_dict)
            logger.debug(f"Successfully normalized Website ticket {ticket_id}")
            return validated.model_dump()
        except MissingFieldError:
            raise
        except Exception as e:
            raise ValidationError(f"Website/Dataset normalization validation failed: {str(e)}")

    @classmethod
    def normalize(cls, source: str, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates normalization to the correct source method."""
        src_lower = source.lower().strip()
        if src_lower == "salesforce":
            return cls.normalize_salesforce(raw_payload)
        elif src_lower == "zendesk":
            return cls.normalize_zendesk(raw_payload)
        elif src_lower == "freshdesk":
            return cls.normalize_freshdesk(raw_payload)
        elif src_lower == "gmail":
            return cls.normalize_gmail(raw_payload)
        elif src_lower == "website":
            return cls.normalize_website(raw_payload)
        else:
            raise ValidationError(f"Unsupported connector source for normalization: {source}")
