from sqlalchemy import Column, String, JSON, DateTime, Float, ForeignKey, func
from sqlalchemy.orm import relationship
from datetime import datetime

from src.database.database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id = Column(String, primary_key=True)
    customer_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False, index=True)
    subject = Column(String, nullable=False)
    customer_message = Column(String, nullable=False)
    priority = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    timestamp = Column(String, nullable=False)
    attachments = Column(JSON, default=[])
    ticket_metadata = Column("metadata", JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to classification
    classification = relationship("TicketClassification", backref="ticket", uselist=False)


class TicketClassification(Base):
    __tablename__ = "ticket_classifications"

    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), primary_key=True)
    predicted_category = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    status = Column(String, default="pending_review", index=True)  # pending_review, approved, corrected
    corrected_category = Column(String, nullable=True)

    predicted_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)


class TicketPriority(Base):
    __tablename__ = "ticket_priorities"

    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), primary_key=True)
    predicted_priority = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, default={})
    status = Column(String, default="pending_review", index=True)  # pending_review, approved, corrected
    corrected_priority = Column(String, nullable=True)

    predicted_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
