"""FastAPI app for Customer Support AI — ticket ingestion, classification, review."""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from pydantic import BaseModel
from sqlalchemy import func
from starlette.responses import Response

from src.core.config import settings
from src.database.database import engine, Base, SessionLocal
from src.database.models import Ticket, TicketClassification, TicketPriority
from src.ml_pipeline.predictor import predict, predict_batch
from src.ml_pipeline.priority_predictor import predict_from_text_and_context


REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Customer Support AI", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.observe(duration)
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


class ClassifyRequest(BaseModel):
    text: str


@app.post("/classify")
def classify_text(req: ClassifyRequest):
    """Classify arbitrary text — does NOT touch the DB."""
    result = predict(req.text)
    return {
        "text": req.text[:100],
        "category": result["category"],
        "confidence": result["confidence"],
    }


@app.post("/analyze")
def analyze_text(req: ClassifyRequest):
    """Text-only convenience endpoint for demos: classifies category AND
    predicts priority in one call. Priority's tabular context (customer tier,
    SLA clock, repeat contacts) isn't derivable from text alone and isn't
    tracked anywhere for arbitrary input, so this uses neutral defaults —
    good enough to show the two models working together without requiring
    the caller to fill in fields live."""
    category_result = predict(req.text)
    priority_result = predict_from_text_and_context(
        req.text,
        category_result["category"],
    )
    return {
        "text": req.text[:200],
        "category": category_result["category"],
        "category_confidence": category_result["confidence"],
        "priority": priority_result["priority"],
        "priority_confidence": priority_result["confidence"],
        "priority_probabilities": priority_result["probabilities"],
    }


@app.post("/tickets/{ticket_id}/classify")
def classify_ticket(ticket_id: str):
    """Classify an existing ticket from DB; store prediction in TicketClassification."""
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).filter_by(ticket_id=ticket_id).first()
        if not ticket:
            return {"error": "Ticket not found", "ticket_id": ticket_id}

        text = f"{ticket.subject} {ticket.customer_message}"
        result = predict(text)
        category = result["category"]
        confidence = result["confidence"]

        existing = db.query(TicketClassification).filter_by(ticket_id=ticket_id).first()
        if existing:
            existing.predicted_category = category
            existing.confidence = confidence
            existing.predicted_at = datetime.utcnow()
            existing.status = "pending_review"
        else:
            db.add(TicketClassification(
                ticket_id=ticket_id,
                predicted_category=category,
                confidence=confidence,
                status="pending_review",
            ))

        db.commit()
        return {
            "ticket_id": ticket_id,
            "category": category,
            "confidence": confidence,
            "status": "pending_review",
        }
    finally:
        db.close()


class PriorityRequest(BaseModel):
    text: str
    category: str
    customer_tier: str = "Basic"
    sla_hours_remaining: float = 24.0
    ticket_age_hours: float = 0.0
    repeat_contact_count: int = 0
    source_channel: str = "website"


@app.post("/predict-priority")
def predict_priority(req: PriorityRequest):
    """Predict priority from raw text + explicit context features — does NOT touch the DB."""
    result = predict_from_text_and_context(
        req.text,
        req.category,
        customer_tier=req.customer_tier,
        sla_hours_remaining=req.sla_hours_remaining,
        ticket_age_hours=req.ticket_age_hours,
        repeat_contact_count=req.repeat_contact_count,
        source_channel=req.source_channel,
    )
    return result


class TicketPriorityOverrides(BaseModel):
    """Tabular context features not yet tracked anywhere in this system's DB
    schema (no CRM/SLA-engine integration exists yet) — caller supplies them
    until that integration exists. All optional; sensible defaults apply."""
    customer_tier: str = "Basic"
    sla_hours_remaining: float = 24.0
    repeat_contact_count: int = 0


@app.post("/tickets/{ticket_id}/predict-priority")
def predict_ticket_priority(ticket_id: str, overrides: TicketPriorityOverrides = TicketPriorityOverrides()):
    """Predict priority for an existing DB ticket; store in TicketPriority.

    Derives what's real from the ticket record (source channel, ticket age
    from its timestamp, category from its stored classification if one
    exists, else runs the category classifier). customer_tier, SLA clock,
    and repeat-contact count aren't tracked in this DB yet, so they come from
    `overrides` (defaults apply if omitted)."""
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).filter_by(ticket_id=ticket_id).first()
        if not ticket:
            return {"error": "Ticket not found", "ticket_id": ticket_id}

        text = f"{ticket.subject} {ticket.customer_message}"

        existing_classification = db.query(TicketClassification).filter_by(ticket_id=ticket_id).first()
        if existing_classification:
            category = existing_classification.corrected_category or existing_classification.predicted_category
        else:
            category = predict(text)["category"]

        try:
            created = datetime.fromisoformat(ticket.timestamp.replace("Z", "+00:00"))
            ticket_age_hours = max(0.0, (datetime.now(created.tzinfo) - created).total_seconds() / 3600)
        except (ValueError, AttributeError):
            ticket_age_hours = 0.0

        result = predict_from_text_and_context(
            text,
            category,
            customer_tier=overrides.customer_tier,
            sla_hours_remaining=overrides.sla_hours_remaining,
            ticket_age_hours=ticket_age_hours,
            repeat_contact_count=overrides.repeat_contact_count,
            source_channel=ticket.source,
        )

        existing = db.query(TicketPriority).filter_by(ticket_id=ticket_id).first()
        if existing:
            existing.predicted_priority = result["priority"]
            existing.confidence = result["confidence"]
            existing.probabilities = result["probabilities"]
            existing.predicted_at = datetime.utcnow()
            existing.status = "pending_review"
        else:
            db.add(TicketPriority(
                ticket_id=ticket_id,
                predicted_priority=result["priority"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            ))
        db.commit()

        return {
            "ticket_id": ticket_id,
            "category_used": category,
            "ticket_age_hours": round(ticket_age_hours, 1),
            **result,
        }
    finally:
        db.close()


@app.get("/queue/pending-combined")
def pending_combined():
    """Tickets that have at least one pending prediction (category or priority).
    Returns both predictions side-by-side for the unified review dashboard."""
    db = SessionLocal()
    try:
        tickets = db.query(Ticket).all()
        result = []
        for t in tickets:
            cat = db.query(TicketClassification).filter_by(ticket_id=t.ticket_id).first()
            pri = db.query(TicketPriority).filter_by(ticket_id=t.ticket_id).first()
            if not cat and not pri:
                continue
            if cat and cat.status not in ("pending_review",) and pri and pri.status not in ("pending_review",):
                continue
            result.append({
                "ticket_id": t.ticket_id,
                "source": t.source,
                "subject": t.subject,
                "message": t.customer_message[:150],
                "category": {
                    "value": cat.predicted_category if cat else None,
                    "confidence": cat.confidence if cat else None,
                    "status": cat.status if cat else None,
                } if cat else None,
                "priority": {
                    "value": pri.predicted_priority if pri else None,
                    "confidence": pri.confidence if pri else None,
                    "status": pri.status if pri else None,
                } if pri else None,
            })
        return result
    finally:
        db.close()


@app.get("/queue/pending-category")
def pending_categories():
    """List all tickets awaiting review."""
    db = SessionLocal()
    try:
        rows = (
            db.query(TicketClassification, Ticket)
            .join(Ticket, TicketClassification.ticket_id == Ticket.ticket_id)
            .filter(TicketClassification.status == "pending_review")
            .all()
        )
        return [
            {
                "ticket_id": t.ticket_id,
                "source": t.source,
                "subject": t.subject,
                "message": t.customer_message[:150],
                "predicted_category": c.predicted_category,
                "confidence": c.confidence,
            }
            for c, t in rows
        ]
    finally:
        db.close()


class ApproveRequest(BaseModel):
    ticket_id: str


class CorrectRequest(BaseModel):
    ticket_id: str
    category: str


@app.post("/review/category/approve")
def approve_category(req: ApproveRequest):
    """Approve the AI's predicted category."""
    db = SessionLocal()
    try:
        c = db.query(TicketClassification).filter_by(ticket_id=req.ticket_id).first()
        if not c:
            return {"error": "Classification not found"}
        c.status = "approved"
        c.reviewed_at = datetime.utcnow()
        db.commit()
        return {"status": "approved", "ticket_id": req.ticket_id}
    finally:
        db.close()


@app.post("/review/category/correct")
def correct_category(req: CorrectRequest):
    """Manually correct the predicted category."""
    db = SessionLocal()
    try:
        c = db.query(TicketClassification).filter_by(ticket_id=req.ticket_id).first()
        if not c:
            return {"error": "Classification not found"}
        c.status = "corrected"
        c.corrected_category = req.category
        c.reviewed_at = datetime.utcnow()
        db.commit()
        return {"status": "corrected", "ticket_id": req.ticket_id, "corrected_category": req.category}
    finally:
        db.close()


@app.get("/queue/stats")
def queue_stats():
    """Counts of tickets by review status, for the dashboard header."""
    db = SessionLocal()
    try:
        counts = {"pending_review": 0, "approved": 0, "corrected": 0}
        for status, count in (
            db.query(TicketClassification.status, func.count(TicketClassification.ticket_id))
            .group_by(TicketClassification.status)
            .all()
        ):
            counts[status] = count
        return {
            "pending": counts["pending_review"],
            "approved": counts["approved"],
            "corrected": counts["corrected"],
        }
    finally:
        db.close()


@app.get("/queue/pending-priority")
def pending_priorities():
    """List all tickets with a priority prediction awaiting review."""
    db = SessionLocal()
    try:
        rows = (
            db.query(TicketPriority, Ticket)
            .join(Ticket, TicketPriority.ticket_id == Ticket.ticket_id)
            .filter(TicketPriority.status == "pending_review")
            .all()
        )
        return [
            {
                "ticket_id": t.ticket_id,
                "source": t.source,
                "subject": t.subject,
                "message": t.customer_message[:150],
                "predicted_priority": p.predicted_priority,
                "confidence": p.confidence,
                "probabilities": p.probabilities,
            }
            for p, t in rows
        ]
    finally:
        db.close()


class ApprovePriorityRequest(BaseModel):
    ticket_id: str


class CorrectPriorityRequest(BaseModel):
    ticket_id: str
    priority: str


@app.post("/review/priority/approve")
def approve_priority(req: ApprovePriorityRequest):
    """Approve the model's predicted priority."""
    db = SessionLocal()
    try:
        p = db.query(TicketPriority).filter_by(ticket_id=req.ticket_id).first()
        if not p:
            return {"error": "Priority prediction not found"}
        p.status = "approved"
        p.reviewed_at = datetime.utcnow()
        db.commit()
        return {"status": "approved", "ticket_id": req.ticket_id}
    finally:
        db.close()


@app.post("/review/priority/correct")
def correct_priority(req: CorrectPriorityRequest):
    """Manually correct the predicted priority."""
    db = SessionLocal()
    try:
        p = db.query(TicketPriority).filter_by(ticket_id=req.ticket_id).first()
        if not p:
            return {"error": "Priority prediction not found"}
        p.status = "corrected"
        p.corrected_priority = req.priority
        p.reviewed_at = datetime.utcnow()
        db.commit()
        return {"status": "corrected", "ticket_id": req.ticket_id, "corrected_priority": req.priority}
    finally:
        db.close()


@app.get("/queue/priority-stats")
def priority_queue_stats():
    """Counts of priority predictions by review status, for the dashboard header."""
    db = SessionLocal()
    try:
        counts = {"pending_review": 0, "approved": 0, "corrected": 0}
        for status, count in (
            db.query(TicketPriority.status, func.count(TicketPriority.ticket_id))
            .group_by(TicketPriority.status)
            .all()
        ):
            counts[status] = count
        return {
            "pending": counts["pending_review"],
            "approved": counts["approved"],
            "corrected": counts["corrected"],
        }
    finally:
        db.close()


@app.get("/dashboard/review", response_class=HTMLResponse)
def review_dashboard():
    """Simple HTML dashboard for reviewing classifications."""
    path = os.path.join(os.path.dirname(__file__), "review_dashboard.html")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "<h1>Dashboard not found</h1>"


# ── Ingestion + list endpoints (used by Airflow DAG) ───────────────────

class IngestRequest(BaseModel):
    source: str
    data: dict


@app.post("/ingest")
def ingest_ticket(req: IngestRequest):
    """Normalize and insert a ticket from any supported source."""
    from ingestion.normalizer import Normalizer

    try:
        normalized = Normalizer.normalize(req.source, req.data)
    except Exception as e:
        return {"error": str(e)}

    db = SessionLocal()
    try:
        existing = db.query(Ticket).filter_by(ticket_id=normalized["ticket_id"]).first()
        if existing:
            return {"ticket_id": normalized["ticket_id"], "status": "already_exists"}

        normalized["ticket_metadata"] = normalized.pop("metadata", {})
        ticket = Ticket(**normalized)
        db.add(ticket)
        db.commit()
        return {"ticket_id": normalized["ticket_id"], "status": "inserted"}
    finally:
        db.close()


@app.get("/tickets")
def list_tickets():
    """Return all tickets in the DB as a flat list."""
    db = SessionLocal()
    try:
        rows = db.query(Ticket).all()
        return [
            {
                "ticket_id": t.ticket_id,
                "source": t.source,
                "subject": t.subject,
                "customer_message": t.customer_message[:200],
                "priority": t.priority,
                "status": t.status,
            }
            for t in rows
        ]
    finally:
        db.close()