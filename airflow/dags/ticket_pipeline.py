"""
Airflow DAG: ticket_pipeline

Runs every hour to process new tickets through the full pipeline:
  ingest -> classify -> predict priority -> log summary

Depends on:
  - FastAPI app running at http://app:8000
  - PostgreSQL at postgres:5432
"""
import json
import logging
import urllib.request
from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

API_BASE = "http://app:8000"


def _api_post(path, body=None):
    """POST to the FastAPI service and return parsed JSON."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"} if data else {},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def _api_get(path):
    """GET from the FastAPI service and return parsed JSON."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, method="GET")
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

SOURCE_TO_TIER = {
    "salesforce": "Enterprise",
    "zendesk": "Pro",
    "freshdesk": "Pro",
    "gmail": "Basic",
    "website": "Basic",
}


def ingest_new_tickets(**context):
    """Generate new mock tickets and ingest them via the API.

    Simulates live ticket flow: generates a small batch of fresh tickets
    with timestamps set to now, then POSTs them to the /ingest endpoint.
    In production, this would pull from real connectors instead.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    sources = ["zendesk", "freshdesk", "salesforce", "gmail", "website"]
    tickets_ingested = 0

    sample_tickets = [
        {
            "source": "zendesk",
            "data": {
                "id": 90000 + int(now.timestamp()) % 10000,
                "requester_id": 500001,
                "subject": "System outage affecting production workloads",
                "description": "Our production environment has been down for 30 minutes. Multiple services are returning 503 errors. This is blocking all customer-facing operations.",
                "priority": "urgent",
                "status": "open",
                "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tags": ["outage", "production"],
                "attachments": [],
            },
        },
        {
            "source": "freshdesk",
            "data": {
                "id": 95000 + int(now.timestamp()) % 10000,
                "requester_id": 600001,
                "subject": "Cannot access billing portal after password reset",
                "description": "<div>Reset my password successfully but the billing portal still says 'Access Denied'. Tried clearing cache and different browsers.</div>",
                "priority": 2,
                "status": 2,
                "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "attachments": [],
            },
        },
        {
            "source": "gmail",
            "data": {
                "id": f"gm_{int(now.timestamp()) % 100000}",
                "threadId": f"thread_{int(now.timestamp()) % 100000}",
                "labelIds": ["INBOX", "UNREAD"],
                "from": "Customer <new.customer@example.com>",
                "subject": "Refund not received after 30 days",
                "body": "I returned my order over a month ago and still haven't received my refund. Reference number: REF-2026-88421. Please escalate.",
                "date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "attachments": [],
            },
        },
    ]

    for ticket in sample_tickets:
        try:
            result = _api_post("/ingest", {
                "source": ticket["source"],
                "data": ticket["data"],
            })
            tickets_ingested += 1
            logger.info(f"Ingested {ticket['source']} ticket: {result.get('ticket_id', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to ingest {ticket['source']} ticket: {e}")

    context["ti"].xcom_push(key="tickets_ingested", value=tickets_ingested)
    logger.info(f"Ingested {tickets_ingested} new tickets")
    return tickets_ingested


def classify_batch(**context):
    """Classify all unclassified tickets via the API."""
    tickets = _api_get("/tickets")
    classified = 0
    errors = 0

    for ticket in tickets:
        ticket_id = ticket.get("ticket_id")
        if not ticket_id:
            continue
        try:
            _api_post(f"/tickets/{ticket_id}/classify")
            classified += 1
        except Exception as e:
            errors += 1
            logger.warning(f"Failed to classify {ticket_id}: {e}")

    context["ti"].xcom_push(key="classified", value=classified)
    context["ti"].xcom_push(key="classify_errors", value=errors)
    logger.info(f"Classified {classified} tickets ({errors} errors)")
    return classified


def predict_priority_batch(**context):
    """Predict priority for all tickets via the API."""
    tickets = _api_get("/tickets")
    predicted = 0
    errors = 0

    for ticket in tickets:
        ticket_id = ticket.get("ticket_id")
        if not ticket_id:
            continue
        source = ticket.get("source", "website")
        tier = SOURCE_TO_TIER.get(source, "Basic")

        try:
            _api_post(f"/tickets/{ticket_id}/predict-priority", {
                "customer_tier": tier,
                "sla_hours_remaining": 24.0,
                "repeat_contact_count": 0,
            })
            predicted += 1
        except Exception as e:
            errors += 1
            logger.warning(f"Failed to predict priority for {ticket_id}: {e}")

    context["ti"].xcom_push(key="priority_predicted", value=predicted)
    context["ti"].xcom_push(key="priority_errors", value=errors)
    logger.info(f"Predicted priority for {predicted} tickets ({errors} errors)")
    return predicted


def log_summary(**context):
    """Log pipeline run summary to MLflow."""
    import os
    import mlflow

    PROJECT_ROOT = os.path.abspath(
        os.path.dirname(__file__) + "/../../customer-support-ai"
    )
    MLFLOW_TRACKING_URI = os.path.join(PROJECT_ROOT, "mlflow")

    tickets_ingested = context["ti"].xcom_pull(
        task_ids="ingest_new_tickets", key="tickets_ingested"
    ) or 0
    classified = context["ti"].xcom_pull(
        task_ids="classify_batch", key="classified"
    ) or 0
    classify_errors = context["ti"].xcom_pull(
        task_ids="classify_batch", key="classify_errors"
    ) or 0
    priority_predicted = context["ti"].xcom_pull(
        task_ids="predict_priority_batch", key="priority_predicted"
    ) or 0
    priority_errors = context["ti"].xcom_pull(
        task_ids="predict_priority_batch", key="priority_errors"
    ) or 0

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=f"pipeline_{context['ds']}"):
        mlflow.log_metrics({
            "tickets_ingested": tickets_ingested,
            "classified": classified,
            "classify_errors": classify_errors,
            "priority_predicted": priority_predicted,
            "priority_errors": priority_errors,
        })
        mlflow.log_params({
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "execution_date": context["ds"],
        })
        logger.info(
            f"Pipeline summary — ingested={tickets_ingested}, "
            f"classified={classified}, priority_predicted={priority_predicted}"
        )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay_minutes": 5,
}

with DAG(
    dag_id="ticket_pipeline",
    default_args=default_args,
    description="Hourly ticket processing: ingest -> classify -> priority -> log",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["tickets", "pipeline"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_new_tickets",
        python_callable=ingest_new_tickets,
    )

    t_classify = PythonOperator(
        task_id="classify_batch",
        python_callable=classify_batch,
    )

    t_priority = PythonOperator(
        task_id="predict_priority_batch",
        python_callable=predict_priority_batch,
    )

    t_summary = PythonOperator(
        task_id="log_summary",
        python_callable=log_summary,
    )

    t_ingest >> t_classify >> t_priority >> t_summary
