"""
Demo runner — shows your mentor the full pipeline end-to-end.
Run this from the project root: python customer-support-ai/scripts/demo.py
"""
import sys
import os
import time
import json
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sqlalchemy import func, text
from sqlalchemy.orm import Session
from src.database.database import engine, SessionLocal
from src.database.models import Ticket


def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def run(cmd: list) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception:
        return False


def main():
    print("*" * 65)
    print("*" + " " * 63 + "*")
    print("*   CUSTOMER SUPPORT AI — END-TO-END DEMO" + " " * 23 + "*")
    print("*" + " " * 63 + "*")
    print("*" * 65)

    # 1 — Docker Status
    section("1. INFRASTRUCTURE — DOCKER CONTAINERS")
    r = subprocess.run(["docker", "compose", "ps", "--format", "table {{.Name}}\t{{.Service}}\t{{.Status}}\t{{.Ports}}"],
                       capture_output=True, text=True, cwd=os.path.dirname(__file__), timeout=10)
    print(r.stdout.strip())

    # 2 — API Health
    section("2. FASTAPI — HEALTH CHECK")
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        data = json.loads(resp.read())
        print(f"  Status    : {data['status']}")
        print(f"  Uptime    : {time.strftime('%H:%M:%S', time.localtime(data['timestamp']))}")
    except Exception as e:
        print(f"  [FAIL] API not reachable: {e}")

    # 3 — Metrics
    section("3. PROMETHEUS METRICS ENDPOINT (snippet)")
    try:
        resp = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5)
        lines = resp.read().decode().split("\n")
        for line in lines[:8]:
            if line and not line.startswith("#"):
                print(f"  {line}")
        print(f"  ... ({len(lines)} lines total)")
    except Exception:
        print("  Metrics endpoint unavailable (app may need restart)")

    # 4 — Database Summary
    section("4. POSTGRESQL — INGESTED TICKETS")
    db = SessionLocal()
    total = db.query(func.count(Ticket.ticket_id)).scalar()
    print(f"  Total tickets in DB: {total}\n")
    print(f"  {'SOURCE':<15} {'COUNT':<8}")
    print(f"  {'-'*22}")
    for src, cnt in db.query(Ticket.source, func.count(Ticket.ticket_id)).group_by(Ticket.source).all():
        print(f"  {src:<15} {cnt:<8}")
    print(f"\n  {'PRIORITY':<12} {'COUNT':<8}")
    print(f"  {'-'*20}")
    for pri, cnt in sorted(db.query(Ticket.priority, func.count(Ticket.ticket_id)).group_by(Ticket.priority).all(), key=lambda x: x[1], reverse=True):
        bar = "#" * cnt
        print(f"  {pri:<12} {str(cnt):<8} {bar}")

    # 5 — Sample tickets
    section("5. SAMPLE TICKETS (1 per source)")
    sources = db.query(Ticket.source).distinct().all()
    for (src,) in sources:
        t = db.query(Ticket).filter_by(source=src).first()
        print(f"\n  [{src.upper()}]")
        print(f"    ID       : {t.ticket_id}")
        print(f"    Subject  : {t.subject}")
        print(f"    Priority : {t.priority}")
        print(f"    Status   : {t.status}")
        print(f"    Message  : {t.customer_message[:90]}...")

    db.close()

    # 6 — Pipeline architecture
    section("6. PIPELINE ARCHITECTURE")
    print("""
  [Zendesk] --+
  [Freshdesk]--+
  [Salesforce]-+  +---------------+   +----------+   +----------+
  [Gmail] -----+->|  Normalizer   |-->|PostgreSQL|<--| FastAPI  |
  [Website] ---+  | (Pydantic)    |   | (Docker) |   | :8000    |
                  +---------------+   +----------+   +----------+
                                                         |
                                                  +------+------+
                                                  | Prometheus  |
                                                  |  :8000      |
                                                  |  /metrics   |
                                                  +-------------+
  """)

    # 7 — Next steps
    section("7. WHAT'S NEXT")
    print("""
  Day 3  -> Train ticket classification model (XGBoost on HF dataset)
  Day 4  -> Sentiment analysis + RAG pipeline
  Day 5  -> Prometheus + Grafana dashboards
  Day 6  -> Fine-tuned LLM response generation
  """)

    print("=" * 65)
    print("  DEMO COMPLETE")

if __name__ == "__main__":
    main()
