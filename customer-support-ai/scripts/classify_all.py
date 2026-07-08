"""Classify all tickets in the DB via the API."""
import sys, os, json, urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.database import SessionLocal
from src.database.models import Ticket

API = "http://localhost:8000"

def main():
    db = SessionLocal()
    tickets = db.query(Ticket).all()
    print(f"Classifying {len(tickets)} tickets...\n")
    for t in tickets:
        req = urllib.request.Request(f"{API}/tickets/{t.ticket_id}/classify", method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            r = json.loads(resp.read())
            print(f"  [{t.source:>10}] {t.ticket_id:<25} -> {r['category']:<40} conf={r['confidence']}")
        except Exception as e:
            print(f"  [{t.source:>10}] {t.ticket_id:<25} -> ERROR: {e}")
    db.close()
    print("\nDone. Verifying DB...")

    from sqlalchemy import func
    db2 = SessionLocal()
    rows = db2.query(TicketClassification).all()
    print(f"\nTotal classifications stored: {len(rows)}\n")
    for c, in db2.query(TicketClassification.predicted_category, func.count(TicketClassification.ticket_id)).group_by(TicketClassification.predicted_category).all():
        pass
    cats = {}
    for c in rows:
        cats[c.predicted_category] = cats.get(c.predicted_category, 0) + 1
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<45} {cnt}")
    db2.close()

from src.database.models import TicketClassification
if __name__ == "__main__":
    main()
