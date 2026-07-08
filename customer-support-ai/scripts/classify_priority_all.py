"""
Run priority prediction on all tickets in the DB via the API.

There's no real CRM/SLA-engine integration in this system yet, so
customer_tier / sla_hours_remaining / repeat_contact_count (required context
the priority model needs) don't exist per-ticket. For this bulk demo run we
derive plausible values heuristically per ticket: customer_tier from the
ticket's source channel (salesforce tickets are B2B/enterprise; gmail/website
skew consumer), and SLA/repeat-contact hints from the ticket's *existing*
mock priority label (set by whoever wrote the mock data, not by this model).
This is illustrative grounding for populating the dashboard, not a model
accuracy test — there's no independent ground truth for these 48 tickets.
"""
import sys, os, json, urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.database import SessionLocal
from src.database.models import Ticket, TicketPriority

API = "http://localhost:8000"

SOURCE_TO_TIER = {
    "salesforce": "Enterprise",
    "zendesk": "Pro",
    "freshdesk": "Pro",
    "gmail": "Basic",
    "website": "Basic",
}

# (sla_hours_remaining, repeat_contact_count) hints keyed by the ticket's
# existing mock priority label
PRIORITY_HINT = {
    "Urgent": (-3.0, 3),
    "High": (6.0, 1),
    "Medium": (24.0, 0),
    "Low": (60.0, 0),
}


def main():
    db = SessionLocal()
    tickets = db.query(Ticket).all()
    print(f"Predicting priority for {len(tickets)} tickets...\n")
    for t in tickets:
        tier = SOURCE_TO_TIER.get(t.source, "Basic")
        sla_hint, repeat_hint = PRIORITY_HINT.get(t.priority, (24.0, 0))

        body = json.dumps({
            "customer_tier": tier,
            "sla_hours_remaining": sla_hint,
            "repeat_contact_count": repeat_hint,
        }).encode()
        req = urllib.request.Request(
            f"{API}/tickets/{t.ticket_id}/predict-priority", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            r = json.loads(resp.read())
            print(f"  [{t.source:>10}] {t.ticket_id:<25} -> {r['priority']:<8} conf={r['confidence']}  (tier={tier}, sla={sla_hint}h, repeats={repeat_hint})")
        except Exception as e:
            print(f"  [{t.source:>10}] {t.ticket_id:<25} -> ERROR: {e}")
    db.close()

    print("\nDone. Verifying DB...")
    db2 = SessionLocal()
    rows = db2.query(TicketPriority).all()
    print(f"\nTotal priority predictions stored: {len(rows)}\n")
    counts = {}
    for r in rows:
        counts[r.predicted_priority] = counts.get(r.predicted_priority, 0) + 1
    for pri, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {pri:<10} {cnt}")
    db2.close()


if __name__ == "__main__":
    main()
