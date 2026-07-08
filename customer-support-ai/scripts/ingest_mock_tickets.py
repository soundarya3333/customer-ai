"""
Multi-source ingestion simulation script.
Generates mock tickets for all 5 sources (Zendesk, Freshdesk, Salesforce,
Gmail, Website/chat), normalizes them, and inserts into PostgreSQL.
"""
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from sqlalchemy import func

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ingestion.normalizer import Normalizer
from src.database.database import SessionLocal, Base, engine
from src.database.models import Ticket


ZENDESK_TEMPLATE = {
    "id": 87214,
    "requester_id": 441098,
    "subject": "placeholder",
    "description": "placeholder",
    "priority": "normal",
    "status": "open",
    "created_at": "2026-07-01T09:00:00Z",
    "tags": ["support"],
    "url": "https://company.zendesk.com/api/v2/tickets/87214.json",
    "attachments": [],
}

FRESHDESK_TEMPLATE = {
    "id": 901235,
    "requester_id": 882012,
    "subject": "placeholder",
    "description": "placeholder",
    "priority": 2,
    "status": 2,
    "created_at": "2026-07-01T09:00:00Z",
    "attachments": [],
}

SF_TEMPLATE = {
    "Id": "5008W00001G1TMPL02",
    "ContactId": "0038W00001Habcdef",
    "ContactEmail": "customer@company.com",
    "Subject": "placeholder",
    "Description": "placeholder",
    "Priority": "Medium",
    "Status": "New",
    "CreatedDate": "2026-07-01T09:00:00.000+0000",
    "CaseNumber": "00001000",
    "Origin": "Web",
    "Reason": "General Inquiry",
    "Type": "Problem",
}

GMAIL_TEMPLATE = {
    "id": "mock_gmail_100",
    "threadId": "mock_thread_100",
    "labelIds": ["INBOX", "UNREAD"],
    "from": "Customer <customer@example.com>",
    "subject": "placeholder",
    "body": "placeholder",
    "date": "2026-07-01T09:00:00Z",
    "attachments": [],
}

WEBSITE_TEMPLATE = {
    "ticket_id": "web_mock_001",
    "customer_id": "web_customer_001",
    "instruction": "placeholder",
    "category": "GENERAL",
    "intent": "general_inquiry",
    "flags": "B",
    "response": "Thank you for reaching out. A support agent will assist you shortly.",
    "timestamp": "2026-07-01T09:00:00Z",
    "priority": "Medium",
    "status": "Open",
}


ZENDESK_TICKETS = [
    {"id": 87201, "subject": "Package stuck at local hub for 5 days", "description": "My order #ORD-8821 has been sitting at the local distribution hub for 5 days with no update. Expected delivery was 2 days ago.", "priority": "high", "status": "open", "tags": ["delivery", "delayed"]},
    {"id": 87202, "subject": "Wrong item received", "description": "I ordered a Bluetooth speaker (SKU BT-200) but I received a phone charger instead. I need the correct item shipped.", "priority": "high", "status": "open", "tags": ["delivery", "wrong_item"]},
    {"id": 87203, "subject": "Website crashes after login on Chrome", "description": "Every time I log in using Chrome on Windows 11, the page goes blank white. Works fine on Firefox. Started after latest site update.", "priority": "high", "status": "open", "tags": ["technical", "browser", "bug"]},
    {"id": 87204, "subject": "Promo code SAVE30 not applying at checkout", "description": "I received the SAVE30 code in your newsletter. It says valid until end of month but checkout shows 'invalid code'. My cart is over the $75 minimum.", "priority": "medium", "status": "open", "tags": ["billing", "promo"]},
    {"id": 87205, "subject": "Account locked after password reset", "description": "I reset my password as instructed, received the confirmation email, but now I'm locked out. Cannot log in or reset again. Urgent — I need access for a meeting in 1 hour.", "priority": "urgent", "status": "open", "tags": ["account", "login"]},
    {"id": 87206, "subject": "Refund still not received after 2 weeks", "description": "I returned my order #ORD-7742 and the warehouse confirmed receipt 2 weeks ago. Customer service said refund would take 5-7 days but nothing yet.", "priority": "high", "status": "open", "tags": ["refund", "billing"]},
    {"id": 87207, "subject": "Product arrived with cracked screen", "description": "The tablet I ordered arrived with a cracked display. The outer box was undamaged so it must have been packed already broken. Photos attached.", "priority": "high", "status": "pending", "tags": ["product_defect", "damaged"]},
    {"id": 87208, "subject": "Duplicate charge on monthly subscription", "description": "I see two identical $29.99 charges from your company on my credit card this month. Please refund one and fix the billing system.", "priority": "urgent", "status": "open", "tags": ["billing", "duplicate"]},
    {"id": 87209, "subject": "Cannot update shipping address for pending order", "description": "My order #ORD-9120 hasn't shipped yet but the Edit Address button in My Orders is greyed out. I moved this week and need it sent to the new place.", "priority": "medium", "status": "open", "tags": ["account", "shipping"]},
    {"id": 87210, "subject": "Mobile app freezes during payment step", "description": "The Android app works fine until I hit Pay Now — then it freezes completely. I have to force close and restart. Happened 4 times now.", "priority": "medium", "status": "open", "tags": ["technical", "mobile", "bug"]},
]

FRESHDESK_TICKETS = [
    {"id": 90001, "subject": "Annual plan auto-renewed despite cancellation", "description": "<div>I cancelled my annual plan on June 15th. Got charged $199 today. This is the second time this year. Please refund immediately.</div>", "priority": 4, "status": 2},
    {"id": 90002, "subject": "Loyalty discount not applied to renewal", "description": "<div>Agent Monica confirmed I qualify for the 20% loyalty discount on renewal but invoice shows full $149. Please correct before payment date.</div>", "priority": 3, "status": 2},
    {"id": 90003, "subject": "Account terminated immediately after upgrade payment", "description": "<div>Upgraded from Basic to Pro plan, payment went through, but my account now shows 'Suspended'. I rely on this for my business operations.</div>", "priority": 4, "status": 2},
    {"id": 90004, "subject": "Invoice download page broken since UI update", "description": "<div>Since the new dashboard UI rolled out, clicking any invoice gives a 404 error. I need my Q2 invoices for tax filing by Friday.</div>", "priority": 2, "status": 3},
    {"id": 90005, "subject": "Package marked delivered but never received", "description": "<div>Tracking shows delivered yesterday at 2pm but I was home all day and no package arrived. Checked with neighbors too. Value is $340.</div>", "priority": 4, "status": 2},
    {"id": 90006, "subject": "Device stopped working after firmware v3.2 update", "description": "<div>Updated my router to firmware v3.2 as prompted. Now it reboots every 10 minutes. Tried factory reset 3 times, same issue. Need replacement unit.</div>", "priority": 3, "status": 2},
    {"id": 90007, "subject": "Refund amount is incorrect", "description": "<div>I was promised a full $89 refund but only received $42.50. Support ticket #44512 confirms the full amount was approved. Please send the remaining $46.50.</div>", "priority": 3, "status": 2},
    {"id": 90008, "subject": "Password reset emails never arrive", "description": "<div>Tried password reset 6 times over 3 days. No emails received, not in spam either. Tried two different email addresses linked to my account.</div>", "priority": 2, "status": 2},
    {"id": 90009, "subject": "Battery swelling and overheating on laptop", "description": "<div>My 6-month-old laptop battery is visibly bulging and the unit gets too hot to touch near the trackpad. This is a safety hazard. Photos included.</div>", "priority": 4, "status": 2},
    {"id": 90010, "subject": "Cannot add team members to business account", "description": "<div>When I go to Team > Add Member and enter their email, the invite button stays greyed out. No error message. My plan includes up to 10 seats.</div>", "priority": 2, "status": 3},
]

SALESFORCE_TICKETS = [
    {"Subject": "REST API returning 500 errors since 2am deployment", "Description": "Our integration pipeline has been failing with HTTP 500 on every POST to the Cases API since the 2am production deployment. 340 orders are stuck in the queue.", "Priority": "Urgent", "Status": "New", "Reason": "Integration Failure", "Type": "Problem"},
    {"Subject": "Bulk CSV upload times out for files over 100 records", "Description": "The bulk upload tool used to handle 5000-row CSVs. Now anything above 100 rows spins for 2 minutes then times out. Blocking our quarterly SKU refresh.", "Priority": "High", "Status": "New", "Reason": "Performance", "Type": "Problem"},
    {"Subject": "Admin getting 403 on Reports dashboard", "Description": "I have full System Admin profile but the Reports tab shows '403 Forbidden'. My team of 12 analysts can't access any dashboards. Blocking daily standup.", "Priority": "High", "Status": "New", "Reason": "Access Issue", "Type": "Problem"},
    {"Subject": "Export missing Region and Priority_Reason fields", "Description": "Exported 2000 cases to CSV for executive review. Two custom fields (Priority_Reason, Region) are blank in the export even though they display fine in the UI.", "Priority": "Medium", "Status": "New", "Reason": "Data Issue", "Type": "Bug"},
    {"Subject": "Enterprise order shipped to wrong warehouse address", "Description": "PO-88421 ($47K hardware order) was shipped to our old warehouse. Customer is demanding delivery to new address by Friday or they'll cancel the deal.", "Priority": "Urgent", "Status": "New", "Reason": "Logistics", "Type": "Problem"},
    {"Subject": "Refund request for defective server rack units", "Description": "Customer ACME Corp reports 4 out of 12 server rack units are dead on arrival. Requesting full refund of $12,400 plus return shipping label.", "Priority": "High", "Status": "New", "Reason": "Product Defect", "Type": "Problem"},
    {"Subject": "Stripe payment gateway declining all cards since noon", "Description": "Our Stripe integration started rejecting every transaction at 12:15 UTC. 0% success rate. Customers are calling support in droves. Estimated $23K in lost sales so far.", "Priority": "Urgent", "Status": "New", "Reason": "Payment Failure", "Type": "Incident"},
    {"Subject": "SSO login broken for all EU region employees", "Description": "Since the Azure AD certificate rotated this morning, none of our 200+ EU employees can log in via SSO. US region works fine. Affects production access.", "Priority": "Urgent", "Status": "New", "Reason": "Authentication", "Type": "Incident"},
    {"Subject": "Product recall alert not showing on customer portal", "Description": "Safety recall RC-2026-03 for the power supply units should be pinned on the customer portal homepage. It's not visible. Legal requires this within 24 hours.", "Priority": "High", "Status": "New", "Reason": "Compliance", "Type": "Problem"},
    {"Subject": "Invoice totals mismatch between quote and final bill", "Description": "Multiple customers reporting final invoice is 8.5% higher than the approved quote. Appears to be a tax calculation error in the billing module.", "Priority": "Medium", "Status": "New", "Reason": "Billing Error", "Type": "Bug"},
]

GMAIL_TICKETS = [
    {"id": "gm_101", "subject": "Payment failed but money was debited", "body": "Hi Support, I tried paying for the premium plan. The payment showed 'failed' but $49 was deducted from my bank account. Transaction ID TXN-882341. Please help.", "from": "Raj Patel <raj.patel@gmail.com>", "date": "2026-06-30T14:20:00Z"},
    {"id": "gm_102", "subject": "Forgot password, recovery options not working", "body": "Hello, I forgot my password and the SMS recovery code never arrives. I tried voice call too but it disconnects. My email on file is rachel.chen@outlook.com.", "from": "Rachel Chen <rachel.chen@outlook.com>", "date": "2026-06-30T15:45:00Z"},
    {"id": "gm_103", "subject": "Returned item but no refund confirmation", "body": "Returned order #ORD-6631 via your prepaid label 10 days ago. UPS tracking shows delivered to your returns center. When will I get my $156 refund?", "from": "Marcus Williams <marcus.w@gmail.com>", "date": "2026-06-30T09:15:00Z"},
    {"id": "gm_104", "subject": "App keeps crashing after latest iOS update", "body": "Since updating to iOS 19.2, your app crashes on launch every single time. I've reinstalled twice. iPhone 15 Pro. I need the app to manage my subscriptions.", "from": "Aisha Khan <aisha.k@icloud.com>", "date": "2026-06-30T11:00:00Z"},
    {"id": "gm_105", "subject": "Delivery was left in the rain, everything ruined", "body": "The delivery driver left my $280 order on the doorstep in pouring rain with no bag or cover. All items are water damaged. I have Ring doorbell footage as proof.", "from": "David Okafor <david.okafor@gmail.com>", "date": "2026-07-01T08:30:00Z"},
    {"id": "gm_106", "subject": "Unauthorized charge on my account", "body": "I just noticed a $179.99 charge from your company on my card. I haven't made any purchase in 3 months. Card ending in 4421. Please reverse this charge immediately.", "from": "Emily Strauss <emily.s@protonmail.com>", "date": "2026-07-01T07:00:00Z"},
    {"id": "gm_107", "subject": "Product arrived with missing parts", "body": "Ordered the desk assembly kit #DK-400. The instruction manual is there but the screw pack and allen key are missing from the box. I'm halfway through assembly and stuck.", "from": "Jake Thompson <jake.t@gmail.com>", "date": "2026-07-01T06:15:00Z"},
    {"id": "gm_108", "subject": "Need to cancel subscription before renewal", "body": "I need to cancel my monthly subscription before it renews on July 5th. I can't find the cancel button anywhere in the account settings. Please confirm cancellation.", "from": "Sofia Garcia <sofia.garcia@yahoo.com>", "date": "2026-07-01T10:00:00Z"},
]

WEBSITE_TICKETS = [
    {"ticket_id": "web_201", "instruction": "I need to update my billing address but the save button does nothing when I click it. Tried on Chrome and Safari. Please fix or update it manually.", "category": "ACCOUNT", "intent": "edit_billing_info", "priority": "Medium", "status": "Open"},
    {"ticket_id": "web_202", "instruction": "My cancellation request from last week still shows as pending. I was told it would be processed within 48 hours. I don't want to be charged again.", "category": "CANCELLATION", "intent": "cancel_account", "priority": "High", "status": "Pending"},
    {"ticket_id": "web_203", "instruction": "I requested a refund on June 20th for a defective product but the status still says Under Review. Customer protection law says refunds must be processed within 14 days.", "category": "REFUND", "intent": "refund_status", "priority": "High", "status": "Open"},
    {"ticket_id": "web_204", "instruction": "Your payment page reloads every time I enter my card details. I tried 3 different cards and 2 browsers. Same issue. How am I supposed to pay?", "category": "PAYMENT", "intent": "payment_failed", "priority": "Urgent", "status": "Open"},
    {"ticket_id": "web_205", "instruction": "The search function on your knowledge base returns completely unrelated articles. Searching for 'return policy' shows articles about password reset. This is useless.", "category": "TECHNICAL", "intent": "search_broken", "priority": "Medium", "status": "Open"},
    {"ticket_id": "web_206", "instruction": "My order says delivered but I never got it. The tracking number goes to a 404 page. I'm concerned this was stolen or delivered to the wrong address.", "category": "DELIVERY", "intent": "missing_package", "priority": "Urgent", "status": "Open"},
    {"ticket_id": "web_207", "instruction": "The wireless earbuds I bought last month stopped holding charge. Right earbud dies after 10 minutes. Left one is fine. This is clearly a manufacturing defect.", "category": "PRODUCT_DEFECT", "intent": "defective_product", "priority": "High", "status": "Open"},
    {"ticket_id": "web_208", "instruction": "I got charged $99.99 for a premium plan I never signed up for. I only used the free trial. This feels fraudulent. Please cancel and refund.", "category": "BILLING", "intent": "unauthorized_charge", "priority": "Urgent", "status": "Open"},
    {"ticket_id": "web_209", "instruction": "How do I change my subscription from monthly to annual billing? The help article says go to Settings but there is no billing section in my settings page.", "category": "ACCOUNT", "intent": "change_plan", "priority": "Low", "status": "Open"},
    {"ticket_id": "web_210", "instruction": "Your chatbot keeps looping the same 3 auto-replies no matter what I type. It's impossible to reach a human agent. Attached is a screenshot after 15 minutes of trying.", "category": "TECHNICAL", "intent": "chatbot_broken", "priority": "Medium", "status": "Open"},
]

ALL_SOURCES = {
    "zendesk": (ZENDESK_TEMPLATE, ZENDESK_TICKETS, "id"),
    "freshdesk": (FRESHDESK_TEMPLATE, FRESHDESK_TICKETS, "id"),
    "salesforce": (SF_TEMPLATE, SALESFORCE_TICKETS, "Id"),
    "gmail": (GMAIL_TEMPLATE, GMAIL_TICKETS, "id"),
    "website": (WEBSITE_TEMPLATE, WEBSITE_TICKETS, "ticket_id"),
}


def generate_variants(template: Dict[str, Any], variants: List[Dict[str, Any]], id_field: str) -> List[Dict[str, Any]]:
    base_time = datetime(2026, 6, 30, 9, 0, 0, tzinfo=timezone.utc)
    generated = []

    for i, variant in enumerate(variants):
        record = dict(template)
        record.update(variant)

        ts = base_time + timedelta(hours=i * 2, minutes=i * 15)
        if "created_at" in record:
            record["created_at"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "CreatedDate" in record:
            record["CreatedDate"] = ts.strftime("%Y-%m-%dT%H:%M:%S.000+0000")
        if "date" in record:
            record["date"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "timestamp" in record:
            record["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        if id_field == "Id":
            record["Id"] = f"5008W00001G1{i:06d}"
            if "CaseNumber" in record:
                record["CaseNumber"] = f"00001{i:05d}"

        generated.append(record)

    return generated


def main():
    print("=" * 70)
    print("  MULTI-SOURCE INGESTION SIMULATION")
    print("  Sources: zendesk | freshdesk | salesforce | gmail | website")
    print("=" * 70)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    total_inserted = 0

    for source, (template, tickets, id_field) in ALL_SOURCES.items():
        print(f"\n[{source.upper()}] Generating {len(tickets)} mock tickets...")

        raw_tickets = generate_variants(template, tickets, id_field)
        normalize_method = getattr(Normalizer, f"normalize_{source}")
        inserted = 0

        for raw in raw_tickets:
            normalized = normalize_method(raw)

            existing = db.query(Ticket).filter_by(ticket_id=normalized["ticket_id"]).first()
            if existing:
                print(f"  [SKIP] {normalized['ticket_id']} already exists")
                continue

            normalized["ticket_metadata"] = normalized.pop("metadata", {})
            ticket = Ticket(**normalized)
            db.add(ticket)
            inserted += 1
            print(f"  [OK] {normalized['ticket_id']} | {normalized['subject'][:55]}... | priority={normalized['priority']}")

        db.commit()
        total_inserted += inserted
        print(f"  -> Inserted {inserted} {source} tickets")

    print(f"\n{'=' * 70}")
    print(f"  TOTAL INSERTED: {total_inserted} tickets across {len(ALL_SOURCES)} sources")
    print(f"{'=' * 70}")

    summary = db.query(Ticket.source, func.count(Ticket.ticket_id)).group_by(Ticket.source).all()
    db.close()

    print("\n  Database summary:")
    for src, cnt in summary:
        print(f"    {src}: {cnt}")

    print("\n  Priority breakdown:")
    psummary = db.query(Ticket.priority, func.count(Ticket.ticket_id)).group_by(Ticket.priority).all()
    if psummary:
        for pri, cnt in psummary:
            print(f"    {pri}: {cnt}")


if __name__ == "__main__":
    main()
