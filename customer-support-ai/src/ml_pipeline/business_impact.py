"""
Business Impact Matrix for ticket priority scoring.

Two transparent methodologies produce every score from auditable inputs:

  1. Category base weights: each category is scored on 5 risk dimensions
     (churn, financial, access, urgency, legal). A weighted sum produces a
     raw risk score normalized so the highest-risk category maps to 0.30.

  2. Intent action weights: each intent is classified into one of 7 action
     types (failure, cancellation, change, setup, tracking, inquiry,
     browsing). Each type has a base severity; each intent gets a modifier
     that adjusts within its type.

Final severity = category_base + intent_action, clamped to [0, 1].
"""
from typing import Dict, Tuple



DIMENSION_WEIGHTS: Dict[str, float] = {
    "churn_risk":      0.30,
    "financial_risk":  0.25,
    "access_risk":     0.20,
    "urgency":         0.15,
    "legal_risk":      0.10,
}

CATEGORY_RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "Complaints & Escalations":           {"churn_risk": 1.0, "financial_risk": 0.6, "access_risk": 0.2, "urgency": 0.6, "legal_risk": 0.4},
    "Account & Authentication":           {"churn_risk": 0.4, "financial_risk": 0.2, "access_risk": 0.8, "urgency": 0.4, "legal_risk": 0.1},
    "Orders & Shipping":                  {"churn_risk": 0.3, "financial_risk": 0.7, "access_risk": 0.1, "urgency": 0.4, "legal_risk": 0.1},
    "Billing & Payments":                 {"churn_risk": 0.3, "financial_risk": 0.6, "access_risk": 0.2, "urgency": 0.3, "legal_risk": 0.2},
    "Returns & Refunds":                  {"churn_risk": 0.3, "financial_risk": 0.6, "access_risk": 0.1, "urgency": 0.3, "legal_risk": 0.1},
    "General Inquiries":                  {"churn_risk": 0.3, "financial_risk": 0.2, "access_risk": 0.3, "urgency": 0.3, "legal_risk": 0.0},
    "Subscription & Account Management":  {"churn_risk": 0.2, "financial_risk": 0.2, "access_risk": 0.3, "urgency": 0.2, "legal_risk": 0.0},
    "Customer Feedback":                  {"churn_risk": 0.4, "financial_risk": 0.0, "access_risk": 0.0, "urgency": 0.3, "legal_risk": 0.0},
}

# The highest-risk category normalizes to MAX_CATEGORY_BASE.
MAX_CATEGORY_BASE = 0.30


def derive_category_base_weights() -> Dict[str, float]:
    """Compute each category's base severity from its risk profile.

    raw_score = sum(dimension_weight * dimension_score)
    base_weight = MAX_CATEGORY_BASE * (raw_score / max_raw_score)
    """
    raw_scores: Dict[str, float] = {}
    for cat, profile in CATEGORY_RISK_PROFILES.items():
        raw = sum(DIMENSION_WEIGHTS[dim] * profile.get(dim, 0.0) for dim in DIMENSION_WEIGHTS)
        raw_scores[cat] = raw

    max_raw = max(raw_scores.values()) if raw_scores else 1.0
    return {cat: round(MAX_CATEGORY_BASE * (raw / max_raw), 4) for cat, raw in raw_scores.items()}




ACTION_TYPE_BASE: Dict[str, float] = {
    "failure":      0.35,
    "cancellation": 0.27,
    "change":       0.21,
    "setup":        0.18,
    "tracking":     0.14,
    "inquiry":      0.08,
    "browsing":     0.02,
}

INTENT_ACTION_PROFILE: Dict[str, Tuple[str, float]] = {
    "payment_issue":              ("failure", +0.04),
    "get_refund":                 ("failure", -0.01),
    "registration_problems":      ("failure", -0.04),
    "cancel_order":               ("cancellation", +0.01),
    "delete_account":             ("cancellation", -0.02),
    "change_order":               ("change", +0.01),
    "switch_account":             ("change", 0.00),
    "change_shipping_address":    ("change", -0.01),
    "recover_password":           ("change", -0.01),
    "set_up_shipping_address":    ("setup", +0.08),
    "get_invoice":                ("setup", +0.07),
    "contact_human_agent":        ("setup", +0.07),
    "place_order":                ("setup", -0.03),
    "newsletter_subscription":    ("setup", -0.03),
    "create_account":             ("setup", -0.08),
    "track_refund":               ("tracking", +0.09),
    "check_cancellation_fee":     ("tracking", +0.05),
    "track_order":                ("tracking", -0.02),
    "delivery_period":            ("tracking", -0.02),
    "contact_customer_service":   ("inquiry", +0.04),
    "check_refund_policy":        ("inquiry", +0.04),
    "check_invoice":              ("inquiry", +0.02),
    "check_payment_methods":      ("inquiry", -0.01),
    "review":                     ("inquiry", +0.07),
    "edit_account":               ("inquiry", +0.03),
    "delivery_options":           ("browsing", -0.05),
}


def derive_intent_action_weights() -> Dict[str, float]:
    """Compute each intent's action weight from its type base + modifier."""
    weights: Dict[str, float] = {}
    for intent, (action_type, modifier) in INTENT_ACTION_PROFILE.items():
        base = ACTION_TYPE_BASE[action_type]
        weights[intent] = round(base + modifier, 4)
    return weights




def derive_business_impact_matrix() -> Dict:
    """Build the (category, intent) -> severity lookup from derived weights.

    Each score = category_base + intent_action, clamped to [0, 1].
    The category fallback is the base weight itself; the global fallback
    is the average of all category bases.
    """
    category_bases = derive_category_base_weights()
    intent_actions = derive_intent_action_weights()

    matrix: Dict[str, float] = {}
    for cat, base in category_bases.items():
        for intent, action in intent_actions.items():
            matrix[f"{cat}||{intent}"] = float(max(0.0, min(1.0, base + action)))

    global_fallback = float(sum(category_bases.values()) / len(category_bases)) if category_bases else 0.0

    return {
        "matrix": matrix,
        "category_fallback": category_bases,
        "global_fallback": global_fallback,
    }


def get_business_impact_score(category: str, intent: str, matrix: Dict) -> float:
    """Three-tier lookup: exact (category, intent) -> category fallback -> global."""
    key = f"{category}||{intent}"
    if key in matrix["matrix"]:
        return matrix["matrix"][key]
    if category in matrix["category_fallback"]:
        return matrix["category_fallback"][category]
    return matrix["global_fallback"]
