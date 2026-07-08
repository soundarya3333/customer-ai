
import os
import json
import threading
from typing import Optional

import pandas as pd
import xgboost as xgb

from src.ml_pipeline.priority_features import extract_all_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "priority_model")
DOCKER_MODEL_DIR = "/app/models/priority_model"
MODEL_DIR = DOCKER_MODEL_DIR if os.path.isdir(DOCKER_MODEL_DIR) else DEFAULT_MODEL_DIR

CATEGORICAL_FEATURES = ["category", "customer_tier", "source_channel"]
NUMERIC_FEATURES = [
    "vader_neu", "vader_pos",
    "sim_frustrated_anchor", "sim_routine_anchor",
    "urgency_keyword_ratio", "time_pressure_ratio",
    "frustration_negation_ratio", "profanity_ratio", "exclamation_ratio",
    "question_ratio", "caps_word_ratio", "text_length_words", "text_length_chars",
    "has_context_flag", "has_imperative_flag", "has_question_flag",
    "has_polite_flag", "has_knowledge_flag", "has_basic_flag", "has_long_flag", "flag_count",
    "ticket_age_hours", "repeat_contact_count",
    "interact_age_frustration",
    "interact_age_frustrated_anchor",
    "interact_age_question",
    "interact_repeats_flags",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

VALID_CATEGORIES = [
    "Account & Authentication", "Billing & Payments", "Orders & Shipping",
    "Returns & Refunds", "Complaints & Escalations", "Customer Feedback",
    "Subscription & Account Management", "General Inquiries",
]
VALID_TIERS = ["Free", "Basic", "Pro", "Enterprise"]
VALID_CHANNELS = ["email", "website", "zendesk", "freshdesk", "salesforce"]


RULE_CATEGORY_FALLBACK = {
    "Security & Fraud": "Complaints & Escalations",
    "Compliance & Privacy Requests": "Complaints & Escalations",
    "Product Defects & Quality Issues": "Returns & Refunds",
    "Technical Support": "Account & Authentication",
    "Feature Requests": "Customer Feedback",
    "Sales & Pre-Sales Inquiries": "General Inquiries",
}


def resolve_category(category: str) -> str:
    return RULE_CATEGORY_FALLBACK.get(category, category)


class PriorityPredictor:
    """Singleton predictor — loads model lazily on first call."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        with open(os.path.join(MODEL_DIR, "label_encoder.json")) as f:
            encoder = json.load(f)
        self.labels = encoder["labels"]
        self.id2label = {int(k): v for k, v in encoder["id2label"].items()}

        
        self.model = xgb.Booster()
        self.model.load_model(os.path.join(MODEL_DIR, "model.json"))
        self._loaded = True

    def predict(self, features: dict) -> dict:
        """features must contain all FEATURES keys."""
        self._load()
        row = {feat: features[feat] for feat in FEATURES}
        X = pd.DataFrame([row])
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("category")

        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        probs = self.model.predict(dmatrix)[0]
        pred_id = int(probs.argmax())
        return {
            "priority": self.id2label[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {self.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }

    def predict_from_text_and_context(
        self,
        text: str,
        category: str,
        customer_tier: str = "Basic",
        sla_hours_remaining: float = 24.0,
        ticket_age_hours: float = 0.0,
        repeat_contact_count: int = 0,
        source_channel: str = "website",
        flags: Optional[str] = None,
    ) -> dict:
       
        self._load()
        resolved_category = resolve_category(category)
        extracted = extract_all_features(text, resolved_category, "", flags)

        features = {
            "category": resolved_category,
            "customer_tier": customer_tier,
            "source_channel": source_channel,
            "ticket_age_hours": ticket_age_hours,
            "repeat_contact_count": repeat_contact_count,
        }
        for feat in NUMERIC_FEATURES:
            if feat in extracted:
                features[feat] = extracted[feat]
        # Interaction features — derived from base features at inference time
        features["interact_age_frustration"] = ticket_age_hours * extracted.get("frustration_negation_ratio", 0.0)
        features["interact_age_frustrated_anchor"] = ticket_age_hours * extracted.get("sim_frustrated_anchor", 0.0)
        features["interact_age_question"] = ticket_age_hours * extracted.get("question_ratio", 0.0)
        features["interact_repeats_flags"] = repeat_contact_count * extracted.get("flag_count", 0)
        return self.predict(features)


predictor = PriorityPredictor()


def predict(features: dict) -> dict:
    return predictor.predict(features)


def predict_from_text_and_context(text: str, category: str, **kwargs) -> dict:
    return predictor.predict_from_text_and_context(text, category, **kwargs)
