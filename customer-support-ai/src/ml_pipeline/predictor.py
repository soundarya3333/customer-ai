"""
Inference wrapper for the trained DistilBERT category classifier.
Loads the model + tokenizer on first use, provides predict() and predict_batch().
"""
import os
import json
import threading

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "category_classifier")
# In Docker, models are mounted at /app/models/ (not under customer-support-ai/)
DOCKER_MODEL_DIR = "/app/models/category_classifier"
MODEL_DIR = DOCKER_MODEL_DIR if os.path.isdir(DOCKER_MODEL_DIR) else DEFAULT_MODEL_DIR
MAX_LENGTH = 256

# The training dataset (Bitext) is e-commerce/account-ops only — it has no examples
# of these 6 categories, so the ML model was never trained to recognize them.
# These are handled by high-precision keyword rules instead, checked BEFORE the
# model. Order matters: more specific/rare terms first to avoid false positives
# (e.g. "fraud" before generic "issue").
RULE_CATEGORIES = [
    ("Security & Fraud", [
        "fraud", "hacked", "hack into", "unauthorized access", "unauthorized transaction",
        "someone accessed my account", "suspicious login", "phishing", "data breach",
        "identity theft", "stolen card", "account compromised",
    ]),
    ("Compliance & Privacy Requests", [
        "gdpr", "ccpa", "delete my data", "delete my personal data", "data deletion request",
        "right to be forgotten", "export my data", "data subject request", "privacy policy violation",
    ]),
    ("Sales & Pre-Sales Inquiries", [
        "before i buy", "before purchasing", "get a quote", "pricing for enterprise",
        "bulk discount", "considering purchasing", "sales rep", "talk to sales",
        "demo request", "request a demo",
    ]),
    ("Feature Requests", [
        "feature request", "please add support for", "would be great if you added",
        "can you add a feature", "it would be nice if", "suggestion for improvement",
        "you should add",
    ]),
    ("Product Defects & Quality Issues", [
        "arrived broken", "arrived damaged", "defective", "faulty item", "item is broken",
        "poor quality", "stopped working after", "fell apart", "manufacturing defect",
    ]),
    ("Technical Support", [
        "app crashes", "app keeps crashing", "error message", "website is down",
        "won't load", "not loading", "500 error", "404 error", "bug in the app",
        "glitch", "login page is broken",
    ]),
]


def _rule_based_category(text: str) -> str | None:
    lc = text.lower()
    for category, phrases in RULE_CATEGORIES:
        if any(phrase in lc for phrase in phrases):
            return category
    return None


class CategoryPredictor:
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
        self.label2id = encoder["label2id"]
        self.id2label = {int(k): v for k, v in encoder["id2label"].items()}

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._loaded = True

    def predict(self, text: str) -> dict:
        self._load()

        rule_category = _rule_based_category(text)
        if rule_category is not None:
            return {"category": rule_category, "confidence": 0.99}

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        return {
            "category": self.id2label[pred_id],
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        self._load()
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


predictor = CategoryPredictor()


def predict(text: str) -> dict:
    return predictor.predict(text)


def predict_batch(texts: list[str]) -> list[dict]:
    return predictor.predict_batch(texts)