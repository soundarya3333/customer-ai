"""
Deterministic feature extraction for priority prediction.

All features are computable from:
  - the ticket text
  - Bitext linguistic flags (if available)
  - basic ticket metadata (category, intent, customer tier, source, SLA, age, repeats)

Two lightweight ML models are loaded lazily and cached:
  - VADER (lexicon-based sentiment)
  - sentence-transformers/all-MiniLM-L6-v2 (semantic embeddings)
"""
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml_pipeline.business_impact import (
    derive_business_impact_matrix,
    get_business_impact_score,
)


_vader_analyzer = None
_embedding_model = None
_anchor_embeddings = None

ANCHOR_SENTENCES = {
    "urgent": "This is an urgent emergency and needs immediate attention",
    "frustrated": "I am very frustrated and angry about this problem",
    "routine": "I have a general question about your product",
    "critical": "Our system is down and we are losing money every hour",
}


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_anchor_embeddings():
    global _anchor_embeddings
    if _anchor_embeddings is None:
        model = _get_embedding_model()
        _anchor_embeddings = {
            k: model.encode(v, convert_to_numpy=True, normalize_embeddings=True)
            for k, v in ANCHOR_SENTENCES.items()
        }
    return _anchor_embeddings


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")
WORD_RE = re.compile(r"[A-Za-z]+")
SENTENCE_RE = re.compile(r"[.!?]+")


def clean_text(text: str) -> str:
    text = PLACEHOLDER_RE.sub("", text)
    return " ".join(text.split())



# ---------------------------------------------------------------------------
def vader_features(text: str) -> Dict[str, float]:
    analyzer = _get_vader()
    scores = analyzer.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
        "vader_pos": scores["pos"],
    }



def embedding_anchor_features(text: str) -> Dict[str, float]:
    model = _get_embedding_model()
    anchors = _get_anchor_embeddings()
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return {
        f"sim_{key}_anchor": float(np.dot(emb, anchor_vec))
        for key, anchor_vec in anchors.items()
    }
def vader_features_batch(texts: List[str]) -> pd.DataFrame:
    analyzer = _get_vader()
    rows = [analyzer.polarity_scores(t) for t in texts]
    df = pd.DataFrame(rows).rename(
        columns={"neg": "vader_neg", "neu": "vader_neu", "pos": "vader_pos", "compound": "vader_compound"}
    )
    return df[["vader_compound", "vader_neg", "vader_neu", "vader_pos"]]


def embedding_anchor_features_batch_df(texts: List[str], batch_size: int = 128, show_progress: bool = False) -> pd.DataFrame:
    model = _get_embedding_model()
    anchors = _get_anchor_embeddings()
    anchor_names = list(anchors.keys())
    anchor_matrix = np.stack([anchors[k] for k in anchor_names])
    embs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=show_progress,
    )
    sims = embs @ anchor_matrix.T
    return pd.DataFrame(sims, columns=[f"sim_{name}_anchor" for name in anchor_names])


URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "right now", "emergency", "critical",
    "furious", "unacceptable", "disgusted", "lawsuit", "lawyer", "legal action",
    "cancel my", "cancelling", "third time", "again and again", "losing money",
    "losing customers", "cannot access", "can't access", "locked out",
    "down for", "not working at all", "escalate", "manager", "refund immediately",
]

THREAT_KEYWORDS = [
    "cancel", "lawsuit", "sue", "lawyer", "legal action", "switch", "competitor",
    "refund", "chargeback", "escalate", "manager", "supervisor", "ceo",
    "terminate", "close account", "dispute",
]

TIME_PRESSURE_KEYWORDS = [
    "now", "today", "tonight", "tomorrow", "immediately", "asap",
    "deadline", "urgent", "right away", "at once", "hour", "minute",
]

FRUSTRATION_KEYWORDS = [
    "not", "no", "never", "can't", "cannot", "won't", "wouldn't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "again", "still", "yet", "finally", "ridiculous", "terrible", "awful",
]

PROFANITY_KEYWORDS = [
    "goddamn", "goddam", "damn", "dammit", "hell", "bloody", "crap",
    "stupid", "idiot", "useless", "pathetic",
]


def _ratio(text: str, keywords: list) -> float:
    lc = text.lower()
    total_words = len(WORD_RE.findall(text))
    if total_words == 0:
        return 0.0
    count = sum(1 for kw in keywords if kw in lc)
    return count / total_words


def _sentence_count(text: str) -> int:
    return max(1, len(SENTENCE_RE.findall(text)))


def text_ratio_features(text: str) -> Dict[str, float]:
    total_words = len(WORD_RE.findall(text))
    total_chars = len(text)
    sentences = _sentence_count(text)
    caps_words = sum(1 for w in WORD_RE.findall(text) if w.isupper() and len(w) > 2)
    caps_ratio = caps_words / total_words if total_words else 0.0

    return {
        "urgency_keyword_ratio": _ratio(text, URGENCY_KEYWORDS),
        "threat_language_ratio": _ratio(text, THREAT_KEYWORDS),
        "time_pressure_ratio": _ratio(text, TIME_PRESSURE_KEYWORDS),
        "frustration_negation_ratio": _ratio(text, FRUSTRATION_KEYWORDS),
        "profanity_ratio": _ratio(text, PROFANITY_KEYWORDS),
        "exclamation_ratio": text.count("!") / sentences,
        "question_ratio": text.count("?") / sentences,
        "caps_word_ratio": caps_ratio,
        "text_length_words": total_words,
        "text_length_chars": total_chars,
    }

FLAG_DEFINITIONS = {
    "has_negative_flag": "N",
    "has_emotional_flag": "E",
    "has_typo_flag": "Z",
    "has_context_flag": "C",
    "has_multi_issue_flag": "M",
    "has_imperative_flag": "I",
    "has_question_flag": "Q",
    "has_polite_flag": "P",
    "has_knowledge_flag": "K",
    "has_basic_flag": "B",
    "has_long_flag": "L",
}


def flag_features(flags: Optional[str]) -> Dict[str, int]:
    if not flags:
        return {name: 0 for name in FLAG_DEFINITIONS}
    flags_str = str(flags).upper()
    return {
        name: (1 if letter in flags_str else 0)
        for name, letter in FLAG_DEFINITIONS.items()
    }


def flag_count(flags: Optional[str]) -> int:
    return len(str(flags)) if flags else 0


def compute_priority_score(row: Dict, matrix: Dict, rng, noise_range: float = 0.03) -> float:
    impact = get_business_impact_score(row["category"], row["intent"], matrix)
    vader_neg = row.get("vader_neg", 0.0)
    vader_compound_neg = max(0.0, -row.get("vader_compound", 0.0))

    score = (
        0.25 * impact
        + 0.18 * vader_neg
        + 0.12 * vader_compound_neg
        + 0.15 * (1.0 if row.get("is_sla_breached", False) else 0.0)
        + 0.05 * row.get("has_negative_flag", 0)
        + 0.05 * row.get("has_emotional_flag", 0)
        + 0.03 * row.get("has_typo_flag", 0)
        + 0.03 * row.get("has_multi_issue_flag", 0)
        + 0.06 * min(row.get("threat_language_ratio", 0.0) * 5.0, 1.0)
        + 0.05 * row.get("sim_urgent_anchor", 0.0)
        + 0.03 * row.get("sim_critical_anchor", 0.0)
    )

    tier = row.get("customer_tier", "Basic")
    if tier in ("Enterprise", "Pro") and impact > 0.5:
        score += 0.05
    if vader_neg > 0.3 and row.get("is_sla_breached", False):
        score += 0.08

    noise = rng.uniform(-noise_range, noise_range)
    return float(np.clip(score + noise, 0.0, 1.0))


DEFAULT_QUANTILE_TARGETS = (0.55, 0.85, 0.95)


def compute_quantile_thresholds(scores, quantile_targets=DEFAULT_QUANTILE_TARGETS) -> Dict[str, float]:
    """Calibrate priority thresholds from score distribution quantiles.

    The formula's weighted sum clusters toward lower values since most tickets
    are routine. Fixed absolute cutoffs produce empty classes; quantile-based
    thresholds guarantee each class has sufficient representation.
    """
    arr = np.asarray(scores, dtype=float)
    medium_t, high_t, urgent_t = (float(np.quantile(arr, q)) for q in quantile_targets)
    return {"medium": medium_t, "high": high_t, "urgent": urgent_t}


def priority_score_to_label(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    if thresholds is None:
        thresholds = {"medium": 0.25, "high": 0.50, "urgent": 0.75}
    if score < thresholds["medium"]:
        return "Low"
    if score < thresholds["high"]:
        return "Medium"
    if score < thresholds["urgent"]:
        return "High"
    return "Urgent"


def extract_all_features(text: str, category: str, intent: str, flags: Optional[str] = None) -> Dict:
    text = clean_text(text)
    features = {"text": text, "category": category, "intent": intent}
    features.update(vader_features(text))
    features.update(embedding_anchor_features(text))
    features.update(text_ratio_features(text))
    features.update(flag_features(flags))
    features["flag_count"] = flag_count(flags)
    return features
