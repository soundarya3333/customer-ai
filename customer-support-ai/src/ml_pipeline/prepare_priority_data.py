"""
Prepare training data for priority prediction.

Loads raw Bitext data directly (not the category classifier's CSV) to preserve
the `flags` column. Applies VADER sentiment, MiniLM anchor similarities, text
ratio features, Bitext linguistic flags, and synthetic tabular context (tier,
SLA, channel, repeat contacts). Labels are generated from a deterministic
formula (compute_priority_score) to avoid LLM dependency.
"""
import os
import json
import random
import re

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.ml_pipeline.prepare_data import INTENT_TO_BUSINESS_CATEGORY
from src.ml_pipeline.priority_features import (
    clean_text, vader_features_batch, embedding_anchor_features_batch_df,
    text_ratio_features, flag_features, flag_count,
    derive_business_impact_matrix, get_business_impact_score,
    compute_priority_score, priority_score_to_label, compute_quantile_thresholds,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "priority_model")
RANDOM_STATE = 42

PRIORITY_LABELS = ["Low", "Medium", "High", "Urgent"]

CUSTOMER_TIERS = ["Free", "Basic", "Pro", "Enterprise"]
TIER_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # realistic: most customers are free/basic

SOURCE_CHANNELS = ["email", "website", "zendesk", "freshdesk", "salesforce"]
SOURCE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """Collapses case/punctuation so exact rephrasings dedup, while distinct
    typo variants (Bitext's Z-flagged rows) stay distinct — dedup only kills
    true near-duplicates, not the deliberate typo signal."""
    lowered = text.lower()
    cleaned = _ALNUM_RE.sub("", lowered)
    return _WS_RE.sub(" ", cleaned).strip()


def generate_engineered_features(n: int, rng: random.Random) -> pd.DataFrame:
    """Synthetic tabular features a real ticketing system would actually have
    (customer tier, SLA clock, ticket age, repeat-contact count, channel)."""
    tiers = rng.choices(CUSTOMER_TIERS, weights=TIER_WEIGHTS, k=n)
    channels = rng.choices(SOURCE_CHANNELS, weights=SOURCE_WEIGHTS, k=n)

    sla_hours_remaining = []
    ticket_age_hours = []
    repeat_contact_count = []
    for tier in tiers:
        base_sla = {"Free": 72, "Basic": 48, "Pro": 24, "Enterprise": 8}[tier]
        age = max(0.0, rng.gauss(base_sla * 0.6, base_sla * 0.5))
        remaining = base_sla - age
        sla_hours_remaining.append(round(remaining, 1))
        ticket_age_hours.append(round(age, 1))
        # higher-tier customers get followed up with more, so repeats skew that way
        repeat_contact_count.append(np.random.poisson(0.6 if tier in ("Enterprise", "Pro") else 0.2))

    return pd.DataFrame({
        "customer_tier": tiers,
        "source_channel": channels,
        "sla_hours_remaining": sla_hours_remaining,
        "ticket_age_hours": ticket_age_hours,
        "repeat_contact_count": repeat_contact_count,
    })


def main():
    print("=" * 65)
    print("  DATA PREPARATION — Priority Prediction (full deterministic rebuild)")
    print("=" * 65)

    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1] Loading raw Bitext HF dataset (need the `flags` column)...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    print(f"    Loaded {len(dataset):,} examples")

    print("\n[2] Mapping (category, intent) -> business category, cleaning text...")
    rows = []
    for ex in dataset:
        raw_text = (ex.get("instruction") or "").strip()
        intent = (ex.get("intent") or "").strip()
        flags = (ex.get("flags") or "").strip()
        if not raw_text or not intent:
            continue
        business_cat = INTENT_TO_BUSINESS_CATEGORY.get(intent)
        if business_cat is None:
            continue
        cleaned = clean_text(raw_text)
        if len(cleaned) < 5:
            continue
        rows.append({
            "text": cleaned, "category": business_cat, "intent": intent,
            "flags": flags, "norm": normalize_for_dedup(raw_text),
        })
    df = pd.DataFrame(rows)
    print(f"    Kept {len(df):,} rows (dropped {len(dataset) - len(df):,} unmapped/empty)")

    print("\n[3] Deduplicating near-identical text...")
    before = len(df)
    df = df.drop_duplicates(subset=["norm"]).drop(columns=["norm"]).reset_index(drop=True)
    print(f"    Removed {before - len(df):,} duplicates -> {len(df):,} rows")

    print("\n[4] Generating synthetic tabular features (tier, SLA, channel, repeat contacts)...")
    rng = random.Random(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    engineered = generate_engineered_features(len(df), rng)
    df = pd.concat([df, engineered], axis=1)
    df["is_sla_breached"] = (df["sla_hours_remaining"] < 0).astype(int)

    print("\n[5] VADER sentiment (all rows, batched)...")
    vader_df = vader_features_batch(df["text"].tolist())
    df = pd.concat([df, vader_df], axis=1)

    print("\n[6] Sentence embeddings + anchor similarities (all-MiniLM-L6-v2, batched)...")
    emb_df = embedding_anchor_features_batch_df(df["text"].tolist(), show_progress=True)
    df = pd.concat([df, emb_df], axis=1)

    print("\n[7] Text ratio features (urgency/threat/time-pressure/etc keyword ratios)...")
    ratio_df = pd.DataFrame([text_ratio_features(t) for t in df["text"]])
    df = pd.concat([df, ratio_df], axis=1)

    print("\n[8] Bitext linguistic flag features...")
    flag_df = pd.DataFrame([flag_features(f) for f in df["flags"]])
    df = pd.concat([df, flag_df], axis=1)
    df["flag_count"] = df["flags"].apply(flag_count)

    print("\n[9] Building rule-based business-impact matrix (no LLM, no calibration data)...")
    matrix = derive_business_impact_matrix()
    df["business_impact_score"] = [
        get_business_impact_score(cat, intent, matrix)
        for cat, intent in zip(df["category"], df["intent"])
    ]
    matrix_path = os.path.join(MODEL_DIR, "business_impact_matrix.json")
    with open(matrix_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"    Saved rule-based matrix: {matrix_path}")

    print("\n[9b] Adding interaction features (non-leaky combinations to help separate Medium from High)...")
    df["interact_age_frustration"] = df["ticket_age_hours"] * df["frustration_negation_ratio"]
    df["interact_age_frustrated_anchor"] = df["ticket_age_hours"] * df["sim_frustrated_anchor"]
    df["interact_age_question"] = df["ticket_age_hours"] * df["question_ratio"]
    df["interact_repeats_flags"] = df["repeat_contact_count"] * df["flag_count"]

    print("\n[10] Computing deterministic priority score + label (formula + noise)...")
    score_rng = random.Random(RANDOM_STATE)
    scores = [compute_priority_score(row, matrix, score_rng) for row in df.to_dict("records")]
    df["priority_score"] = scores

    thresholds = compute_quantile_thresholds(scores)
    print(f"    Score distribution: min={min(scores):.3f} max={max(scores):.3f} "
          f"mean={np.mean(scores):.3f}")
    print(f"    Calibrated thresholds (quantile-based, since the formula's raw range "
          f"rarely spans [0,1]): {thresholds}")
    thresholds_path = os.path.join(MODEL_DIR, "priority_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"    Saved thresholds: {thresholds_path}")

    df["priority"] = [priority_score_to_label(s, thresholds) for s in scores]

    print("\n    Priority distribution:")
    for p, count in df["priority"].value_counts().reindex(PRIORITY_LABELS).items():
        print(f"      {p:<10} {count:>6}  ({count / len(df):.1%})")

    print("\n[11] Train/test split (80/20, stratified by priority)...")
    class_counts = df["priority"].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        print(f"    Classes too rare to stratify ({rare_classes}) go entirely into train.")
        rare_df = df[df["priority"].isin(rare_classes)]
        stratifiable_df = df[~df["priority"].isin(rare_classes)]
        train_df, test_df = train_test_split(
            stratifiable_df, test_size=0.2, random_state=RANDOM_STATE, stratify=stratifiable_df["priority"]
        )
        train_df = pd.concat([train_df, rare_df])
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["priority"]
        )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"    Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    print("\n[12] Saving artifacts...")
    out_path = os.path.join(TRAINING_DATA_DIR, "priority_training.csv")
    df.to_csv(out_path, index=False)
    print(f"    Saved CSV: {out_path}  ({len(df):,} rows, {len(df.columns)} columns)")

    label2id = {label: idx for idx, label in enumerate(PRIORITY_LABELS)}
    id2label = {str(idx): label for label, idx in label2id.items()}
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.json")
    with open(encoder_path, "w") as f:
        json.dump({"labels": PRIORITY_LABELS, "label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"    Saved label encoder: {encoder_path}")

    print(f"\n{'=' * 65}")
    print(f"  DONE — {len(df):,} rows, {len(PRIORITY_LABELS)} priority classes")
    print(f"  Ready to train: python -m src.ml_pipeline.train_priority_model")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
