
import os
import sys
import json
import re

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "category_classifier")
RANDOM_STATE = 42


ML_CATEGORIES = [
    "Account & Authentication",
    "Billing & Payments",
    "Orders & Shipping",
    "Returns & Refunds",
    "Customer Feedback",
    "Complaints & Escalations",
    "Subscription & Account Management",
    "General Inquiries",
]

# categories with zero signal in Bitext — routed via keyword rules, not ML.
RULE_ONLY_CATEGORIES = [
    "Product Defects & Quality Issues",
    "Technical Support",
    "Security & Fraud",
    "Feature Requests",
    "Sales & Pre-Sales Inquiries",
    "Compliance & Privacy Requests",
]

PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")


INTENT_TO_BUSINESS_CATEGORY = {
    "create_account": "Account & Authentication",
    "delete_account": "Account & Authentication",
    "edit_account": "Account & Authentication",
    "recover_password": "Account & Authentication",
    "registration_problems": "Account & Authentication",
    "switch_account": "Account & Authentication",
    "check_cancellation_fee": "Returns & Refunds",
    "contact_customer_service": "General Inquiries",
    
    "contact_human_agent": "General Inquiries",
    "delivery_options": "Orders & Shipping",
    "delivery_period": "Orders & Shipping",
    "complaint": "Complaints & Escalations",
    "review": "Customer Feedback",
    "check_invoice": "Billing & Payments",
    "get_invoice": "Billing & Payments",
    "cancel_order": "Orders & Shipping",
    "change_order": "Orders & Shipping",
    "place_order": "Orders & Shipping",
    "track_order": "Orders & Shipping",
    "check_payment_methods": "Billing & Payments",
    "payment_issue": "Billing & Payments",
    "check_refund_policy": "Returns & Refunds",
    "get_refund": "Returns & Refunds",
    "track_refund": "Returns & Refunds",
    "change_shipping_address": "Orders & Shipping",
    "set_up_shipping_address": "Orders & Shipping",
    "newsletter_subscription": "Subscription & Account Management",
}


def intent_mapping(hf_category: str, intent: str) -> str | None:
    """Returns the business category for a real HF (category, intent) pair, or
    None if the intent is unknown (should not happen with the current dataset,
    but we don't want to silently bucket unknowns into a catch-all)."""
    return INTENT_TO_BUSINESS_CATEGORY.get(intent.strip())


def normalize_text(text: str) -> str:
    no_placeholder = PLACEHOLDER_RE.sub("", text)
    lowered = no_placeholder.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", "", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def main():
    print("=" * 65)
    print("  DATA PREPARATION — Bitext HF Dataset (cleaned)")
    print("=" * 65)

    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1] Loading HuggingFace dataset...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    print(f"    Loaded {len(dataset):,} examples")

    raw_categories = sorted(set(c.strip() for c in dataset["category"] if c))
    raw_intents = sorted(set(i.strip() for i in dataset["intent"] if i))
    print(f"\n[2] Real unique HF categories ({len(raw_categories)}): {raw_categories}")
    print(f"    Real unique HF intents ({len(raw_intents)}): {raw_intents}")
    unmapped = sorted(set(raw_intents) - set(INTENT_TO_BUSINESS_CATEGORY))
    if unmapped:
        print(f"    [WARNING] {len(unmapped)} intents have no mapping and will be dropped: {unmapped}")
    print(f"    ML-trainable business categories ({len(ML_CATEGORIES)}): {ML_CATEGORIES}")
    print(f"    Rule-only business categories, no HF signal ({len(RULE_ONLY_CATEGORIES)}): {RULE_ONLY_CATEGORIES}")

    print("\n[3] Extracting text, category, and intent (group key)...")
    rows = []
    for ex in dataset:
        raw_text = (ex.get("instruction") or "").strip()
        hf_cat = (ex.get("category") or "").strip()
        intent = (ex.get("intent") or "").strip()
        if not raw_text or not hf_cat or not intent:
            continue
        business_cat = intent_mapping(hf_cat, intent)
        if business_cat is None:
            continue
        cleaned_text = PLACEHOLDER_RE.sub("", raw_text).strip()
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        if len(cleaned_text) < 5:
            continue
        rows.append({
            "text": cleaned_text,
            "category": business_cat,
            "intent": intent,
            "norm": normalize_text(raw_text),
        })
    df = pd.DataFrame(rows)
    print(f"    Extracted {len(df):,} rows")

    print("\n[4] Deduplicating...")
    before = len(df)
    df = df.drop_duplicates(subset=["norm"]).reset_index(drop=True)
    print(f"    Removed {before - len(df):,} exact/near duplicates")
    print(f"    Remaining: {len(df):,} rows")

    print("\n[5] Category distribution (after dedup, before balancing):")
    for cat, count in df["category"].value_counts().items():
        print(f"      {cat:<45} {count:>5}")

    print(f"\n[6] Group-aware train/test split (by intent, stratified per category)...")
    min_per_class = 30
    counts = df["category"].value_counts()
    present = counts[counts >= min_per_class].index.tolist()
    df = df[df["category"].isin(present)].reset_index(drop=True)
    print(f"    Classes with >={min_per_class} samples: {len(present)}")

   
    df["split"] = "train"
    for cat in present:
        cat_mask = df["category"] == cat
        cat_intents = df.loc[cat_mask, "intent"].unique().tolist()
        if len(cat_intents) > 1:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
            cat_df = df.loc[cat_mask].reset_index()
            train_idx, test_idx = next(gss.split(cat_df, groups=cat_df["intent"]))
            test_original_idx = cat_df.loc[test_idx, "index"]
            if len(test_original_idx) == 0:
                # extreme edge case: force at least one intent into test
                fallback_intent = cat_intents[-1]
                test_original_idx = df.loc[cat_mask & (df["intent"] == fallback_intent)].index
            df.loc[test_original_idx, "split"] = "test"
        else:
            
            cat_idx = df.loc[cat_mask].sample(frac=0.2, random_state=RANDOM_STATE).index
            df.loc[cat_idx, "split"] = "test"

    train_full = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    print(f"\n[7] Balancing training set (oversample minority, cap majority)...")
    TARGET_PER_CLASS = 1200

    def rebalance(group):
        n = len(group)
        if n >= TARGET_PER_CLASS:
            return group.sample(TARGET_PER_CLASS, random_state=RANDOM_STATE, replace=False)
        return group.sample(TARGET_PER_CLASS, random_state=RANDOM_STATE, replace=True)

    train_df = (
        train_full.groupby("category", group_keys=False)
        .apply(rebalance)
        .reset_index(drop=True)
    )
    print(f"    Target per class: {TARGET_PER_CLASS}")
    for cat, count in train_full["category"].value_counts().items():
        oversampled = count < TARGET_PER_CLASS
        print(f"      {cat:<45} {count:>5} -> {TARGET_PER_CLASS:>5} {'(oversampled)' if oversampled else '(downsampled)'}")

    df = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)
    n_test = len(test_df)
    print(f"\n    Train: {n_train:,}  |  Test: {n_test:,}")
    print(f"    Unique intents in train: {df[df['split']=='train']['intent'].nunique()}")
    print(f"    Unique intents in test:  {df[df['split']=='test']['intent'].nunique()}")
    overlap = set(df[df['split']=='train']['intent']) & set(df[df['split']=='test']['intent'])
    print(f"    Intent overlap train/test: {len(overlap)} (should be 0)")

    print(f"\n    Test-set category distribution:")
    for cat, count in df[df["split"] == "test"]["category"].value_counts().items():
        print(f"      {cat:<45} {count:>5}")

    n_classes = df["category"].nunique()
    print(f"\n    Classes: {n_classes}")

    print("\n[8] Saving artifacts...")
    csv_path = os.path.join(TRAINING_DATA_DIR, "category_training.csv")
    df[["text", "category", "intent", "split"]].to_csv(csv_path, index=False)
    print(f"    Saved CSV: {csv_path}")

    present_categories = sorted(df["category"].unique().tolist())
    label2id = {cat: idx for idx, cat in enumerate(present_categories)}
    id2label = {str(idx): cat for cat, idx in label2id.items()}
    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.json")
    with open(label_encoder_path, "w") as f:
        json.dump({
            "labels": present_categories,
            "label2id": label2id,
            "id2label": id2label,
        }, f, indent=2)
    print(f"    Saved label encoder: {label_encoder_path}")

    print(f"\n{'=' * 65}")
    print(f"  DONE — {len(df):,} rows, {n_classes} classes")
    print(f"  Train: {n_train:,}  |  Test: {n_test:,} (no intent leakage)")
    print(f"  Ready to train: python -m src.ml_pipeline.train_classifier")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
