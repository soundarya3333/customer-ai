"""
Prepare HF dataset for category classification training
Maps HF categories to your 14 business categories
"""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from datasets import load_dataset
import pandas as pd
import json

print("[LOADING] Hugging Face dataset...")
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

print(f"[OK] Loaded {len(dataset)} examples")

# Your 14 business categories
YOUR_CATEGORIES = [
    "Account & Authentication",
    "Billing & Payments",
    "Orders & Shipping",
    "Returns & Refunds",
    "Product Defects & Quality Issues",
    "Technical Support",
    "Subscription & Account Management",
    "Security & Fraud",
    "Feature Requests",
    "Customer Feedback",
    "Complaints & Escalations",
    "Sales & Pre-Sales Inquiries",
    "General Inquiries",
    "Compliance & Privacy Requests"
]

# Map HF categories to your categories
category_mapping = {
    "Account Access": "Account & Authentication",
    "Account management": "Subscription & Account Management",
    "Account Management": "Subscription & Account Management",
    "Billing": "Billing & Payments",
    "Complaint": "Complaints & Escalations",
    "Feedback": "Customer Feedback",
    "Feature request": "Feature Requests",
    "Feature Request": "Feature Requests",
    "Issue": "Technical Support",
    "Payment issue": "Billing & Payments",
    "Refund": "Returns & Refunds",
    "Return": "Returns & Refunds",
    "Shipping": "Orders & Shipping",
    "Delivery": "Orders & Shipping",
    "Delivery Issue": "Orders & Shipping",
    "Product Defect": "Product Defects & Quality Issues",
    "Product defect": "Product Defects & Quality Issues",
    "Defect": "Product Defects & Quality Issues",
    "Technical Problem": "Technical Support",
    "Technical problem": "Technical Support",
    "Security": "Security & Fraud",
    "Fraud": "Security & Fraud",
    "Sales": "Sales & Pre-Sales Inquiries",
    "Pre-sales": "Sales & Pre-Sales Inquiries",
    "Pre-Sales": "Sales & Pre-Sales Inquiries",
    "General": "General Inquiries",
    "General Inquiry": "General Inquiries",
    "Compliance": "Compliance & Privacy Requests",
    "Privacy": "Compliance & Privacy Requests",
}

# Extract and map data
print("\n[PROCESSING] Processing dataset...")
texts = []
categories = []
unmapped_count = 0

for example in dataset:
    text = example.get("instruction", "")
    hf_category = example.get("category", "")

    if text and hf_category:
        # Try exact match first
        if hf_category in category_mapping:
            mapped_category = category_mapping[hf_category]
        # Try case-insensitive match
        elif hf_category.lower() in [k.lower() for k in category_mapping.keys()]:
            mapped_category = category_mapping[next(k for k in category_mapping.keys() if k.lower() == hf_category.lower())]
        else:
            # Default to General Inquiries
            mapped_category = "General Inquiries"
            unmapped_count += 1

        texts.append(text)
        categories.append(mapped_category)

print(f"[OK] Processed {len(texts)} examples")
print(f"[WARNING] Unmapped categories (defaulted to General): {unmapped_count}")

# Create DataFrame
df = pd.DataFrame({
    "text": texts,
    "category": categories
})

# Show distribution
print("\n[STATS] Category distribution:")
print(df["category"].value_counts())

# Save to CSV
output_path = "customer-support-ai/training_data/category_training.csv"
os.makedirs("customer-support-ai/training_data", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n[OK] Saved to {output_path}")

# Also save category list for reference
with open("customer-support-ai/training_data/categories.json", "w") as f:
    json.dump(YOUR_CATEGORIES, f, indent=2)

print("[OK] Ready to train!")
