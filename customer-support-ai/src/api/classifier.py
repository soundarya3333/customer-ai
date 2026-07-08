"""
Zero-shot category classifier - NO TRAINING NEEDED
Uses pre-trained model directly from HuggingFace
"""
from transformers import pipeline

CATEGORIES = [
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

# Zero-shot classifier - no training required!
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_ticket(text: str):
    """Classify text into one of the 14 categories without training"""
    result = classifier(text, CATEGORIES, multi_class=False)
    return {
        "category": result["labels"][0],
        "confidence": round(result["scores"][0], 2)
    }

if __name__ == "__main__":
    # Test
    test = "My order is late and I want a refund"
    result = classify_ticket(test)
    print(f"Test: {test}")
    print(f"Result: {result}")
