| Component             | Technology                                      |
| --------------------- | ----------------------------------------------- |
| Backend API           | FastAPI                                         |
| LLM                   | Llama 3 / Mistral 7B                            |
| Fine-tuning           | LoRA + PEFT + Hugging Face Transformers         |
| Vector DB             | FAISS or ChromaDB                               |
| Embeddings            | Sentence Transformers or BAAI/bge-small-en-v1.5 |
| RAG                   | LangChain or LlamaIndex                         |
| Sentiment             | DistilBERT or RoBERTa                           |
| Ticket Classification | Fine-tuned BERT or classical XGBoost            |
| Priority Prediction   | XGBoost / LightGBM                              |
| Database              | PostgreSQL (Docker)                             |
| Monitoring            | Prometheus + Grafana                            |
| Container             | Docker                                          |
|                             |
| Experiment Tracking   | MLflow                                          |
| Dataset Versioning    | DVC                                             |
| CI/CD                 | GitHub Actions                                  |

Project Overview
This project is an AI-powered Customer Support Platform that automates the process of understanding customer queries, classifying tickets, prioritizing them, retrieving relevant company knowledge, generating responses, and allowing human agents to approve replies before they're sent.

Step 1 – Customer submits a request
Step 2 – Sentiment Analysis through bert
Step 3 – Ticket Classification
Example categories:
Delivery Issue
Refund Request
Technical Problem
Account Access
Billing
Product Defect
Step 4 – Priority Prediction using lightgbm or xgboost
Step 5 – Retrieval-Augmented Generation (RAG)
Step 6 – Fine-Tuned LLM Generates the Response
Step 8 – Store Everything in posgresql
Step 9 – Monitoring
The application exposes metrics that Prometheus collects.
A Grafana dashboard displays operational and model metrics
Step 10 – MLOps 

Progress
day 1
normalization after all the data ingestion
final format

{
    "ticket_id":"...",
    "customer_id":"...",
    "source":"...",
    "subject":"...",
    "customer_message":"...",
    "priority":"...",
    "status":"...",
    "timestamp":"..."
}







Category	Fallback Score
Complaints & Escalations	0.559
Account & Authentication	0.399
Orders & Shipping	0.356
Billing & Payments	0.346
Returns & Refunds	0.341
General Inquiries	0.292
Subscription & Account Management	0.254
Customer Feedback	0.228



Individual Feature Weights Chart
These are the weights used in the final priority score:






score = (
    # Business Impact (most important)
    0.25 * impact
    
    # Sentiment Features
    + 0.18 * vader_neg           # Negative sentiment proportion
    + 0.12 * vader_compound_neg  # Negative intensity (0 to 1)
    
    # SLA & Operational
    + 0.15 * is_sla_breached     # 1 if SLA breached, else 0
    
    # Linguistic Flags (from Bitext)
    + 0.05 * has_negative_flag   # 'N' flag
    + 0.05 * has_emotional_flag  # 'E' flag
    + 0.03 * has_typo_flag       # 'Z' flag (typos = rushed)
    + 0.03 * has_multi_issue_flag # 'M' flag (multiple problems)
    
    # Text Ratios
    + 0.06 * min(threat_language_ratio * 5.0, 1.0)  # Capped threat score
    
    # Semantic Embeddings
    + 0.05 * sim_urgent_anchor   # Similarity to "urgent" anchor
    + 0.03 * sim_critical_anchor # Similarity to "critical" anchor
)

Feature	Value	Source
category	"Billing & Payments" (categorical)	INTENT_TO_BUSINESS_CATEGORY mapping
customer_tier	"Pro" (categorical)	
source_channel	"zendesk" (categorical)	
vader_neu	0.42	VADER
vader_pos	0.00	VADER
sim_frustrated_anchor	0.71	MiniLM embedding
sim_routine_anchor	0.18	MiniLM embedding
urgency_keyword_ratio	0.08	Text ratio
time_pressure_ratio	0.00	Text ratio
frustration_negation_ratio	0.15	Text ratio
profanity_ratio	0.00	Text ratio
exclamation_ratio	1.0	Text ratio
question_ratio	0.0	Text ratio
caps_word_ratio	0.0	Text ratio
text_length_words	12	Text ratio
text_length_chars	68	Text ratio
has_context_flag	1	Bitext flags
has_imperative_flag	1	Bitext flags
has_question_flag	0	Bitext flags
has_polite_flag	0	Bitext flags
has_knowledge_flag	0	Bitext flags
has_basic_flag	1	Bitext flags
has_long_flag	1	Bitext flags
flag_count	4	Bitext flags
ticket_age_hours	18.3	Synthetic
repeat_contact_count	3	Synthetic
LABEL	High (2)	Deterministic formula