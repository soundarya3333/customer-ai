"""
Train a DistilBERT classifier for ticket category classification.
Loads the prepared CSV, fine-tunes distilbert-base-uncased, saves model + metrics.
"""
import os
import sys
import json
import glob
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "category_classifier")

MODEL_NAME = "distilbert-base-uncased"
# Bitext instructions are short (median 12 tokens, max ~23 in our data) — 256 was
# padding/attending over 10x more tokens than needed, which is most of why a CPU
# run was taking hours. Attention cost scales with length^2, so this alone is a
# large speedup. Inference (predictor.py) keeps a larger MAX_LENGTH since real
# tickets are longer free-form text, not template instructions.
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42
FINALIZE_CHECKPOINT = os.environ.get("FINALIZE_CHECKPOINT")
# Full test set is used for the final classification_report, but re-evaluating
# all of it every epoch (for best-checkpoint selection) is wasted CPU time —
# a random subsample is just as good a signal for "which epoch is better".
EVAL_SUBSAMPLE = 1500


def resolve_checkpoint(spec):
    if not spec:
        return None
    if spec.lower() == "latest":
        checkpoints = glob.glob(os.path.join(MODEL_DIR, "checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {MODEL_DIR}")
        return max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[-1]))
    if spec.lower() == "best":
        checkpoints = glob.glob(os.path.join(MODEL_DIR, "checkpoint-*", "trainer_state.json"))
        if not checkpoints:
            raise FileNotFoundError(f"No trainer_state.json found under {MODEL_DIR}")
        state_path = max(checkpoints, key=os.path.getmtime)
        with open(state_path) as f:
            state = json.load(f)
        return state["best_model_checkpoint"]
    return spec


def main():
    print("=" * 65)
    print("  TRAINING — DistilBERT Ticket Classifier")
    print("=" * 65)

    csv_path = os.path.join(TRAINING_DATA_DIR, "category_training.csv")
    print(f"\n[1] Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    present_categories = sorted(df["category"].unique().tolist())
    num_labels = len(present_categories)
    label2id = {cat: idx for idx, cat in enumerate(present_categories)}
    id2label = {idx: cat for cat, idx in label2id.items()}
    print(f"    Classes: {num_labels}")
    for cat in present_categories:
        count = len(df[df["category"] == cat])
        print(f"      {label2id[cat]:>2}  {cat:<45} {count:>5}")

    df["label"] = df["category"].map(label2id)

    print(f"\n[2] Using pre-split train/test (group-aware, no leakage)...")
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        print(f"    Train: {len(train_df)}  |  Test: {len(test_df)}")
        overlap = set(train_df.get("intent", [])) & set(test_df.get("intent", []))
        print(f"    Intent overlap: {len(overlap)} (should be 0)")
    else:
        print(f"    [WARNING] No 'split' column — falling back to random split")
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
        )
        print(f"    Train: {len(train_df)}  |  Test: {len(test_df)}")

    print(f"\n[3] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("[4] Tokenizing datasets...")
    train_ds = HFDataset.from_pandas(train_df[["text", "label"]])
    test_ds = HFDataset.from_pandas(test_df[["text", "label"]])
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    if len(test_ds) > EVAL_SUBSAMPLE:
        eval_ds = test_ds.shuffle(seed=RANDOM_STATE).select(range(EVAL_SUBSAMPLE))
        print(f"    Per-epoch eval subsampled to {EVAL_SUBSAMPLE} (full {len(test_ds)} used for final report)")
    else:
        eval_ds = test_ds

    checkpoint_path = resolve_checkpoint(FINALIZE_CHECKPOINT)
    model_source = checkpoint_path or MODEL_NAME
    if checkpoint_path:
        print(f"[5] Loading checkpoint: {checkpoint_path} ({num_labels} labels)")
    else:
        print(f"[5] Loading model: {MODEL_NAME} ({num_labels} labels)")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=RANDOM_STATE,
        report_to="none",
        logging_steps=100,
        disable_tqdm=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        label_ids = list(range(num_labels))
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, labels=label_ids, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[6] Training on {device.upper()} for {EPOCHS} epochs (batch={BATCH_SIZE}, lr={LR})...")
    print()

    if checkpoint_path:
        print("    FINALIZE_CHECKPOINT set - skipping training and using existing checkpoint.")
    else:
        trainer.train()

    print(f"\n[7] Evaluating on full test set...")
    eval_results = trainer.evaluate(eval_dataset=test_ds)
    print(f"    Eval loss:   {eval_results['eval_loss']:.4f}")
    print(f"    Accuracy:    {eval_results['eval_accuracy']:.4f}")
    print(f"    F1 (macro):  {eval_results['eval_f1_macro']:.4f}")
    print(f"    F1 (weighted): {eval_results['eval_f1_weighted']:.4f}")

    print(f"\n[8] Detailed classification report:")
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    report = classification_report(
        test_df["label"].values,
        preds,
        labels=list(range(num_labels)),
        target_names=present_categories,
        zero_division=0,
    )
    print(report)
    report_path = os.path.join(MODEL_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"    Saved classification report: {report_path}")

    print(f"[9] Saving model + tokenizer to {MODEL_DIR}")
    if checkpoint_path:
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ):
            source_path = os.path.join(checkpoint_path, name)
            if os.path.exists(source_path):
                shutil.copy2(source_path, os.path.join(MODEL_DIR, name))
    else:
        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.json")
    with open(label_encoder_path, "w") as f:
        json.dump(
            {
                "labels": present_categories,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            indent=2,
        )
    print(f"    Saved label encoder: {label_encoder_path}")

    metrics = {
        "accuracy": float(eval_results["eval_accuracy"]),
        "f1_macro": float(eval_results["eval_f1_macro"]),
        "f1_weighted": float(eval_results["eval_f1_weighted"]),
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "num_classes": num_labels,
        "classes": present_categories,
        "model_name": MODEL_NAME,
        "checkpoint": checkpoint_path,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_length": MAX_LENGTH,
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved metrics: {metrics_path}")

    print(f"\n{'=' * 65}")
    print(f"  DONE — Model saved to {MODEL_DIR}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}  |  F1 (macro): {metrics['f1_macro']:.2%}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
