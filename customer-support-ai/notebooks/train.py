import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    # 1. Configuration
    MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # Swap with Mistral-7B-v0.3 if you have >16GB VRAM
    DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    OUTPUT_DIR = "./models/customer_support_lora"

    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and Prepare the Bitext Dataset
    print("Loading Bitext dataset from Hugging Face...")
    dataset = load_dataset(DATASET_ID, split="train")

    # The dataset contains 'instruction' and 'response' columns.
    # Map them to a standard training format.
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = (
                f"### System:\nYou are an AI customer support assistant. Provide accurate and polite replies.\n\n"
                f"### User:\n{example['instruction'][i]}\n\n"
                f"### Assistant:\n{example['response'][i]}"
            )
            output_texts.append(text)
        return output_texts

    # 3. Configure 4-bit Quantization (QLoRA) to save VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model in 4-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for gradient checkpointing / low precision training
    model = prepare_model_for_kbit_training(model)

    # 4. Configure LoRA (Targeting attention layers)
    peft_config = LoraConfig(
        r=16,                  # Rank: higher means more parameters to train
        lora_alpha=32,         # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Standard for Llama/Mistral
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Set Up Training Hyperparameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,     # Small batch size prevents Out-Of-Memory (OOM) errors
        gradient_accumulation_steps=4,     # Simulates a total batch size of 8 (2 * 4)
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,                     # For testing. Increase to 1000+ for full training
        optim="paged_adamw_8bit",          # Special Windows/QLoRA friendly optimizer
        fp16=True,
        save_strategy="steps",
        save_steps=50,
        report_to="none"                   # Set to "mlflow" later once MLflow is wired up
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    # 7. Start Training
    print("Starting fine-tuning pipeline...")
    trainer.train()

    # 8. Save the LoRA Adapter weights
    print(f"Training finished! Saving weights to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()