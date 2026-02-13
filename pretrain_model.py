import argparse
import os
from transformers import (
    ModernBertConfig, 
    ModernBertForMaskedLM, 
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from utils import extract_body_from_gzipped_csvs

# This script is designed to pretrain the ModernBERT model on a custom dataset.

# Read input parameters from the command line
parser = argparse.ArgumentParser(description="ModernBERT pretraining script")
parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing the zipped training data")
parser.add_argument("--tokenizer_path", type=str, default="./my_tokenizer", help="Path to the trained tokenizer")
parser.add_argument("--output_dir", type=str, default="./modernbert_pretrained", help="Output directory for the model")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
args = parser.parse_args()

if not args.data_folder:
    raise ValueError("Data folder path is required. Please provide it using the --data_folder argument.")
if not os.path.exists(args.data_folder):
    raise ValueError(f"The provided data folder path does not exist: {args.data_folder}")
print(f"Data folder: {args.data_folder}")

# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)
print(f"Output directory: {args.output_dir}")

# Load configuration and initialize model with random weights
config = ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base")
print(config)

# Weights are randomly initialized
model = ModernBertForMaskedLM(config)
print(f"Model initialized with {model.num_parameters():,} parameters")

# Load the trained tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
print(f"Loaded tokenizer from {args.tokenizer_path} with vocab size: {tokenizer.vocab_size}")

# Resize model embeddings to match tokenizer vocab size
# as we do not use the pretrained tokenizer, our vocab size is different
model.resize_token_embeddings(len(tokenizer))

# Create dataset from generator
# The huggingface dataset class expects a dictionary with a "text" key, which is what we will use for the input sequences.
# Each line in the csv is a separate dict with a "text" key containing the article text.
print("Loading dataset...")
def generate_examples():
    for text in extract_body_from_gzipped_csvs(args.data_folder):
        yield {"text": text}

# As we are working with streaming data
dataset = Dataset.from_generator(generate_examples)
print(f"Dataset loaded with {len(dataset)} examples")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )

# The whole dataset is tokenized in a streaming fashion, which is memory efficient.
# This is done in batches.
# Note that the tokenization is not done on the fly during training. It is done before.
cache_path = os.path.join(args.output_dir, "tokenized_dataset.arrow")
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing",
    cache_file_name=cache_path,
    load_from_cache_file=True
)

# Data collator for MLM: it does the masking.
# mlm_probability=0.15 means 15% of tokens are masked (standard BERT approach)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Training arguments
# Here mostly standard settings - investigation what is the best settings
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # Use mixed precision if GPU supports it
    dataloader_num_workers=4,
    report_to="none",  # Set to "wandb" if you want W&B logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
print("Starting pretraining...")
trainer.train()

# Save the final model
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")