from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
import argparse
import os
from utils import extract_body_from_gzipped_csvs

# This script is designed to pretrain the ModernBERT tokenizer on a custom dataset.
# Since we also need to make sure to avoid lookahead bias / data leakage here, we will train the tokenizer from scratch.
# The tokenizer library is a Rust-based library.

# Read input parameters from the command line
parser = argparse.ArgumentParser(description="ModernBERT pretraining script")
parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing the zipped training data")
args = parser.parse_args()

if not args.data_folder:
    raise ValueError("Data folder path is required. Please provide it using the --data_folder argument.")
print(f"Data folder: {args.data_folder}")

# Validate if the provided path exists and is a directory
if not os.path.exists(args.data_folder):
    raise ValueError(f"The provided data folder path does not exist: {args.data_folder}")
if not os.path.isdir(args.data_folder):
    raise ValueError(f"The provided data folder path is not a directory: {args.data_folder}")

# ModernBERT uses a BPE tokenizer, which is a subword tokenization method
# The pretokenizer is responsible for splitting the input text into smaller units.
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# I am telling the tokenizer that I have 5 special tokens.
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Create trainer
# The trainer finds the subword units and builds the vocabulary based on the input data.
trainer = trainers.BpeTrainer(
    vocab_size=50265,  # You have to set a vocab size. 
    special_tokens=special_tokens, # The first 5 IDs are reserved for the special tokens
    min_frequency=2, # tokens that do not appear at lest 2 times are not included in the vocab.
)

# Train directly from generator - no temp file, truly streaming!
# Each line in the csv is a separate sequence (article).
# The articles are split into sentences separated by [SEP] tokens in utils.py.
print("Training tokenizer (streaming from gzipped files)...")
tokenizer.train_from_iterator(
    extract_body_from_gzipped_csvs(args.data_folder),
    trainer=trainer,
)
print("Tokenizer training finished!")

# Add post-processor for BERT-style formatting
# The post-processor is responsible for adding special tokens to the input sequences in the correct format for BERT.
# This is already during inference, not during training.
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
)

# Convert to a Hugging Face tokenizer for use with transformers
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
hf_tokenizer.save_pretrained("./my_tokenizer")

print("Tokenizer training complete. Saved to ./my_tokenizer")

# hf_tokenizer.push_to_hub("tobileitner/ModernBERT-tokenizer")