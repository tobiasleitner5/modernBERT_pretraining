
from transformers import PreTrainedTokenizerFast
import argparse
from tqdm import tqdm
from utils import extract_body_from_gzipped_csvs


parser = argparse.ArgumentParser(description="ModernBERT pretraining script")
parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing the zipped training data")
parser.add_argument("--tokenizer_path", type=str, default="./my_tokenizer", help="Path to the trained tokenizer")
args = parser.parse_args()

# Load the trained tokenizer
hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
print(f"Loaded tokenizer from {args.tokenizer_path} with vocab size: {hf_tokenizer.vocab_size}")

# print number of tokens in the dataset (stream through data again)
print("Counting total tokens in dataset...")
total_tokens = 0
text_count = 0
for text in tqdm(extract_body_from_gzipped_csvs(args.data_folder), desc="Encoding texts", unit="article"):
    total_tokens += len(hf_tokenizer.encode(text))
    text_count += 1
print(f"Total articles: {text_count}")
print(f"Total tokens in the dataset: {total_tokens}")