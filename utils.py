import gzip
import csv
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Download punkt tokenizer for sentence splitting (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def split_into_sentences(text, sep_token="[SEP]"):
    """Split text into sentences and join with [SEP] token."""
    sentences = sent_tokenize(text)
    return f" {sep_token} ".join(sentences)


def count_gz_files(data_folder):
    """Count the number of gzipped files in a folder."""
    return len(list(Path(data_folder).glob("**/*.gz")))


# Load data from gzipped CSV files, extracting the "Body" column
def extract_body_from_gzipped_csvs(data_folder, add_sentence_separators=True, text_column="Body"):
    """Generator that yields text from 'Body' column of all gzipped CSV files.
    
    Args:
        data_folder: Path to folder containing gzipped CSV files
        add_sentence_separators: If True, split articles into sentences separated by [SEP]
        text_column: The column name to extract text from
    
    Yields:
        Processed text strings one at a time (memory efficient)
    """
    gz_files = list(Path(data_folder).glob("**/*.gz"))
    print(f"Found {len(gz_files)} gzipped files")

    for gz_file in tqdm(gz_files, desc="Processing files", unit="file"):
        with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row and row[text_column]:
                    text = row[text_column]
                    if add_sentence_separators:
                        text = split_into_sentences(text)
                    yield text