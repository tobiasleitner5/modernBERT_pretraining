import argparse
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertTokenizer

# This script is designed to pretrain the ModernBERT model on a custom dataset.

# Read input parameters from the command line
parser = argparse.ArgumentParser(description="ModernBERT pretraining script")
parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing the zipped training data")
args = parser.parse_args()

if not args.data_folder:
    raise ValueError("Data folder path is required. Please provide it using the --data_folder argument.")
print(f"Data folder: {args.data_folder}")

config = ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base")

# Here you can see the exact configuration of the model, which is based on the ModernBERT architecture.
print(config)

# Configuration can be changed easily e.g.
# config.num_hidden_layers = 6

# Note, a model in transformer consists of two parts: the configuration and the weights. 
# The configuration defines the architecture of the model; the weights are the learned parameters.
#  When you load a model using `from_pretrained`, it automatically loads both.
# Since we want to do the pretraining ourselves, we will initialize the model with the configuration only.

# Weights are randomly initialized
model = ModernBertForMaskedLM(config)