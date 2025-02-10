# HuggingFace libraries
    # library for downlaoading datasets
from datasets import load_dataset
    # library for training the tokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
    # training your tokenizer
from tokenizers.trainers import WordLevelTrainer
    # customize how pre-tokenization (e.g., splitting into words) is done
from tokenizers.pre_tokenizers import Whitespace

# torch libraries 
import torch 
import torch.nn as nn

# Other libraries
import os
import wandb
from pathlib import Path


# Load the dataset
dataset_path = os.getcwd()+'/dataset'  # Get the current working directory
if not os.path.exists(dataset_path): # If the dataset directory exists
    os.makedirs(dataset_path) # Create the dataset directory
books = load_dataset("opus_books", "en-fr", cache_dir=dataset_path )
print(books.cache_files)
