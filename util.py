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
from torch.utils.data import random_split, Dataset, DataLoader
# Other libraries
import os
import wandb
from pathlib import Path
from typing import Dict, Iterable

# Build tokenizer


def batch_iterator(dataset: Iterable[Dict], language: str):
    """
    Iterates over the dataset and yields the text for the specified language.

    Args:
        dataset (Iterable[Dict]): The dataset to iterate over. Should be a collection of dictionaries 
                                    where each item contains a 'translation' key.
        language (str): The language key to retrieve from each item's 'translation'. 

    Yields:
        str: The text in the specified language from each item in the dataset.
    """
    for item in dataset:
        yield item['translation'][language]


def build_tokenizer(config: Dict, dataset: Iterable[Dict], language: str):
    """
    Builds and trains a tokenizer for the specified language from the provided dataset. 
    If a tokenizer already exists, it is loaded from the specified path.

    Args:
        config (Dict): The configuration dictionary that contains the path for saving/loading the tokenizer.
                       The path should include a placeholder for the language (e.g., './tokenizer/tokenizer_{0}.json').
        dataset (Iterable[Dict]): The dataset used for training the tokenizer. Should be a collection of
                                    dictionaries where each item contains a 'translation' key with text.
        language (str): The language key used to extract the text for training the tokenizer from the 'translation' field.

    Returns:
        Tokenizer: A trained tokenizer object.
    """
    tokenizer_path = Path(config['DATASET']['tokenizer_file'].format(language))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        os.makedirs(tokenizer_path.parent, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token='[unk]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[
            "[PAD]",
            "[SOS]",
            "[EOS]",
            "[unk]",
        ], min_frequency=2)

        tokenizer.train_from_iterator(
            batch_iterator(dataset, language), trainer=trainer)
        # Save the tokenizer
        tokenizer.save(str(tokenizer_path))
    return tokenizer


# Load the dataset

def load_dataset(config):
    cache_dir = os.path.join(
        config['DATASET']['dataset_path'], config['DATASET']['dataset_name'])
    if not os.path.exists(cache_dir):  # If the dataset directory does not exists
        os.makedirs(cache_dir)  # Create the dataset directory

    source_lang = config['DATASET']['source_lang']
    target_lang = config['DATASET']['target_lang']
    dataset_name = config['DATASET']['dataset_name']
    max_seq_len = config['MODEL']['source_sq_len']

    dataset = load_dataset(
        dataset_name, f'{source_lang}-{target_lang}', cache_dir=cache_dir, split='train')

    # build tokenizer
    source_tokenizer = build_tokenizer(config, dataset, source_lang)
    target_tokenizer = build_tokenizer(config, dataset, target_lang)
    # create train, val, and test datasets

    train_dataset_size = int(0.8*len(dataset))
    val_dataset_size = int((len(dataset) - train_dataset_size) // 2)
    test_dataset_size = len(dataset) - train_dataset_size - val_dataset_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_dataset_size, val_dataset_size, test_dataset_size])

    # generate datasets
    train_dataset = Seq2SeqDataset(
        train_dataset, source_tokenizer, target_tokenizer, source_lang, target_lang, max_seq_len)
    val_dataset = Seq2SeqDataset(
        val_dataset,   source_tokenizer, target_tokenizer, source_lang, target_lang, max_seq_len)
    test_dataset = Seq2SeqDataset(
        test_dataset,  source_tokenizer, target_tokenizer, source_lang, target_lang, max_seq_len)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config['TRAIN']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, source_tokenizer, target_tokenizer


# Seq2SeqDataset
class Seq2SeqDataset(Dataset):
    def __init__(self, dataset, source_tokenizer, target_tokenizer, source_lang, target_lang, max_seq_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_seq_len = max_seq_len

        # start of a sentence ID
        self.sos_token = torch.Tensor(
            [self.source_tokenizer.token_to_id(['[SOS]'])], dtype=torch.int64)
        # end of a sentence ID
        self.eos_token = torch.Tensor(
            [self.source_tokenizer.token_to_id(['[EOS]'])], dtype=torch.int64)
        # padds
        self.pad_token = torch.Tensor(
            [self.source_tokenizer.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_target_pair = self.dataset[idx]
        source_text = source_target_pair['translation'][self.source_lang]
        target_text = source_target_pair['translation'][self.target_lang]

        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        # in encoder we add two SOS and EOS token to the encoder tokens
        encoder_padding_tokens = self.max_seq_len - \
            len(encoder_input_tokens) - 2
        # in decoder, we only add SOS token to the decoder tokens
        decoder_padding_tokens = self.max_seq_len - \
            len(decoder_input_tokens) - 1

        # Check the number of tokens is not bigger than max_seq_len
        if encoder_padding_tokens < 0 or decoder_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        # We need to add SOS and EOS tokens into encoder tokens and pad the sentence to the max_seq_len
        encoder_input = torch.cat([self.sos_token,
                                   torch.tensor(
                                       encoder_input_tokens, dtype=torch.int64),
                                   self.eos_token,
                                   torch.tensor([self.pad_token] * encoder_padding_tokens, dtype=torch.int64)])

        # We need to add SOS token into decoder tokens and pad the sentence to the max_seq_len
        decoder_input = torch.cat([self.sos_token,
                                   torch.tensor(
                                       decoder_input_tokens, dtype=torch.int64),
                                   torch.tensor([self.pad_token] * decoder_padding_tokens, dtype=torch.int64)])

        # Create a label by adding EOS to the label ( the output that we expect from the decoder)
        label = torch.cat([torch.tensor(decoder_input_tokens, dtype=torch.int64),
                           self.eos_token,
                           torch.tensor([self.pad_token] * decoder_padding_tokens, dtype=torch.int64)])

        # Check the length of the inputs are correct
        assert encoder_input.size(0) == self.max_seq_len
        assert decoder_input.size(0) == self.max_seq_len
        assert label.size(0) == self.max_seq_len

        # Create encoder mask
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(
            0).unsqueeze(0).int()  # (1, 1, max_seq_len)

        # Create decoder
        # Padding mask for the decoder
        decoder_padding_mask = (decoder_input != self.pad_token).unsqueeze(
            0).unsqueeze(0).int()  # (1, 1, max_seq_len)
        # Causal mask for the decoder (lower triangular matrix)
        seq_len = decoder_input.size(0)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), diagonal=1)).type(
            torch.int64)  # (seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(
            0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        # Combine padding mask and causal mask for the decoder
        # (1, 1, seq_len, seq_len)
        decoder_mask = (causal_mask & decoder_padding_mask.unsqueeze(2))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,  # (1, 1, max_seq_len)
            "decoder_mask": decoder_mask,  # (1, 1, max_seq_len)
            "label": label,  # max_seq_len
            "source_text": source_text,
            "target_text": target_text,
        }

# get saved model
def get_weights(config, epoch: str):
    model_folder = config['BENCHMARK']['model_folder']
    model_name = config['BENCHMARK']['model_name']
    model_filename = f'{model_name}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)
