from tqdm import tqdm
import os
from typing import Dict
from datasets import load_dataset, load_from_disk, IterableDataset
from torch.utils.data import DataLoader
from transformers import MarianTokenizer, AutoTokenizer
from config import config
import random

# => IterableDataset (tokenized)
def tkize_dataset(dataset, src_tkizer, tgt_tkizer) -> IterableDataset:
    def tkize_sample(example):
        src_tkized = src_tkizer(
            example["src"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        tgt_tkized = tgt_tkizer(
            example["tgt"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": src_tkized["input_ids"].squeeze(0),
            "attention_mask": src_tkized["attention_mask"].squeeze(0),
            "labels": tgt_tkized["input_ids"].squeeze(0)
        }

    return IterableDataset.from_generator(lambda: map(tkize_sample, dataset))

# => IterableDataset
def get_dataset() -> IterableDataset:
    dataset = load_dataset(
        "ai4bharat/samanantar",
        f"{config.target_lang}",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    return dataset

# => Dict[str, IterableDataset]
def split_dataset(dataset: IterableDataset, train_size=0.7, val_size=0.15, test_size=0.15, seed=42) -> Dict[str, IterableDataset]:
    random.seed(seed)

    def split_generator(dataset: IterableDataset, splits: Dict[str, float]):
        for item in dataset:
            yield (random.choices(list(splits.keys()), weights=list(splits.values()))[0], item)

    splits = {"train": train_size, "val": val_size, "test": test_size}

    def create_subset(split_name: str):
        for split, item in split_generator(dataset, splits):
            if split == split_name:
                yield item

    return {
        "train": IterableDataset.from_generator(lambda: create_subset("train")),
        "val": IterableDataset.from_generator(lambda: create_subset("val")),
        "test": IterableDataset.from_generator(lambda: create_subset("test"))
    }

# => T5Tokenizers
def build_tkizers(dataset: IterableDataset):
    # load directly if saved
    if config.tkizer_save:
        if os.path.exists(config.src_tkizer_save_path) and os.path.exists(config.tgt_tkizer_save_path):
            src_tkizer = AutoTokenizer.from_pretrained(config.src_tkizer_save_path, use_fast=True)
            tgt_tkizer = AutoTokenizer.from_pretrained(config.tgt_tkizer_save_path, use_fast=True)
            print(f"Tokenizers loaded")
            return src_tkizer, tgt_tkizer

    # iterator over dataset
    src_iterator = (item['src'] for item in dataset)
    tgt_iterator = (item['tgt'] for item in dataset)

    # train tkizer from our datasets
    pretrained_tkizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    src_tkizer = pretrained_tkizer.train_new_from_iterator(src_iterator, vocab_size=config.src_vocab_size)
    tgt_tkizer = pretrained_tkizer.train_new_from_iterator(tgt_iterator, vocab_size=config.tgt_vocab_size)

    # Ensure all necessary special tokens are present
    special_tokens = {
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
    }

    # add special tokens
    src_tkizer.add_special_tokens(special_tokens)
    tgt_tkizer.add_special_tokens(special_tokens)


    # Add extra_id tokens (sentinel tokens)
    num_extra_ids = 100  # T5 typically uses 100 sentinel tokens
    src_tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})
    tgt_tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})

    print(f"Tokenizers built")

    # save tkizers
    if config.tkizer_save:
        if not os.path.exists(config.src_tkizer_save_path):
            os.makedirs(config.src_tkizer_save_path)
            src_tkizer.save_pretrained(config.src_tkizer_save_path)
        if not os.path.exists(config.tgt_tkizer_save_path):
            os.makedirs(config.tgt_tkizer_save_path)
            tgt_tkizer.save_pretrained(config.tgt_tkizer_save_path)
        print(f"Tokenizer saved")

    return src_tkizer, tgt_tkizer
