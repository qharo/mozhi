from tqdm import tqdm
import os
import torch
import math
from typing import Dict
from datasets import load_dataset, load_from_disk, IterableDataset, Dataset
from torch.utils.data import DataLoader
from transformers import MarianTokenizer, AutoTokenizer
from config import config
import random
import itertools

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
            "input_ids": torch.Tensor(src_tkized["input_ids"].squeeze(0)),
            "attention_mask": torch.Tensor(src_tkized["attention_mask"].squeeze(0)),
            "labels": torch.Tensor(tgt_tkized["input_ids"].squeeze(0))
        }

    return IterableDataset.from_generator(lambda: map(tkize_sample, dataset))

# => IterableDataset
def get_iterable_dataset() -> IterableDataset:
    dataset = load_dataset(
        "ai4bharat/samanantar",
        f"{config.target_lang}",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    return dataset

# => Dict[str, IterableDataset]
def split_dataset(dataset: IterableDataset, train_size=0.7, val_size=0.01, test_size=0.19, seed=42) -> Dict[str, IterableDataset]:
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

