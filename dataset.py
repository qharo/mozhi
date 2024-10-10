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

from accelerate import Accelerator

class MultiGPUIterableDataset(IterableDataset):
    def __init__(self, dataset, accelerator: Accelerator):
        self.dataset = dataset
        self.accelerator = accelerator

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.accelerator.process_index
            iter_end = None
        else:
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))

        return iter(itertools.islice(self.dataset, iter_start, iter_end))


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

