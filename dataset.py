from tqdm import tqdm
import os
from typing import Dict
from datasets import load_dataset, load_from_disk, IterableDataset, Dataset
from torch.utils.data import DataLoader
from transformers import MarianTokenizer, AutoTokenizer
from config import config
import random

from concurrent.futures import ProcessPoolExecutor

# def tokenize(example, src_tkizer, tgt_tkizer):
#         src_text = example['src']
#         tgt_text = example['tgt']

#         src_tokens = src_tkizer(src_text, max_length=self.max_length, truncation=True, padding='max_length')
#         tgt_tokens = tgt_tkizer(tgt_text, max_length=self.max_length, truncation=True, padding='max_length')
        
#         return {
#             'input_ids': src_tokens['input_ids'],
#             'attention_mask': src_tokens['attention_mask'],
#             'labels': tgt_tokens['input_ids'],
#         }


# class TokenizedDataset(Dataset):
#     def __init__(self, i_dataset: IterableDataset, src_tokenizer, tgt_tokenizer, max_length: int = config.max_length):
#         self._data = []
#         self.src_tokenizer = src_tokenizer
#         self.tgt_tokenizer = tgt_tokenizer
#         self.max_length = max_length

#         buffer = []
#         for string_sample in tqdm(i_dataset, desc="Creating tokenized dataset"):
#             buffer.append(string_sample)

#             if


#             tkized_sample = self.tokenize_sample(string_sample)
#             self._data.append(tkized_sample)

#     def tokenize_sample(self, item):
#         src_text = item['src']
#         tgt_text = item['tgt']

#         src_tokens = self.src_tokenizer(src_text, max_length=self.max_length, truncation=True, padding='max_length')
#         tgt_tokens = self.tgt_tokenizer(tgt_text, max_length=self.max_length, truncation=True, padding='max_length')
        
#         return {
#             'input_ids': src_tokens['input_ids'],
#             'attention_mask': src_tokens['attention_mask'],
#             'labels': tgt_tokens['input_ids'],
#         }

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

#     def save_to_disk(self, path):
#         dataset = datasets.Dataset.from_dict({k: [d[k] for d in self.data] for k in self.data[0]})
#         dataset.save_to_disk(path)


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

    return IterableDataset.from_generator(lambda: map(tkize_sample, dataset)).batch(config.batch_size)

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

