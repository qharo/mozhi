from typing import Dict, Iterator
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
from config import config
import random
from tqdm import tqdm

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

# dataset => split_dataset => split_dataloaders
def get_split_loaders(dataset: IterableDataset, train_size: float = 0.7, val_size: float = 0.01, 
                  test_size: float = 0.19, seed: int = 42) -> Dict[str, DataLoader]:
    random.seed(seed)
    
    splits = {"train": train_size, "val": val_size, "test": test_size}
    
    # checks if split proportions add up
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1, got {total}")
    
    split_names, weights = zip(*splits.items())
    
    # generator given a dataset
    # => Tuple(split_name: str, sample)
    def split_generator(dataset: IterableDataset) -> Iterator[tuple]:
        split_names, weights = zip(*splits.items())
        for item in dataset:
            yield random.choices(split_names, weights=weights)[0], item

    # generator given a split_name
    # => Tuple(split_name: str, sample) of only specified split_name
    def create_subset(split_name: str) -> Iterator:
        return (item for split, item in split_generator(dataset) if split == split_name)

    # creates DataLoader from split_name
    def create_dataloader(split_name: str) -> DataLoader:
        return DataLoader(
            IterableDataset.from_generator(lambda: create_subset(split_name)),
            batch_size=config.batch_size if config else 1,
            num_workers=1,
            prefetch_factor=2
        )

    return (
        create_dataloader("train"),
        create_dataloader("val"),
        create_dataloader("test"),
    )
