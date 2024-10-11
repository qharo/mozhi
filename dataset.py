from typing import Dict, Iterator
from datasets import load_dataset, IterableDataset, Dataset
from torch.utils.data import DataLoader, Dataset, random_split
from config import config
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os
import torch
from functools import partial

class RawDataset(Dataset):
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


# # => IterableDataset (tokenized)
# def tkize_dataset(df_path: str, src_tkizer: AutoTokenizer, tgt_tkizer: AutoTokenizer) -> IterableDataset:
    
#     # if os.path.exists(os.path.join(config.data_save_path, 'input_ids.npy')) and os.path.exists(os.path.join(config.data_save_path, 'attention_mask.npy')):
#     #     if os.path.exists(os.path.join(config.data_save_path, 'labels.npy')):
#     #         return TkizedDataset(
#     #             os.path.join(config.data_save_path, 'input_ids.npy'),
#     #             os.path.join(config.data_save_path, 'attention_mask.npy'),
#     #             os.path.join(config.data_save_path, 'labels.npy'),
#     #         )

#     df = pd.read_csv(df_path)
#     print(len(df))
#     input_ids = np.zeros((len(df), config.max_length))
#     attention_mask = np.zeros((len(df), config.max_length))
#     labels = np.zeros((len(df), config.max_length))

#     for i in tqdm(range(0, len(df), config.batch_size), total=len(df)/config.batch_size, desc="Tokenizing"):

#         batch = df.iloc[i:i+config.batch_size]
        
#         src_tokens = src_tkizer(
#             batch['src'].tolist(),
#             padding="max_length",
#             truncation=True,
#             max_length=config.max_length,
#             return_tensors="np"
#         )       
#         tgt_tokens = tgt_tkizer(
#             batch['tgt'].tolist(),
#             padding="max_length",
#             truncation=True,
#             max_length=config.max_length,
#             return_tensors="np"
#         )
        
#         end_idx = min(i + config.batch_size, len(df))

#         input_ids[i:end_idx] = src_tokens.input_ids[:end_idx-i]
#         attention_mask[i:end_idx] = src_tokens.attention_mask[:end_idx-i]
        
#         # For labels, we'll use the input_ids. Adjust this if you have separate labels.
#         labels[i:end_idx] = tgt_tokens.input_ids[:end_idx-i]

#     np.save(os.path.join(config.data_save_path, 'input_ids'), input_ids)
#     np.save(os.path.join(config.data_save_path, 'attention_mask'), attention_mask)
#     np.save(os.path.join(config.data_save_path, 'labels'), labels)

#     return TkizedDataset(
#         os.path.join(config.data_save_path, 'input_ids.npy'),
#         os.path.join(config.data_save_path, 'attention_mask.npy'),
#         os.path.join(config.data_save_path, 'labels.npy'),
#     )

# creates a DataFrame from HF Dataset
# => str (DataFrame save path)
def get_dataset() -> IterableDataset:
    if os.path.exists(os.path.join(config.data_save_path, 'df.csv')):
        print("DF already exists, loaded")
        return os.path.join(config.data_save_path, 'df.csv')
    
    dataset = load_dataset(
        "ai4bharat/samanantar",
        f"{config.target_lang}",
        split="train",
        streaming=True,
        trust_remote_code=True
    ).batch(config.transfer_batch_size)

    src = []
    tgt = []
    for i, sample in tqdm(enumerate(dataset), desc="Downloading dataset"):
        src += sample['src']
        tgt += sample['tgt']

    df = pd.DataFrame({
      "src": src,
      "tgt": tgt  
    })

    df.to_csv(os.path.join(config.data_save_path, "df.csv"))
    return os.path.join(config.data_save_path, "df.csv")

def split_dataloader(dataset, tkizers, seed=42):
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print("train seize", len(train_dataset))

    def tkize(tkizers, batch):
        batch = pd.DataFrame(batch)
        src_tkizer, tgt_tkizer = tkizers
        src_tokens = src_tkizer(
            batch['src'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )       
        tgt_tokens = tgt_tkizer(
            batch['tgt'].tolist(),
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": src_tokens['input_ids'],
            'attention_mask': src_tokens['attention_mask'],
            "labels": tgt_tokens['input_ids'],
        }

    collator = partial(tkize, tkizers)

    create_dataloader = lambda dataset : DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collator)
    
    return (
        create_dataloader(train_dataset),
        create_dataloader(val_dataset),
        create_dataloader(test_dataset),
    )


# dataset => split_dataset => split_dataloaders
def get_split_loaders(dataset: IterableDataset, seed: int = 42) -> Dict[str, DataLoader]:
    random.seed(seed)
    
    splits = {"train": config.train_split, "val": config.val_split, "test": config.test_split}
    
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
            # num_workers=1,
            # prefetch_factor=2
        )

    print(f"Train: {config.n_samples*config.train_split//config.batch_size}, Val: {config.n_samples*config.val_split//config.batch_size}, Test: {config.n_samples*config.test_split//config.batch_size}")

    return (
        create_dataloader("train"),
        create_dataloader("val"),
        create_dataloader("test"),
    )
