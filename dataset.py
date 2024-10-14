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

# serves from DataFrame
class RawDataset(Dataset):
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

# creates a DataFrame from HF Dataset
# => str (DataFrame save path)
def download_dataset() -> str:
    df_path = os.path.join(config.data_save_path, 'df.csv')
    
    # load DataFrame if already saved
    if os.path.exists(df_path):
        print("Dataset DataFrame already exists, loaded")
        return os.path.join(config.data_save_path, 'df.csv')
    
    # else, stream it in batch mode
    dataset = load_dataset(
        "ai4bharat/samanantar",
        f"{config.target_lang}",
        split="train",
        streaming=True,
        trust_remote_code=True
    ).batch(config.transfer_batch_size)

    # create and save DataFrame
    data = {"src": [], "tgt": []}
    for sample in tqdm(dataset, desc="Downloading dataset"):
        data["src"] += sample['src']
        data["tgt"] += sample['tgt']
    os.makedirs(config.data_save_path, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(df_path)

    return df_path

# tokenizes a batch
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

import time
def tkize_dataset(df_path, tkizers):
    df = pd.read_csv(df_path).head(2000000)

    input_ids = np.zeros((len(df), config.max_length))
    attention_masks = np.zeros((len(df), config.max_length))
    labels = np.zeros((len(df), config.max_length))

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        src_tokens = tkizers[0](
                        row['src'])
                        # padding="max_length",
                        # truncation=True,
                        # max_length=config.max_length,
                        # return_tensors="pt")
        tgt_tokens = tkizers[1](
                        row['tgt'])
                        # padding="max_length",
                        # truncation=True,
                        # max_length=config.max_length,
                        # return_tensors="pt")

        src_length = len(src_tokens['input_ids'])
        tgt_length = len(tgt_tokens['input_ids'])
        input_ids[i][:src_length] = src_tokens['input_ids'][:config.max_length]
        attention_masks[i][:src_length] = src_tokens['attention_mask'][:config.max_length]
        labels[i][:tgt_length] = tgt_tokens['input_ids'][:config.max_length]

    np.save("data/input_ids", input_ids)
    np.save("data/attention_mask", attention_masks)
    np.save("data/labels", labels)
    return "data/input_ids", "data/attention_mask", "data/labels"


from torch.utils.data import TensorDataset, DataLoader

def create_dataloader_from_numpy(input_ids_path, attention_mask_path, labels_path, batch_size=32, shuffle=True):
    # Load numpy arrays
    input_ids = np.load(input_ids_path)
    attention_mask = np.load(attention_mask_path)
    labels = np.load(labels_path)
    
    # Convert numpy arrays to PyTorch tensors
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create TensorDataset
    dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


# creates split dataloaders
def get_split_loaders(dataset, tkizers, seed=42):
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    collator = partial(tkize, tkizers)

    create_dataloader = lambda dataset, to_shuffle = False: DataLoader(dataset, 
                                        batch_size=config.batch_size, 
                                        shuffle=to_shuffle, 
                                        num_workers=config.num_workers, 
                                        collate_fn=collator)
    
    return (
        create_dataloader(train_dataset, True),
        create_dataloader(val_dataset),
        create_dataloader(test_dataset),
    )

# from typing import Dict, Iterator
# from datasets import load_dataset, IterableDataset, Dataset
# from torch.utils.data import DataLoader, random_split
# from config import config
# import random
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer
# import os
# import torch
# from functools import partial

# # serves from DataFrame
# # class RawDataset(Dataset):
# #     def __init__(self, df_path):
# #         self.df = pd.read_csv(df_path)

# #     def __len__(self):
# #         return len(self.df)

# #     def __getitem__(self, idx):
# #         return self.df.iloc[idx]

# # creates a DataFrame from HF Dataset
# # => str (DataFrame save path)
# def download_dataset():
#     df_path = os.path.join(config.data_save_path, 'df.csv')
    
#     # load DataFrame if already saved
#     if os.path.exists(df_path):
#         print("Dataset already exists, loaded")
#         df = pd.read_csv(df_path)
#         translation_dataset = Dataset.from_dict({
#             "src": df["src"].to_list(),
#             "tgt": df["tgt"].to_list()
#         })
#         return translation_dataset, os.path.join(config.data_save_path, 'df.csv')
    
#     # else, stream it in batch mode
#     dataset = load_dataset(
#         "ai4bharat/samanantar",
#         f"{config.target_lang}",
#         split="train",
#         streaming=True,
#         trust_remote_code=True
#     ).batch(config.transfer_batch_size)

#     # create and save DataFrame
#     data = {"src": [], "tgt": []}
#     for sample in tqdm(dataset, desc="Downloading dataset"):
#         data["src"] += sample['src']
#         data["tgt"] += sample['tgt']
#     os.makedirs(config.data_save_path, exist_ok=True)
#     df = pd.DataFrame(data)
#     df.to_csv(df_path)

#     translation_dataset = Dataset.from_dict({
#         "src": df["src"].to_list(),
#         "tgt": df["tgt"].to_list()
#     }),

#     return translation_dataset, df_path

# # Tokenizes a batch
# def tkize_batch(tkizers, batch):
#     src_tkizer, tgt_tkizer = tkizers
#     # Tokenize the whole batch at once
#     src_tokens = src_tkizer(
#         batch['src'],
#         padding="max_length",
#         truncation=True,
#         max_length=config.max_length,
#         return_tensors="pt"
#     )
#     tgt_tokens = tgt_tkizer(
#         batch['tgt'],
#         padding="max_length",
#         truncation=True,
#         max_length=config.max_length,
#         return_tensors="pt"
#     )
    
#     return {
#         "input_ids": src_tokens['input_ids'],
#         'attention_mask': src_tokens['attention_mask'],
#         "labels": tgt_tokens['input_ids'],
#     }

# def get_split_sets(dataset, tkizers, seed=42):
#     total_size = len(dataset)
#     train_size = int(config.train_split * total_size)
#     val_size = int(config.val_split * total_size)
#     test_size = total_size - train_size - val_size
    
#     # Batch dataset, apply tokenization in parallel, then split
#     dataset = dataset.map(
#         lambda examples: tkize_batch(tkizers, examples),
#         remove_columns=dataset.column_names,
#         batched=True,  # Process batches of examples in one go
#         num_proc=2     # Parallelize tokenization with 4 processes (adjust based on system)
#     )
#     dataset.save_to_disk("data/dataset.hf")

#     # Split dataset
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, 
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(seed)
#     )

#     return train_dataset, val_dataset, test_dataset