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
    df = pd.read_csv(df_path).head(100)
    time1 = time.time()

    progress_bar = tqdm(total=len(df), desc="Tokenizing")
    def get_value(row, tkizer):
        progress_bar.update(1)
        output = tkizer(row)
        return np.array(output['input_ids']), np.array(output['attention_mask'])

    # Apply the function to the 'src' column
    output = df['src'].apply(lambda x: get_value(x, tkizers[0]))
    print(output[0])
    tgt_input_ids, _ = df['tgt'].apply(lambda x: get_value(x, tkizers[1]))

    print(type(src_input_ids))
    print(time.time() - time1)

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