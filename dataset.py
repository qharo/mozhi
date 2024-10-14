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
from torch.utils.data import TensorDataset, DataLoader

# creates a DataFrame from HF Dataset
# => str (DataFrame save path)
def download_dataset() -> str:
    df_path = os.path.join(config.data_save_path, 'df.csv')
    
    # load DataFrame if already saved
    if os.path.exists(df_path):
        print("DataFrame already exists, loaded")
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

# from np arrays create dataloaders
# np_array => dataloaders
def create_dataloaders(np_input_ids, np_attention_mask, np_labels):
    tensors = [torch.tensor(arr, dtype=torch.long) for arr in (np_input_ids, np_attention_mask, np_labels)]
    dataset = TensorDataset(*tensors)
    sizes = [int(config.train_split * len(dataset)), int(config.val_split * len(dataset))]
    sizes.append(len(dataset) - sum(sizes))
    datasets = random_split(dataset, sizes)
    
    def collate_fn(batch):
        return {k: torch.stack(v) for k, v in zip(['input_ids', 'attention_mask', 'labels'], zip(*batch))}
    
    return [DataLoader(ds, batch_size=config.batch_size, shuffle=(i==0), num_workers=2, pin_memory=True, collate_fn=collate_fn) for i, ds in enumerate(datasets)]

# from df_path, tokenizers
# df_path, tkizers => dataloaders
def get_tkized_dataloaders(df_path, tkizers):
    cache_files = ["data/input_ids.npy", "data/attention_mask.npy", "data/labels.npy"]
    if all(os.path.exists(f) for f in cache_files):
        return create_dataloaders(*[np.load(f) for f in cache_files])
    
    df = pd.read_csv(df_path).head(500000)
    src_tkizer, tgt_tkizer = tkizers
    arrays = [np.zeros((len(df), config.max_length)) for _ in range(3)]
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        for j, (tkizer, col) in enumerate(zip([src_tkizer, src_tkizer, tgt_tkizer], ['src', 'src', 'tgt'])):
            tokens = tkizer(row[col])
            arrays[j][i][:len(tokens['input_ids'])] = tokens['input_ids' if j != 1 else 'attention_mask'][:config.max_length]
    
    for arr, name in zip(arrays, ['input_ids', 'attention_mask', 'labels']):
        np.save(f"data/{name}", arr)
    
    return create_dataloaders(*arrays)

# serves from DataFrame
# class RawDataset(Dataset):
#     def __init__(self, df_path):
#         self.df = pd.read_csv(df_path)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         return self.df.iloc[idx]
# tokenizes a batch
# def tkize(tkizers, batch):
#     batch = pd.DataFrame(batch)
#     src_tkizer, tgt_tkizer = tkizers
#     src_tokens = src_tkizer(
#         batch['src'].tolist(),
#         padding="max_length",
#         truncation=True,
#         max_length=config.max_length,
#         return_tensors="pt"
#     )       
#     tgt_tokens = tgt_tkizer(
#         batch['tgt'].tolist(),
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

# def create_dataloaders(np_input_ids, np_attention_mask, np_labels):
#     # Convert numpy arrays to PyTorch tensors
#     input_ids_tensor = torch.tensor(np_input_ids, dtype=torch.long)
#     attention_mask_tensor = torch.tensor(np_attention_mask, dtype=torch.long)
#     labels_tensor = torch.tensor(np_labels, dtype=torch.long)

#     # Create TensorDataset
#     dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

#     total_size = len(dataset)
#     train_size = int(config.train_split * total_size)
#     val_size = int(config.val_split * total_size)
#     test_size = total_size - train_size - val_size
    
#     # Split the dataset
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#     # Create DataLoaders
#     train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
#     # Custom collate function to return dictionary
#     def collate_fn(batch):
#         input_ids, attention_mask, labels = zip(*batch)
#         return {
#             "input_ids": torch.stack(input_ids),
#             "attention_mask": torch.stack(attention_mask),
#             "labels": torch.stack(labels)
#         }
    
#     # Apply custom collate function to DataLoaders
#     train_dataloader.collate_fn = collate_fn
#     val_dataloader.collate_fn = collate_fn
#     test_dataloader.collate_fn = collate_fn

#     return train_dataloader, val_dataloader, test_dataloader

# def tkize_dataset(df_path, tkizers):

#     if os.path.exists("data/input_ids.npy") and os.path.exists("data/labels.npy") and os.path.exists("data/attention_mask.npy"):
#         return create_dataloaders(np.load("data/input_ids.npy"), np.load("data/labels.npy"), np.load("data/attention_mask.npy"))

#     df = pd.read_csv(df_path).head(500000)
#     src_tkizer, tgt_tkizer = tkizers

#     input_ids = np.zeros((len(df), config.max_length))
#     attention_masks = np.zeros((len(df), config.max_length))
#     labels = np.zeros((len(df), config.max_length))

#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
#         src_tokens = src_tkizer(row['src'])
#         tgt_tokens = tgt_tkizer( row['tgt'])

#         src_length = len(src_tokens['input_ids'])
#         tgt_length = len(tgt_tokens['input_ids'])
#         input_ids[i][:src_length] = src_tokens['input_ids'][:config.max_length]
#         attention_masks[i][:src_length] = src_tokens['attention_mask'][:config.max_length]
#         labels[i][:tgt_length] = tgt_tokens['input_ids'][:config.max_length]

#     np.save("data/input_ids", input_ids)
#     np.save("data/attention_mask", attention_masks)
#     np.save("data/labels", labels)

#     return create_dataloaders(input_ids, attention_masks, labels)



# def create_dataloader_from_numpy(input_ids_path, attention_mask_path, labels_path, shuffle=True):
#     # Load numpy arrays
#     input_ids = np.load(input_ids_path)
#     attention_mask = np.load(attention_mask_path)
#     labels = np.load(labels_path)

#     # Convert numpy arrays to PyTorch tensors
#     input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
#     attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
#     labels_tensor = torch.tensor(labels, dtype=torch.long)
    
#     # Create TensorDataset
#     dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)

#     total_size = len(dataset)
#     train_size = int(config.train_split * total_size)
#     val_size = int(config.val_split * total_size)
#     test_size = total_size - train_size - val_size
    
#     # Split the dataset
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
#     # Create DataLoaders
#     train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
#     # Custom collate function to return dictionary
#     def collate_fn(batch):
#         input_ids, attention_mask, labels = zip(*batch)
#         return {
#             "input_ids": torch.stack(input_ids),
#             "attention_mask": torch.stack(attention_mask),
#             "labels": torch.stack(labels)
#         }
    
#     # Apply custom collate function to DataLoaders
#     train_dataloader.collate_fn = collate_fn
#     val_dataloader.collate_fn = collate_fn
#     test_dataloader.collate_fn = collate_fn


#     return train_dataloader, val_dataloader, test_dataloader


# # creates split dataloaders
# def get_split_loaders(dataset, tkizers, seed=42):
#     total_size = len(dataset)
#     train_size = int(config.train_split * total_size)
#     val_size = int(config.val_split * total_size)
#     test_size = total_size - train_size - val_size

#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, 
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(seed)
#     )

#     collator = partial(tkize, tkizers)

#     create_dataloader = lambda dataset, to_shuffle = False: DataLoader(dataset, 
#                                         batch_size=config.batch_size, 
#                                         shuffle=to_shuffle, 
#                                         num_workers=config.num_workers, 
#                                         collate_fn=collator)
    
#     return (
#         create_dataloader(train_dataset, True),
#         create_dataloader(val_dataset),
#         create_dataloader(test_dataset),
#     )

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