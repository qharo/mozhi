from tqdm import tqdm
import os
from typing import Dict
from datasets import load_dataset, load_from_disk, IterableDataset
from torch.utils.data import DataLoader
from transformers import MarianTokenizer, AutoTokenizer
from config import config
import random

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

# Debugging wrapper
def debug_dataset(dataset):
    for batch in dataset:
        print(f"Dataset output shapes: {', '.join(f'{k}: {v.shape}' for k, v in batch.items())}")
        return batch  # Just print the first batch and stop

# -> IterableDataset (tokenized)
def tkize_dataset2(dataset, src_tkizer, tgt_tkizer) -> IterableDataset:
    def tkize_sample(examples):
        src_tkized = src_tkizer(
            examples["src"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        tgt_tkized = tgt_tkizer(
            examples["tgt"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": src_tkized["input_ids"],
            "attention_mask": src_tkized["attention_mask"],
            "labels": tgt_tkized["input_ids"]
        }

    return IterableDataset.from_generator(
        lambda: map(tkize_sample, dataset)
    )

    return dataset.map(
        tkize_sample,
        batched=True,
        remove_columns=dataset.column_names
    )


# -> Dataset object
def download_dataset() -> IterableDataset:
    # Check if the dataset exists in the specified location
    # if config.data_save and os.path.exists(config.data_save_path):
    #     print(f"Loading existing dataset from {config.data_save_path}")
    #     return load_from_disk(config.data_save_path)

    # If not, download the dataset
    dataset = load_dataset(
        "ai4bharat/samanantar",
        f"{config.target_lang}",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Save the dataset to disk
    # if config.data_save:
    #     os.makedirs(config.data_save_path)
    #     print(f"Saving dataset to {config.data_save_path}")
    #     dataset.save_to_disk(config.data_save_path)

    # print(f"Dataset 'ai4bharat/samanantar' downloaded and saved!")
    return dataset


def split_iterable_dataset(dataset: IterableDataset, train_size=0.7, val_size=0.15, test_size=0.15, seed=42) -> Dict[str, IterableDataset]:
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

# -> Dict[str: Dataset]
def split_dataset(dataset, train_size=0.7, val_size=0.15, test_size=0.15, seed=42) -> Dict[str, IterableDataset]:
    dataset = dataset.shuffle(seed=seed)
    total_size = len(dataset)

    train_end = int(total_size * train_size)
    val_end = train_end + int(total_size * val_size)
    time1 = time.time()
    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]
    print(f"Dataset split: train->{train_end}, val->{val_end-train_end}, test->{total_size-val_end}")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

# -> T5Tokenizers
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
    src_tokenizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})
    tgt_tokenizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})

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

# # -> Dict[str, DataLoader]
# def prepare_dataloaders(dataset_splits, src_tokenizer, tgt_tokenizer):
#     dataloaders = {}
#     for split, data in tqdm(dataset_splits.items(), total=len(dataset_splits.keys()), desc="Preparing dataloaders..."):
#         custom_dataset = TokenizedDataset(data, src_tokenizer, tgt_tokenizer, config.max_length)
#         dataloaders[split] = DataLoader(custom_dataset, batch_size=config.batch_size, shuffle=(split == 'train'))
#     print("Dataloaders prepared")
#     return dataloaders

# # split dataset
# def split_dataset(dataset, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):

#     stream = dataset.shuffle(seed=seed)

#     # size of the dataset
#     # total_size = 0
#     #for _ in tqdm(stream, desc="Creating split"):
#     #    total_size += 1

#     total_size = 5264867

#     config.n_samples = total_size
#     config.n_steps = (total_size * config.num_train_epochs) // config.batch_size


#     # Calculate split sizes
#     train_end = int(total_size * train_size)
#     val_end = train_end + int(total_size * val_size)

#     # Create split datasets
#     train_dataset = stream.take(train_end)
#     val_dataset = stream.skip(train_end).take(val_end - train_end)
#     test_dataset = stream.skip(val_end)

#     return {
#         "train": train_dataset,
#         "val": val_dataset,
#         "test": test_dataset
#     }


# # given the dataset, builds a tokenizer from vocab
# def build_tokenizer(dataset, ext: str, tkizer_save_path: str, tkizer_to_save: bool = False, vocab_size: int = 50000) -> MarianTokenizer:
#     # loads tkizer if already created
#     if os.path.exists(tkizer_save_path):
#         print(f"Tokenizer loaded from {tkizer_save_path}")
#         return AutoTokenizer.from_pretrained(tkizer_save_path)

#     # supplies lists (batches) of sentences (ext : src/tgt)
#     text_iterator = lambda: (batch[ext] for batch in dataset.shuffle().iter(batch_size=1000))

#     # creates the tokenizer
#     pretrained_tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
#     custom_tokenizer = pretrained_tokenizer.train_new_from_iterator(text_iterator(), vocab_size=vocab_size)

#     # saves the tokenizer
#     if tkizer_to_save:
#         if not os.path.exists(tkizer_save_path):
#             os.makedirs(tkizer_save_path)
#         custom_tokenizer.save_pretrained(tkizer_save_path)

#     return custom_tokenizer

# # downloads HF dataset
# def download_dataset():

#     # get dataset
#     dataset = load_dataset("ai4bharat/samanantar",
#         f"{config.target_lang}",
#         split="train",
#         trust_remote_code=True
#     )
#     return dataset


# def tkize_dataset(examples, src_tkizer, tgt_tkizer):
#     src_tkized = src_tkizer(
#         examples["src"],
#         padding="max_length",
#         truncation=True,
#         max_length=config.max_length,
#         return_tensors="pt"
#     )
#     tgt_tkized = tgt_tkizer(
#         examples["tgt"],
#         padding="max_length",
#         truncation=True,
#         max_length=config.max_length,
#         return_tensors="pt"
#     )
#     return {
#         "input_ids": src_tkized["input_ids"],
#         "attention_mask": src_tkized["attention_mask"],
#         "labels": tgt_tkized["input_ids"]
#     }


# class TokenizedDataset(Dataset):
#     def __init__(self, data, src_tokenizer, tgt_tokenizer):
#         super().__init__()
#         self._data = data
#         self.src_tokenizer = src_tokenizer
#         self.tgt_tokenizer = tgt_tokenizer
#         self.max_length = config.max_length

#     @property
#     def data(self):
#         return self._data

#     def __len__(self):
#         return len(self._data)

#     def __getitem__(self, idx):
#         item = self._data[idx]
#         src_encoding = self.src_tokenizer(
#             item['src'],
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         tgt_encoding = self.tgt_tokenizer(
#             item['tgt'],
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': src_encoding['input_ids'].squeeze(),
#             'attention_mask': src_encoding['attention_mask'].squeeze(),
#             'labels': tgt_encoding['input_ids'].squeeze()
#         }
