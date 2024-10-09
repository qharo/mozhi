from tqdm import tqdm
import os
from datasets import load_dataset
from transformers import MarianTokenizer, AutoTokenizer
from config import config
import random

# split dataset
def split_dataset(dataset, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):

    stream = dataset.shuffle(seed=seed)

    # size of the dataset
    # total_size = 0
    #for _ in tqdm(stream, desc="Creating split"):
    #    total_size += 1

    total_size = 5264867

    config.n_samples = total_size
    config.n_steps = (total_size * config.num_train_epochs) // config.batch_size


    # Calculate split sizes
    train_end = int(total_size * train_size)
    val_end = train_end + int(total_size * val_size)

    # Create split datasets
    train_dataset = stream.take(train_end)
    val_dataset = stream.skip(train_end).take(val_end - train_end)
    test_dataset = stream.skip(val_end)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


# given the dataset, builds a tokenizer from vocab
def build_tokenizer(dataset, ext: str, tkizer_save_path: str, tkizer_to_save: bool = False, vocab_size: int = 50000) -> MarianTokenizer:
    # loads tkizer if already created
    if os.path.exists(tkizer_save_path):
        print(f"Tokenizer loaded from {tkizer_save_path}")
        return AutoTokenizer.from_pretrained(tkizer_save_path)

    # supplies lists (batches) of sentences (ext : src/tgt)
    text_iterator = lambda: (batch[ext] for batch in dataset.shuffle().iter(batch_size=1000))

    # creates the tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    custom_tokenizer = pretrained_tokenizer.train_new_from_iterator(text_iterator(), vocab_size=vocab_size)

    # saves the tokenizer
    if tkizer_to_save:
        if not os.path.exists(tkizer_save_path):
            os.makedirs(tkizer_save_path)
        custom_tokenizer.save_pretrained(tkizer_save_path)

    return custom_tokenizer

# downloads HF dataset, saves it offline
def download_dataset():

    # get streaming dataset
    dataset = load_dataset("ai4bharat/samanantar",
        f"{config.target_lang}",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    return dataset
