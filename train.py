from tracking import WandbTrainer
from dataset import build_tkizers, download_dataset, tkize_dataset, build_tkizers, split_dataset
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import TrainingArguments

def main():
    # get dataset
    dataset = download_dataset()

    # get tokenizer from source and target vocabularies
    src_tkizer, tgt_tkizer = build_tkizers(dataset)

    # Apply the tokenization to the dataset
    tkized_dataset = tkize_dataset(dataset, src_tkizer, tgt_tkizer)

    # split data and get corresponding dataloaders
    dataset_splits = split_dataset(tkized_dataset)
    # print(type(dataset_splits))
    # dataloader_splits = prepare_dataloaders(dataset_splits, src_tkizer, tgt_tkizer)
    train_dataloader = dataset_splits['train']
    test_dataloader = dataset_splits['test']



if __name__ == '__main__':
    main()
