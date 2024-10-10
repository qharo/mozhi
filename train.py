import torch
from tracking import WandbTrainer
from dataset import get_iterable_dataset, tkize_dataset, split_dataset, MultiGPUIterableDataset
from model import build_tkizers
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import TrainingArguments, Trainer

def main():

    # using accelerate
    accelerator = Accelerator()

    # get dataset
    dataset = get_iterable_dataset()

    # get tokenizer from source and target vocabularies
    src_tkizer, tgt_tkizer = build_tkizers(dataset)

    # print(TokenizedDataset(i_dataset, src_tkizer, tgt_tkizer))

    # Apply the tokenization to the dataset
    tkized_dataset = tkize_dataset(dataset, src_tkizer, tgt_tkizer)    
    multi_gpu_dataset = MultiGPUIterableDataset(tkized_dataset, accelerator)
    

    # split data and get corresponding dataloaders
    dataset_splits = split_dataset(multi_gpu_dataset)
    train_dataset = dataset_splits['train']
    test_dataset = dataset_splits['test']

    config.pad_token_id = src_tkizer.pad_token_id
    model = create_model()
    print(f"{config.model_name} created")

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity)
        wandb.config.update(config)

    # training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        #gradient_accumulation_steps=4,
        fp16=True,
        learning_rate=config.learning_rate,
        max_steps = config.n_steps,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        run_name="t5_small_normal",
        report_to="wandb" if config.use_wandb else None,
    )

    # allows tracking with wandb
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tgt_tkizer,
    )

    trainer = accelerator.prepare(trainer)
    trainer.train()


if __name__ == '__main__':
    main()
