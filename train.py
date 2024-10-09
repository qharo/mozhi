from tracking import WandbTrainer
from dataset import build_tkizers, download_dataset, tkize_dataset, build_tkizers, prepare_dataloaders, split_dataset
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

    # create model
    model = create_model()

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity)
        wandb.config.update(config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else None,
    )

    trainer = WandbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=test_dataloader,
        tgt_tokenizer=tgt_tkizer
    )

    accelerator = Accelerator()
    trainer = accelerator.prepare(trainer)
    trainer.train()

    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
