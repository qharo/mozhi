from tracking import compute_metrics, WandbTrainer
from dataset import download_dataset, build_tokenizer, prepare_dataloaders, split_dataset
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
import psutil
import GPUtil

def main():
    # get dataset
    dataset = download_dataset()

    # get tokenizer from source and target vocabularies
    src_tkizer = build_tokenizer(dataset, 'src', 'data/src_tkizer', False, config.src_vocab_size)
    tgt_tkizer = build_tokenizer(dataset, 'tgt', 'data/tgt_tkizer', False, config.tgt_vocab_size)

    # split data and get corresponding dataloaders
    dataset_splits = split_dataset(dataset)
    dataloader_splits = prepare_dataloaders(dataset_splits, src_tkizer, tgt_tkizer, config.batch_size)
    train_dataloader = dataloader_splits['train']
    test_dataloader = dataloader_splits['test']

    # create model
    #
    # model = create_model()




    # if config.use_wandb:
    #     wandb.init(project=config.wandb_project, entity=config.wandb_entity)
    #     wandb.config.update(config)

    # training_args = TrainingArguments(
    #     output_dir=config.output_dir,
    #     num_train_epochs=config.num_train_epochs,
    #     per_device_train_batch_size=config.batch_size,
    #     per_device_eval_batch_size=config.batch_size,
    #     learning_rate=config.learning_rate,
    #     max_steps = config.n_steps,
    #     weight_decay=0.01,
    #     eval_strategy="steps",
    #     eval_steps=1000,
    #     save_strategy="steps",
    #     save_steps=1000,
    #     load_best_model_at_end=True,
    #     report_to="wandb" if config.use_wandb else None,
    # )

    # trainer = WandbTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tgt_tokenizer=tgt_tokenizer
    # )

    # accelerator = Accelerator()
    # trainer = accelerator.prepare(trainer)
    # trainer.train()

    if config.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
