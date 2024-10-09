import torch
from tracking import WandbTrainer
from dataset import build_tkizers, download_dataset, tkize_dataset, build_tkizers, split_iterable_dataset, debug_dataset
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
    debug_dataset(tkized_dataset)

    # split data and get corresponding dataloaders
    dataset_splits = split_iterable_dataset(tkized_dataset)
    train_dataset = dataset_splits['train']
    test_dataset = dataset_splits['test']

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
        max_steps = config.n_steps,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else None,
    )

    trainer = WandbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tgt_tokenizer=tgt_tkizer
    )



    # Debugging: Check the first batch
    for batch in trainer.get_train_dataloader():
        print(f"Dataloader output shapes: {', '.join(f'{k}: {v.shape}' for k, v in batch.items())}")
        break
    
    # Debugging: Check model input
    model = trainer.model
    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
        print(f"Model output keys: {outputs.keys()}")
        print(f"Loss: {outputs.loss}")





    accelerator = Accelerator()
    trainer = accelerator.prepare(trainer)
    trainer.train()

if __name__ == '__main__':
    main()
