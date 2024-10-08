from tracking import compute_metrics, WandbTrainer
from dataset import download_dataset, build_tokenizer, split_dataset
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

    # create model
    model = create_model()

    # get tokenizer from source and target vocabularies
    #src_tkizer = build_tokenizer(dataset, 'src', 'data/src_tkizer', False, 3000)
    #tgt_tkizer = build_tokenizer(dataset, 'tgt', 'data/tgt_tkizer', False, 3000)

    dataset_splits = split_dataset(dataset) 
    train_dataset = dataset_splits['train']
    eval_dataset = dataset_splits['val']

    # setup wandb
    # if config.use_wandb:
    #     wandb.init(project=config.wandb_project, entity=config.wandb_entity)
    #     wandb.config.update(config)


    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps",
        max_steps = config.n_steps,
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else None,
    )

    trainer = WandbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )



if __name__ == '__main__':
    main()
