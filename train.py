import torch
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from accelerate import Accelerator
from model import create_model
from dataset import load_and_process_data
from config import config
import wandb
import psutil
import GPUtil
import time

def compute_metrics(eval_pred):
    metric = load_metric("sacrebleu")
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, axis=-1)

    # Remove ignored index (special tokens)
    predictions = [
        [config.id2label[p] for p in prediction if p != -100]
        for prediction in predictions
    ]
    references = [
        [[config.id2label[l] for l in label if l != -100]]
        for label in labels
    ]

    return metric.compute(predictions=predictions, references=references)

def get_system_metrics():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    gpu_metrics = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_metrics[f'gpu_{i}_usage'] = gpu.load * 100
            gpu_metrics[f'gpu_{i}_memory'] = gpu.memoryUtil * 100
    except:
        # If GPUtil fails or no GPU is available
        gpu_metrics['gpu_0_usage'] = 0
        gpu_metrics['gpu_0_memory'] = 0

    return {
        'cpu_usage': cpu_percent,
        'memory_usage': memory_percent,
        **gpu_metrics
    }

class WandbTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_interval = 100  # Log every 100 steps

    def training_step(self, model, inputs):
        outputs = super().training_step(model, inputs)

        if self.state.global_step % self.log_interval == 0:
            system_metrics = get_system_metrics()
            wandb.log({
                'train/loss': outputs.loss.item(),
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/step': self.state.global_step,
                **system_metrics
            })

        return outputs

    def evaluate(self, *args, **kwargs):
        output = super().evaluate(*args, **kwargs)

        metrics = output.metrics
        system_metrics = get_system_metrics()

        wandb.log({
            'eval/loss': metrics['eval_loss'],
            'eval/bleu': metrics['eval_bleu'],
            'eval/step': self.state.global_step,
            **system_metrics
        })

        return output

def main():
    accelerator = Accelerator()
    model = create_model()
    tokenized_datasets = load_and_process_data()

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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else None,
    )

    trainer = WandbTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    trainer = accelerator.prepare(trainer)
    trainer.train()

    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
