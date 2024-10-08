from datasets import load_metric
import torch
import psutil
import GPUtil
from transformers import Trainer

def compute_metrics(eval_pred):
    metric = load_metric("sacrebleu")
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, axis=-1)

    # Remove ignored index (special tokens)
    predictions = [
        [tgt_tokenizer.decode([p]) for p in prediction if p != -100]
        for prediction in predictions
    ]
    references = [
        [[tgt_tokenizer.decode([l]) for l in label if l != -100]]
        for label in labels
    ]

    return metric.compute(predictions=predictions, references=references)

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
