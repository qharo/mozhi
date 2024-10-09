import torch
from transformers import Trainer
import wandb
import psutil
import GPUtil
from evaluate import load
from typing import Dict, List, Optional, Union, Tuple, Any
from torch.utils.data import DataLoader

class WandbTrainer(Trainer):
    def __init__(self, *args, tgt_tokenizer=None, use_wandb=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_interval = 100  # Log every 100 steps
        self.tgt_tokenizer = tgt_tokenizer
        self.eval_preds = []
        self.metric = load("sacrebleu")
        self.use_wandb = use_wandb

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        outputs = super().training_step(model, inputs)
        if self.use_wandb and self.state.global_step % self.log_interval == 0:
            system_metrics = self.get_system_metrics()
            wandb.log({
                'train/loss': outputs.loss.item(),
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/step': self.state.global_step,
                **system_metrics
            })
        return outputs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        self.eval_preds = []
        eval_losses = []
        self.model.eval()

        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs.loss
            eval_losses.append(loss.item())

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            self.eval_preds.append({
                "predictions": predictions.detach().cpu(),
                "labels": inputs["labels"].detach().cpu()
            })

            if step % 10 == 0:
                self.log({"eval_loss": sum(eval_losses) / len(eval_losses)})

        metrics = self.compute_metrics(self.eval_preds)
        metrics["loss"] = sum(eval_losses) / len(eval_losses)

        for key, value in metrics.items():
            self.log({f"{metric_key_prefix}_{key}": value})

        return metrics

    def log(self, logs: Dict[str, float]) -> None:
        if self.use_wandb:
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)
            output = {**logs, **self.get_system_metrics()}
            wandb.log(output)

    def compute_metrics(self, eval_preds: List[Dict]) -> Dict[str, float]:
        predictions = []
        references = []
        for batch in eval_preds:
            preds = batch["predictions"]
            labels = batch["labels"]
            decoded_preds = self.tgt_tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tgt_tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
        result = self.metric.compute(predictions=predictions, references=references)
        return {"bleu": result["score"]}

    @staticmethod
    def get_system_metrics() -> Dict[str, float]:
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

    @staticmethod
    def postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[List[str]]]:
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
