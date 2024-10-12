import torch
from dataset import download_dataset, RawDataset, get_split_loaders, get_split_loaders
from model import build_tkizers
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm import tqdm
import psutil
import GPUtil
from torch.utils.data import DataLoader
from evaluate import load
import os


def get_data_model_tkizer():
    df_path = download_dataset()
    tkizers = build_tkizers(df_path)   # build tkizer from src/tgt vocabs
    # tkized_dataset = tkize_dataset(df_path, src_tkizer, tgt_tkizer) # tkized data
    raw_dataset = RawDataset(df_path)
    dataloaders = get_split_loaders(
            raw_dataset,
            tkizers
    )
    model = create_model()
    model.to(config.device)
    return model, dataloaders, tkizers

# performance details
def compute_metrics(eval_preds, tgt_tkizer) -> dict[str, float]:
    metric = load("sacrebleu")
    predictions = []
    references = []
    for batch in eval_preds:
        preds = batch["predictions"]
        labels = batch["labels"]
        decoded_preds = tgt_tkizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tgt_tkizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        predictions.extend(decoded_preds)
        references.extend(decoded_labels)
    result = metric.compute(predictions=predictions, references=references)
    return {"bleu": result["score"]}

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# system details
def get_system_metrics() -> dict[str, float]:
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

def evaluate(model, val_dataloader: DataLoader, tkizers: tuple, epoch: int, step: int, accelerator, best_eval_loss: float):
    model.eval()
    eval_loss = 0
    eval_preds = []
    for eval_batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Evaluation => Epoch {epoch + 1}, Step {step + 1}"):
        with torch.no_grad():
            eval_outputs = model(**eval_batch)
            eval_loss += eval_outputs.loss.item()

            logits = eval_outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            eval_preds.append({
                "predictions": predictions.detach().cpu(),
                "labels": eval_batch["labels"].detach().cpu()
            })

    # Compute metrics
    perf_metrics = compute_metrics(eval_preds, tkizers[1])
    
    # Calculate average loss
    avg_loss = eval_loss / len(val_dataloader)
    perf_metrics["val_loss"] = avg_loss
    
    # Log all metrics to wandb
    if config.use_wandb:
        wandb.log(perf_metrics)

    print(f"Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")

    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        accelerator.save(model.state_dict(), f"{config.output_dir}/best_model.pt")

    model.train()
    return best_eval_loss


def main():
    accelerator = Accelerator()

    # ====== LOAD DATA, TKIZER AND MODEL======= #
    model, dataloaders, tkizers = get_data_model_tkizer()
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    # ===== TRAINING PARAMS ======= #
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_train_epochs)
    )

    if accelerator.is_main_process:
        if config.use_wandb:
            wandb.init(project=config.wandb_project, entity=config.wandb_entity)
            wandb.config.update(config)

    # setup accelerate
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    ) # mixed precision training

    best_eval_loss = float('inf')
    accumulation_steps = 4 # Adjust based on your needs

    # ======== MAIN TRAINING LOOP ==========
    for epoch in range(config.num_train_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{config.num_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
            
            accelerator.backward(loss)
            total_loss += loss.item() * accumulation_steps
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": total_loss / (step + 1)})
            
            # ======= EVALUATE ============
            if (step+1) % 10 == 0:
                if config.use_wandb and accelerator.is_main_process:
                    system_metrics = get_system_metrics()
                    wandb.log(system_metrics)

            if (step + 1) % 1000 == 0:  # Evaluate less frequently
                model.eval()
                eval_loss = 0
                eval_preds = []
                for eval_batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Evaluation => Epoch {epoch + 1}, Step {step + 1}"):
                    with torch.no_grad():
                        eval_outputs = model(**eval_batch)
                        eval_loss += eval_outputs.loss.item()

                        logits = eval_outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        eval_preds.append({
                            "predictions": predictions.detach().cpu(),
                            "labels": eval_batch["labels"].detach().cpu()
                        })

                # Compute metrics
                perf_metrics = compute_metrics(eval_preds, tkizers[1])
                
                # Calculate average loss
                avg_loss = eval_loss / len(val_dataloader)
                perf_metrics["val_loss"] = avg_loss
                
                # Log all metrics to wandb
                if config.use_wandb and accelerator.is_main_process:
                    wandb.log(perf_metrics)

                print(f"Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")

                #print(eval_loss, best_eval_loss, eval_loss < best_eval_loss)
                if eval_loss < best_eval_loss:
                    #print("comes here")
                    best_eval_loss = eval_loss

                    os.makedirs(config.output_dir, exist_ok=True)
                    accelerator.save(accelerator.unwrap_model(model).state_dict(), f"{config.output_dir}/best_model.pt")


                model.train()   


    # Final test evaluation
    model.eval()
    test_loss = 0
    for test_batch in test_dataloader:
        with torch.no_grad():
            test_outputs = model(**test_batch)
            test_loss += test_outputs.loss.item()
    test_loss /= len(test_dataloader)
    print(f"Final Test Loss: {test_loss:.4f}")


if __name__ == '__main__':
    main()



# # def objective(trial):
# #     # Define the hyperparameters to optimize
# #     config = {
# #         "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
# #         "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-2),
# #         "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
# #     }
    
# #     # Initialize the model, dataloaders, etc. with the trial config
# #     model = initialize_model(config)
# #     train_dataloader, eval_dataloader = get_dataloaders(config["batch_size"], subset=True)
# #     trainer = Trainer(model, config, train_dataloader, eval_dataloader, tgt_tokenizer, metric)
    
# #     # Train the model for a fixed number of steps or epochs
# #     num_optimization_steps = 1000  # Adjust based on your needs
# #     best_eval_loss = trainer.train(max_steps=num_optimization_steps)
    
# #     return best_eval_loss

# # def hyperparameter_tuning():
# #     study = optuna.create_study(direction="minimize", 
# #                                 pruner=optuna.pruners.MedianPruner())
# #     study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])
    
# #     print("Best trial:")
# #     trial = study.best_trial
# #     print(f"  Value: {trial.value}")
# #     print("  Params: ")
# #     for key, value in trial.params.items():
# #         print(f"    {key}: {value}")
    
# #     # Log the best hyperparameters to W&B
# #     with wandb.init(project=wandb_kwargs["project"], entity=wandb_kwargs["entity"], job_type="hparam-tuning-results"):
# #         wandb.log({"best_eval_loss": trial.value, **trial.params})

# #     return study.best_params

               
#   # model.eval()
#                 # eval_loss = 0
#                 # total_steps = ( config.n_steps_per_epoch * config.val_split ) // config.train_split
#                 # for eval_batch in tqdm(val_dataloader, total=total_steps, desc=f"Evaluation => Epoch {epoch + 1}, Step {step + 1}"):
#                 #     with torch.no_grad():
#                 #         eval_outputs = model(**eval_batch)
#                 #         eval_loss += eval_outputs.loss.item()
#                 # eval_loss /= total_steps
#                 # print(f"Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")
#                 # model.train()
#             # =============================

#     #             model.eval()
#     #             eval_loss = 0
#     #             for eval_batch in val_dataloader:
#     #                 with torch.no_grad():
#     #                     eval_outputs = model(**eval_batch)
#     #                     eval_loss += eval_outputs.loss.item()
#     #             eval_loss /= len(val_dataloader)
#     #             print(f"Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")
                
#     #             if config.use_wandb:
#     #                 wandb.log({
#     #                     "epoch": epoch + 1,
#     #                     "step": step + 1,
#     #                     "eval_loss": eval_loss,
#     #                     "train_loss": total_loss / (step + 1)
#     #                 })
                
#     #             if eval_loss < best_eval_loss:
#     #                 best_eval_loss = eval_loss
#     #                 accelerator.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
                
#     #             model.train()
        
#     #     print(f"Epoch {epoch + 1}/{config.num_train_epochs} completed. Average Loss: {total_loss / len(train_dataloader):.4f}")

