import torch
from dataset import get_dataset, RawDataset, get_split_loaders, split_dataloader
from model import build_tkizers
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import TrainingArguments, Trainer, get_scheduler
from tqdm import tqdm
import optuna
from torch.utils.data import DataLoader


def get_data_model_tkizer():
    df_path = get_dataset()
    tkizers = build_tkizers(df_path)   # build tkizer from src/tgt vocabs
    # tkized_dataset = tkize_dataset(df_path, src_tkizer, tgt_tkizer) # tkized data
    raw_dataset = RawDataset(df_path)
    dataloaders = split_dataloader(
            raw_dataset,
            tkizers
    )
    model = create_model()
    model.to(config.device)
    return model, dataloaders, tkizers

def evaluate(model, val_dataloader: DataLoader, epoch: int, step: int):
    model.eval()
    eval_loss = 0
    for eval_batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Evaluation => Epoch {epoch + 1}, Step {step + 1}"):
        with torch.no_grad():
            eval_outputs = model(**eval_batch)
            eval_loss += eval_outputs.loss.item()
    eval_loss /= len(val_dataloader)
    print(f"Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")
    model.train()
    


def main():
    # setup accelerate
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

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    ) # mixed precision training

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity)
        wandb.config.update(config)

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
            if (step + 1) % 100 == 0:  # Evaluate less frequently
                evaluate(model, val_dataloader, epoch, step)
#                 # model.eval()
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



#     # # Final test evaluation
#     # model.eval()
#     # test_loss = 0
#     # for test_batch in test_dataloader:
#     #     with torch.no_grad():
#     #         test_outputs = model(**test_batch)
#     #         test_loss += test_outputs.loss.item()
#     # test_loss /= len(test_dataloader)
#     # print(f"Final Test Loss: {test_loss:.4f}")


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