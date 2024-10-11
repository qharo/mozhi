import torch
from dataset import get_dataset, tkize_dataset, get_split_loaders
from model import build_tkizers
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import TrainingArguments, Trainer, get_scheduler
from tqdm import tqdm


def get_data_model_tkizer():
    dataset = get_dataset()
    src_tkizer, tgt_tkizer = build_tkizers(dataset)   # build tkizer from src/tgt vocabs
    tkized_dataset = tkize_dataset(dataset, src_tkizer, tgt_tkizer) # tkized data
    dataloaders = get_split_loaders(
            tkized_dataset,
    )
    model = create_model()
    model.to(config.device)
    return model, dataloaders

def main():
    # setup accelerate
    accelerator = Accelerator()

    # ====== LOAD DATA, TKIZER AND MODEL======= #
    model, dataloaders = get_data_model_tkizer()
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    # ===== TRAINING ======= #
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.n_steps
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    ) # mixed precision training

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity)
        wandb.config.update(config)

    best_eval_loss = float('inf')
    accumulation_steps = 4 # Adjust based on your needs

    for epoch in range(config.num_train_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=config.n_steps_per_epoch, desc=f"Epoch {epoch + 1}/{config.num_train_epochs}")
        
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
            
            if (step + 1) % 100 == 0:  # Evaluate less frequently
                model.eval()
                eval_loss = 0
                for eval_batch in val_dataloader:
                    with torch.no_grad():
                        eval_outputs = model(**eval_batch)
                        eval_loss += eval_outputs.loss.item()
                eval_loss /= len(val_dataloader)
                print(f"Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")
                
                if config.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss / (step + 1)
                    })
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    accelerator.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
                
                model.train()
        
        print(f"Epoch {epoch + 1}/{config.num_train_epochs} completed. Average Loss: {total_loss / len(train_dataloader):.4f}")



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
