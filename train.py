import torch
from dataset import get_dataset, tkize_dataset, get_split_loaders
from model import build_tkizers
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import TrainingArguments, Trainer, get_scheduler
from tqdm import tqdm

def main():
    # setup accelerate
    accelerator = Accelerator()

    # ====== LOAD DATA, TKIZER AND MODEL======= #
    dataset = get_dataset()
    src_tkizer, tgt_tkizer = build_tkizers(dataset)   # build tkizer from src/tgt vocabs
    tkized_dataset = tkize_dataset(dataset, src_tkizer, tgt_tkizer) # tkized data
    train_dataloader, eval_dataloader, test_dataloader = get_split_loaders(
            tkized_dataset,
            train_size=0.8,
            val_size=0.01,
            test_size=0.19,
    )
    config.pad_token_id = src_tkizer.pad_token_id
    model = create_model()
    model.to(config.device)
    # ======================== #


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # Prepare for mixed precision training
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity)
        wandb.config.update(config)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.n_steps
    )

    # if config.use_wandb:
    #     wandb.init(project="t5_small_normal", config=config)

    best_eval_loss = float('inf')
    
    progress_bar = tqdm(range(config.n_steps))
    accumulation_steps = 4  # Adjust based on your needs

    for step in range(config.n_steps):
        model.train()
        for i, batch in enumerate(train_dataloader):
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps

            accelerator.backward(loss)
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            
            if (step + 1) % 1000 == 0:  # Evaluate less frequently
                model.eval()
                eval_loss = 0
                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        eval_outputs = model(**eval_batch)
                        eval_loss += eval_outputs.loss.item()
                eval_loss /= len(eval_dataloader)
                
                print(f"Step {step+1}: Eval Loss: {eval_loss:.4f}")
                if config.use_wandb:
                    wandb.log({"eval_loss": eval_loss, "train_loss": loss.item() * accumulation_steps})
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    accelerator.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
                
                model.train()

    if config.use_wandb:
        wandb.finish()


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
