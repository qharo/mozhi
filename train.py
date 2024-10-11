import torch
from tracking import WandbTrainer
from dataset import get_iterable_dataset, tkize_dataset, split_dataset, DataLoader
from model import build_tkizers
from config import config
import wandb
from model import create_model
from accelerate import Accelerator
from transformers import TrainingArguments, Trainer, get_scheduler
from tqdm import tqdm

def main():

    # using accelerate
    accelerator = Accelerator()

    # get dataset
    dataset = get_iterable_dataset()

    # get tokenizer from source and target vocabularies
    src_tkizer, tgt_tkizer = build_tkizers(dataset)

    # print(TokenizedDataset(i_dataset, src_tkizer, tgt_tkizer))

    # Apply the tokenization to the dataset
    tkized_dataset = tkize_dataset(dataset, src_tkizer, tgt_tkizer)    
    # multi_gpu_dataset = MultiGPUIterableDataset(tkized_dataset, accelerator)
    

    # split data and get corresponding dataloaders
    dataset_splits = split_dataset(tkized_dataset)
    train_dataset = dataset_splits['train']
    test_dataset = dataset_splits['val']


    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=1, prefetch_factor=2)
    eval_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=1, prefetch_factor=2)


    config.pad_token_id = src_tkizer.pad_token_id
    model = create_model()
    model.to(config.device)


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

    if config.use_wandb:
        wandb.init(project="t5_small_normal", config=config)

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



if __name__ == '__main__':
    main()
