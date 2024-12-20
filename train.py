import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import psutil
import GPUtil
from evaluate import load
import os
import wandb

from dataset import download_dataset, get_tkized_dataloaders
from model import build_tkizer, create_model
from config import config

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def init_wandb(rank):
    """Initialize wandb with proper error handling"""
    if rank == 0 and config.use_wandb:
        try:
            # Check if WANDB_API_KEY is set in environment
            api_key = os.getenv('WANDB_API_KEY')
            if api_key is None:
                print("Warning: WANDB_API_KEY not found in environment. Disabling wandb logging.")
                config.use_wandb = False
                return False
            
            wandb.login(key=api_key)
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.__dict__
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {str(e)}. Disabling wandb logging.")
            config.use_wandb = False
            return False
    return False

def collate_fn(batch):
    """
    Collate function to create batches
    """
    return {
        k: torch.stack([example[k] for example in batch])
        for k in batch[0].keys()
    }

def get_data_model_tkizer(rank, world_size):
    df_path = download_dataset()
    tkizer = build_tkizer(df_path)
    datasets = get_tkized_dataloaders(df_path, tkizer)
    
    # Modify dataloaders to use DistributedSampler
    train_sampler = DistributedSampler(datasets[0], num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(datasets[1], num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(datasets[2], num_replicas=world_size, rank=rank)
    

    train_dataloader = torch.utils.data.DataLoader(
        datasets[0], sampler=train_sampler, batch_size=config.batch_size, num_workers=1, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        datasets[1], sampler=val_sampler, batch_size=config.batch_size, num_workers=1, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets[2], sampler=test_sampler, batch_size=config.batch_size, num_workers=1, collate_fn=collate_fn
    )
    
    model = create_model()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    return model, (train_dataloader, val_dataloader, test_dataloader), tkizer

# BLEU, Validation Loss
def log_perf_metrics(eval_preds, tgt_tkizer, val_loss) -> dict[str, float]:
    metric = load("sacrebleu")
    predictions = []
    references = []
    for batch in eval_preds:
        preds = batch["predictions"]
        labels = batch["labels"]
        decoded_preds = tgt_tkizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tgt_tkizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        predictions.extend(decoded_preds)
        references.extend(decoded_labels)
    result = {"bleu": metric.compute(predictions=predictions, references=references)["score"]}
    result["val_loss"] = val_loss
    wandb.log(result)


# CPU Usage, GPU Usage, Memory, etc
def log_system_metrics(total_loss):    
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
    system_metrics = {
        'cpu_usage': cpu_percent,
        'memory_usage': memory_percent,
        **gpu_metrics
    }
    system_metrics['training_loss'] = total_loss
    wandb.log(system_metrics)

def main(rank, world_size):
    setup(rank, world_size)
    
    model, dataloaders, tkizer = get_data_model_tkizer(rank, world_size)
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    
    wandb_enabled = init_wandb(rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_train_epochs)
    )

    start_epoch, start_step = 0, 0
    checkpoint_path = f"{config.output_dir}/checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only=True)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch, start_step = checkpoint['epoch'], checkpoint['step']

    for n_epoch in range(start_epoch, config.num_train_epochs):
        train_dataloader.sampler.set_epoch(n_epoch)
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {n_epoch + 1}/{config.num_train_epochs}", disable=rank != 0)
        
        best_eval_loss = float('inf')
        for n_step, batch in enumerate(train_dataloader):
            if n_step < start_step:
                progress_bar.update(1)
                continue
        
            batch = {k: v.to(rank) for k, v in batch.items()}

            model.train()
            outputs = model(**batch)
            loss = outputs.loss / config.accumulation_steps
            
            loss.backward()
            total_loss += loss.item() * config.accumulation_steps
            
            if (n_step + 1) % config.accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": total_loss / (n_step + 1)})

                if (n_step+1) % 10 == 0 and wandb_enabled:
                    log_system_metrics(loss)

            if (n_step + 1) % config.eval_save_steps == 0: 
                model.eval()
                eval_loss = 0
                eval_preds = []
                for eval_batch in val_dataloader:
                    eval_batch = {k: v.to(rank) for k, v in eval_batch.items()}
                    with torch.no_grad():
                        eval_outputs = model(**eval_batch)
                        eval_loss += eval_outputs.loss.item()
                        logits = eval_outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        eval_preds.append({
                            "predictions": predictions.detach().cpu(),
                            "labels": eval_batch["labels"].detach().cpu()
                        })

                eval_loss = eval_loss / len(val_dataloader)
                dist.all_reduce(torch.tensor(eval_loss).to(rank))
                eval_loss = eval_loss / world_size

                if rank == 0:
                    if wandb_enabled:
                        log_perf_metrics(eval_preds, tkizer, eval_loss)

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        os.makedirs(config.output_dir, exist_ok=True)    
                        checkpoint = {
                            'epoch': n_epoch,
                            'step': n_step,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_eval_loss': best_eval_loss
                        }
                        torch.save(checkpoint, f"{config.output_dir}/checkpoint.pt")

        start_step = 0

    model.eval()
    test_loss = 0
    for test_batch in test_dataloader:
        test_batch = {k: v.to(rank) for k, v in test_batch.items()}
        with torch.no_grad():
            test_outputs = model(**test_batch)
            test_loss += test_outputs.loss.item()
    test_loss /= len(test_dataloader)
    dist.all_reduce(torch.tensor(test_loss).to(rank))
    test_loss = test_loss / world_size

    if rank == 0:
        print(f"Final Test Loss: {test_loss:.4f}")

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)











# import torch
# from dataset import download_dataset, get_split_sets
# from model import build_tkizers
# from config import config
# import wandb
# from model import create_model
# from accelerate import Accelerator
from transformers import get_scheduler, Seq2SeqTrainer, Seq2SeqTrainingArguments
# from datasets import Dataset
# from tqdm import tqdm
# import psutil
# import GPUtil
# from torch.utils.data import DataLoader
# from functools import partial
# from evaluate import load
# import os

# import torch
# from dataset import download_dataset, get_tkized_dataloaders
# from model import build_tkizer
# from config import config
# import wandb
# from model import create_model
# from accelerate import Accelerator
# from transformers import get_scheduler
# from tqdm import tqdm
# import psutil
# import GPUtil
# from torch.utils.data import DataLoader
# from evaluate import load
# import os

# # ======== SETUP ========= #
# # get model, dataloaders and tokenizer
# def get_data_model_tkizer():
#     df_path = download_dataset()
#     tkizer = build_tkizer(df_path)   # build tkizer from src/tgt vocabs
#     dataloaders = get_tkized_dataloaders(df_path, tkizer)
#     model = create_model()
#     model.to(config.device)
#     return model, dataloaders, tkizer

# # BLEU, Validation Loss
# def log_perf_metrics(eval_preds, tgt_tkizer, val_loss) -> dict[str, float]:
#     metric = load("sacrebleu")
#     predictions = []
#     references = []
#     for batch in eval_preds:
#         preds = batch["predictions"]
#         labels = batch["labels"]
#         decoded_preds = tgt_tkizer.batch_decode(preds, skip_special_tokens=True)
#         decoded_labels = tgt_tkizer.batch_decode(labels, skip_special_tokens=True)
#         decoded_preds = [pred.strip() for pred in decoded_preds]
#         decoded_labels = [[label.strip()] for label in decoded_labels]
#         predictions.extend(decoded_preds)
#         references.extend(decoded_labels)
#     result = {"bleu": metric.compute(predictions=predictions, references=references)["score"]}
#     result["val_loss"] = val_loss
#     wandb.log(result)


# # CPU Usage, GPU Usage, Memory, etc
# def log_system_metrics(total_loss):    
#     cpu_percent = psutil.cpu_percent()
#     memory_percent = psutil.virtual_memory().percent
#     gpu_metrics = {}
#     try:
#         gpus = GPUtil.getGPUs()
#         for i, gpu in enumerate(gpus):
#             gpu_metrics[f'gpu_{i}_usage'] = gpu.load * 100
#             gpu_metrics[f'gpu_{i}_memory'] = gpu.memoryUtil * 100
#     except:
#         # If GPUtil fails or no GPU is available
#         gpu_metrics['gpu_0_usage'] = 0
#         gpu_metrics['gpu_0_memory'] = 0
#     system_metrics = {
#         'cpu_usage': cpu_percent,
#         'memory_usage': memory_percent,
#         **gpu_metrics
#     }
#     system_metrics['training_loss'] = total_loss
#     wandb.log(system_metrics)



# def main():
#     accelerator = Accelerator()

#     # ====== LOAD DATA, TKIZER AND MODEL======= #
#     model, dataloaders, tkizer = get_data_model_tkizer()
#     train_dataloader, val_dataloader, test_dataloader = dataloaders


#     # ===== WANDB CONFIGURATION ===== #
#     if config.use_wandb and accelerator.is_main_process:
#         wandb.init(project=config.wandb_project, entity=config.wandb_entity)
#         wandb.config.update(config)



#     start_epoch, start_step = 0, 0   
#     checkpoint_path = f"{config.output_dir}/checkpoint.pt"
#     if os.path.exists(checkpoint_path):
#         checkpoint = accelerator.load(checkpoint_path)
#         accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
#         start_epoch, start_step = checkpoint['epoch'], checkpoint['step']


#     # ===== TRAINING PARAMS ======= #
#     optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#     lr_scheduler = get_scheduler(
#         "linear",
#         optimizer=optimizer,
#         num_warmup_steps=config.warmup_steps,
#         num_training_steps=(len(train_dataloader) * config.num_train_epochs)
#     )

#     # ======= WANDB ========= #
#     if accelerator.is_main_process:
#         if config.use_wandb:
#             wandb.init(project=config.wandb_project, entity=config.wandb_entity)
#             wandb.config.update(config)

#     # preparing using accelerate // mixed precision training
#     model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
#         model, optimizer, train_dataloader, val_dataloader
#     ) 

#     start_epoch, start_step = 0, 0
#     checkpoint_path = f"{config.output_dir}/checkpoint.pt"
#     if os.path.exists(checkpoint_path):
#         checkpoint = accelerator.load(checkpoint_path)
#         accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
#         start_epoch, start_step = checkpoint['epoch'], checkpoint['step']

#     # === TRAINING LOOP ===
#     for n_epoch in range(config.num_train_epochs):

#         # === EPOCH LOOP ===

#         total_loss = 0
#         progress_bar = tqdm(total=len(dataloaders[0]), desc=f"Epoch {n_epoch + 1}/{config.num_train_epochs}")
        
#         best_eval_loss = float('inf')
#         for n_step, batch in enumerate(dataloaders[0]):
#             if n_step < start_step:
#                 continue
            
#             # Move batch to the device
#             batch = {k: v.to(accelerator.device) for k, v in batch.items()}

#             with accelerator.autocast():
#                 outputs = model(**batch)
#                 loss = outputs.loss / config.accumulation_steps
            
#             accelerator.backward(loss)
#             total_loss += loss.item() * config.accumulation_steps
            
#             if (n_step + 1) % config.accumulation_steps == 0:
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
            
#             progress_bar.update(1)
#             progress_bar.set_postfix({"Loss": total_loss / (n_step + 1)})

#             if (n_step+1) % 10 == 0:
#                 if config.use_wandb and accelerator.is_main_process:
#                     log_system_metrics(total_loss)


#             # ===================== EVALUATE ==================== #
#             # perform validation, log performance metrics and save model checkpoint
#             if (n_step + 1) % 100 == 0: 
#                 model.eval()

#                 # evaluate batch
#                 eval_loss = 0
#                 eval_preds = []
#                 model.to(accelerator.device)
#                 for eval_batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Evaluation => Epoch {n_epoch + 1}, Step {n_step + 1}"):
#                     batch = {k: v.to(accelerator.device) for k, v in eval_batch.items()}

#                     with torch.no_grad():
#                         eval_outputs = model(**eval_batch)
#                         eval_loss += eval_outputs.loss.item()

#                         logits = eval_outputs.logits
#                         predictions = torch.argmax(logits, dim=-1)
#                         eval_preds.append({
#                             "predictions": predictions.detach().cpu(),
#                             "labels": eval_batch["labels"].detach().cpu()
#                         })

#                 # log performance metrics
#                 if config.use_wandb and accelerator.is_main_process:
#                     log_perf_metrics(eval_preds, tkizer, (eval_loss / len(val_dataloader)))

#                 # save checkpoint
#                 if eval_loss < best_eval_loss:
#                     best_eval_loss = eval_loss
#                     os.makedirs(config.output_dir, exist_ok=True)    
#                     checkpoint = {
#                         'epoch': n_epoch,
#                         'step': n_step,
#                         'model_state_dict': accelerator.unwrap_model(model).state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
#                         'best_eval_loss': best_eval_loss
#                     }
#                     accelerator.save(checkpoint, f"{config.output_dir}/checkpoint.pt")

#                 model.train() 
#                 # === END EVALUATE === #

#         start_step = 0
#     # === END TRAIN LOOP === #


#     # ==== FINAL TEST ==== #
#     model.eval()
#     test_loss = 0
#     for test_batch in test_dataloader:
#         with torch.no_grad():
#             test_outputs = model(**test_batch)
#             test_loss += test_outputs.loss.item()
#     test_loss /= len(test_dataloader)
#     print(f"Final Test Loss: {test_loss:.4f}")


# if __name__ == '__main__':
#     main()

    # ===== DATA COLLATOR ===== #
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tkizers[0],  # Assuming the first tokenizer is source tokenizer
    #     model=model,
    #     padding="longest"
    # )

 # ===== TRAINING ARGS ===== #
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=config.output_dir,
    #     evaluation_strategy="steps",
    #     eval_steps=config.eval_save_steps,  # Evaluate every 100 steps
    #     save_steps=config.eval_save_steps,  # Save checkpoint every 100 steps
    #     logging_steps=10,
    #     learning_rate=config.learning_rate,
    #     per_device_train_batch_size=config.batch_size,
    #     per_device_eval_batch_size=config.batch_size,
    #     weight_decay=config.weight_decay,
    #     save_total_limit=3,  # Max number of checkpoints to keep
    #     num_train_epochs=config.num_train_epochs,
    #     predict_with_generate=True,  # For seq2seq models
    #     warmup_steps=config.warmup_steps,
    #     gradient_accumulation_steps=config.accumulation_steps,
    #     fp16=True,
    #     report_to="wandb" if config.use_wandb else None,
    #     load_best_model_at_end=True
    # )


# # ======== SETUP ========= #
# def get_data_model_tkizer():
#     raw_dataset, df_path = download_dataset()
#     tkizers = build_tkizers(df_path)   # build tkizer from src/tgt vocabs
#     datasets = get_split_sets(
#             raw_dataset,
#             tkizers
#     )
#     model = create_model()
#     model.to(config.device)
#     return model, datasets, tkizers


# def compute_metrics(tgt_tkizer, eval_preds):
#     metric = evaluate.load("sacrebleu")
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
    
#     decoded_preds = tgt_tkizer.batch_decode(preds, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tgt_tkizer.pad_token_id)
#     decoded_labels = tgt_tkizer.batch_decode(labels, skip_special_tokens=True)

#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [[label.strip()] for label in decoded_labels]

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     return {"bleu": result["score"]}

# def main():
#     accelerator = Accelerator()

#     # ====== LOAD DATA, TKIZER AND MODEL======= #
#     model, datasets, tkizers = get_data_model_tkizer()
#     train_dataset, val_dataset, test_dataset = datasets
#     partial_compute_metrics = partial(compute_metrics, tkizers[1])

#     training_args = Seq2SeqTrainingArguments(
#         output_dir=config.output_dir,
#         num_train_epochs=config.num_train_epochs,
#         per_device_train_batch_size=config.batch_size,
#         per_device_eval_batch_size=config.batch_size,
#         warmup_steps=config.warmup_steps,
#         weight_decay=config.weight_decay,
#         logging_dir=f"{config.output_dir}/logs",
#         logging_steps=10,
#         evaluation_strategy="steps",
#         eval_steps=config.eval_save_steps,
#         save_steps=config.eval_save_steps,
#         save_total_limit=3,
#         load_best_model_at_end=True,
#         metric_for_best_model="bleu",
#         greater_is_better=True,
#         fp16=True,  # Enable mixed precision training
#         report_to="wandb" if config.use_wandb else None,
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tkizers[1],
#         compute_metrics=partial_compute_metrics
#     )

#     if config.use_wandb:
#         wandb.init(project=config.wandb_project, entity=config.wandb_entity)
#         wandb.config.update(config)

#     # Check if there's a checkpoint to resume from
#     if os.path.exists(f"{config.output_dir}/checkpoint-latest"):
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         trainer.train()

#     # Evaluate on the test set
#     test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
#     print(f"Final Test Results: {test_results}")

#     # Save the final model
#     trainer.save_model(f"{config.output_dir}/final_model")

# if __name__ == '__main__':
#     main()





    # # ===== TRAINING PARAMS ======= #
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=config.warmup_steps,
    #     num_training_steps=(len(train_dataloader) * config.num_train_epochs)
    # )

    # # ======= WANDB ========= #
    # if accelerator.is_main_process:
    #     if config.use_wandb:
    #         wandb.init(project=config.wandb_project, entity=config.wandb_entity)
    #         wandb.config.update(config)

    # # preparing using accelerate // mixed precision training
    # model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    #     model, optimizer, train_dataloader, val_dataloader
    # ) 

    # start_epoch, start_step = load_checkpoint(accelerator, model, optimizer, lr_scheduler)

    # for n_epoch in range(config.num_train_epochs):
    #     train_epoch(accelerator, model, dataloaders, optimizer, lr_scheduler, n_epoch, start_step if n_epoch == start_epoch else 0)
    #     start_step = 0

    # model.eval()
    # test_loss = 0
    # for test_batch in test_dataloader:
    #     with torch.no_grad():
    #         test_outputs = model(**test_batch)
    #         test_loss += test_outputs.loss.item()
    # test_loss /= len(test_dataloader)
    # print(f"Final Test Loss: {test_loss:.4f}")





# def load_checkpoint(accelerator, model, optimizer, lr_scheduler):
#     checkpoint_path = f"{config.output_dir}/checkpoint.pt"
#     if os.path.exists(checkpoint_path):
#         checkpoint = accelerator.load(checkpoint_path)
#         accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
#         return checkpoint['epoch'], checkpoint['step']
#     return 0, 0

# def save_checkpoint(accelerator, model, optimizer, lr_scheduler, epoch, step, best_eval_loss):
#     checkpoint = {
#         'epoch': epoch,
#         'step': step,
#         'model_state_dict': accelerator.unwrap_model(model).state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'lr_scheduler_state_dict': lr_scheduler.state_dict(),
#         'best_eval_loss': best_eval_loss
#     }
#     accelerator.save(checkpoint, f"{config.output_dir}/checkpoint.pt")

# # ========= EVALUATE ========= #
# def log_perf_metrics(eval_preds, tgt_tkizer, val_loss) -> dict[str, float]:
#     metric = load("sacrebleu")
#     predictions = []
#     references = []
#     for batch in eval_preds:
#         preds = batch["predictions"]
#         labels = batch["labels"]
#         decoded_preds = tgt_tkizer.batch_decode(preds, skip_special_tokens=True)
#         decoded_labels = tgt_tkizer.batch_decode(labels, skip_special_tokens=True)
#         decoded_preds = [pred.strip() for pred in preds]
#         decoded_labels = [[label.strip()] for label in labels]
#         predictions.extend(decoded_preds)
#         references.extend(decoded_labels)
#     result = {"bleu": metric.compute(predictions=predictions, references=references)["score"]}
#     result["val_loss"] = val_loss
#     wandb.log(result)


# def log_system_metrics(total_loss):    
#     cpu_percent = psutil.cpu_percent()
#     memory_percent = psutil.virtual_memory().percent
#     gpu_metrics = {}
#     try:
#         gpus = GPUtil.getGPUs()
#         for i, gpu in enumerate(gpus):
#             gpu_metrics[f'gpu_{i}_usage'] = gpu.load * 100
#             gpu_metrics[f'gpu_{i}_memory'] = gpu.memoryUtil * 100
#     except:
#         # If GPUtil fails or no GPU is available
#         gpu_metrics['gpu_0_usage'] = 0
#         gpu_metrics['gpu_0_memory'] = 0
#     system_metrics = {
#         'cpu_usage': cpu_percent,
#         'memory_usage': memory_percent,
#         **gpu_metrics
#     }
#     system_metrics['training_loss'] = total_loss
#     wandb.log(system_metrics)


# def evaluate(accelerator, model, val_dataloader: DataLoader, tkizers: tuple, epoch: int, step: int,  best_eval_loss: float):
#     model.eval()

#     # evaluate batch
#     eval_loss = 0
#     eval_preds = []
#     for eval_batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Evaluation => Epoch {epoch + 1}, Step {step + 1}"):
#         with torch.no_grad():
#             eval_outputs = model(**eval_batch)
#             eval_loss += eval_outputs.loss.item()

#             logits = eval_outputs.logits
#             predictions = torch.argmax(logits, dim=-1)
#             eval_preds.append({
#                 "predictions": predictions.detach().cpu(),
#                 "labels": eval_batch["labels"].detach().cpu()
#             })

#     if config.use_wandb and accelerator.is_main_process:
#         log_perf_metrics(eval_preds, tkizers[1], (eval_loss / len(val_dataloader)))

#     print(f"Process {accelerator.process_index} || Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")

#     if eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         os.makedirs(config.output_dir, exist_ok=True)
#         accelerator.save(accelerator.unwrap_model(model).state_dict(), f"{config.output_dir}/best_model.pt")


#     model.train() 


# # ========== TRAIN =========== #
# def train_epoch(accelerator, model, dataloaders, tkizers, optimizer, lr_scheduler, n_epoch, start_step=0):
#     total_loss = 0
#     progress_bar = tqdm(total=len(dataloaders[0]), desc=f"Epoch {n_epoch + 1}/{config.num_train_epochs}")
    
#     best_eval_loss = float('inf')
#     for n_step, batch in enumerate(dataloaders[0]):
#         if n_step < start_step:
#             continue
        
#         with accelerator.autocast():
#             outputs = model(**batch)
#             loss = outputs.loss / config.accumulation_steps
        
#         accelerator.backward(loss)
#         total_loss += loss.item() * config.accumulation_steps
        
#         if (n_step + 1) % config.accumulation_steps == 0:
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
        
#         progress_bar.update(1)
#         progress_bar.set_postfix({"Loss": total_loss / (n_step + 1)})

#         if (n_step+1) % 10 == 0:
#             if config.use_wandb and accelerator.is_main_process:
#                 log_system_metrics(total_loss)

#         # log performance metrics
#         if (n_step + 1) % 100 == 0:  # Evaluate less frequently
#             evaluate(accelerator, model, dataloaders[1], tkizers, n_epoch, n_step, best_eval_loss)
#             save_checkpoint(accelerator, model, optimizer, lr_scheduler, n_epoch, n_step, best_eval_loss)



# def train(accelerator, model, dataloaders, optimizer, lr_scheduler):
    # # ======== MAIN TRAINING LOOP ==========
    # best_eval_loss = float('inf')
    # for epoch in range(config.num_train_epochs):
    #     model.train()
    #     total_loss = 0
    #     progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{config.num_train_epochs}")
        
    #     for step, batch in enumerate(train_dataloader):
    #         with accelerator.autocast():
    #             outputs = model(**batch)
    #             loss = outputs.loss / config.accumulation_steps
            
    #         accelerator.backward(loss)
    #         total_loss += loss.item() * config.accumulation_steps
            
    #         if (step + 1) % config.accumulation_steps == 0:
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
            
    #         progress_bar.update(1)
    #         progress_bar.set_postfix({"Loss": total_loss / (step + 1)})
            
    #         # ======= EVALUATE ============
    #         # log system metrics
    #         if (step+1) % 10 == 0:
    #             if config.use_wandb and accelerator.is_main_process:
    #                 system_metrics = get_system_metrics()
    #                 system_metrics['training_loss'] = total_loss
    #                 wandb.log(system_metrics)


    #         # log performance metrics
    #         if (step + 1) % 100 == 0:  # Evaluate less frequently
    #             model.eval()
    #             eval_loss = 0
    #             eval_preds = []
    #             for eval_batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f"Evaluation => Epoch {epoch + 1}, Step {step + 1}"):
    #                 with torch.no_grad():
    #                     eval_outputs = model(**eval_batch)
    #                     eval_loss += eval_outputs.loss.item()

    #                     logits = eval_outputs.logits
    #                     predictions = torch.argmax(logits, dim=-1)
    #                     eval_preds.append({
    #                         "predictions": predictions.detach().cpu(),
    #                         "labels": eval_batch["labels"].detach().cpu()
    #                     })

    #             # Compute metrics
    #             perf_metrics = compute_metrics(eval_preds, tkizers[1])
    #             perf_metrics["val_loss"] = eval_loss / len(val_dataloader)

    #             if config.use_wandb and accelerator.is_main_process:
    #                 wandb.log(perf_metrics)

    #             print(f"Process {process_index} || Epoch {epoch + 1}, Step {step + 1}: Eval Loss: {eval_loss:.4f}")

    #             if eval_loss < best_eval_loss:
    #                 best_eval_loss = eval_loss
    #                 os.makedirs(config.output_dir, exist_ok=True)
    #                 accelerator.save(accelerator.unwrap_model(model).state_dict(), f"{config.output_dir}/best_model.pt")


            
            # =============================== #


    # Final test evaluation




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

