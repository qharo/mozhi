from dataclasses import dataclass
import torch

@dataclass
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset config
    data_save: bool = True
    data_save_path: str = 'data/dataset'
    n_samples = 5000000
    train_split = 0.8
    val_split = 0.001
    test_split = 1 - (train_split + val_split)

    # tkizer config
    tkizer_save: bool = True
    src_tkizer_save_path: str = 'data/src_tkizer'
    tgt_tkizer_save_path: str = 'data/tgt_tkizer'
    source_lang: str = "en"
    target_lang: str = "ta" # AI4Bharat Identifier
    src_vocab_size = 30000
    tgt_vocab_size = 30000

    # model arch
    model_name: str = "google-t5/t5-base"
    use_bit_linear: bool = False
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 1024
    max_length: int = 256

    # model training
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    output_dir: str = "data/training"
    n_steps = (n_samples * num_train_epochs * train_split) // batch_size
    n_steps_per_epoch = (n_samples * train_split) // batch_size

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "mozhi"
    wandb_entity: str = "qharo"  # Set this to your wandb username or team name

config = Config()
