from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "google-t5/t5-base"

    tkizer_save: bool = False
    src_tkizer_save_path: str = 'data/src_tkizer'
    tgt_tkizer_save_path: str = 'data/tgt_tkizer'

    source_lang: str = "en"
    target_lang: str = "ta" # AI4Bharat Identifier

    use_bit_linear: bool = False

    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    output_dir: str = "./output"

    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = "mozhi"
    wandb_entity: str = "qharo"  # Set this to your wandb username or team name

config = Config()
