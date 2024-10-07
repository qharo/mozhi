from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "bert-base-uncased"  # We'll use this architecture, but initialize from scratch
    src_lang: str = "en"
    tgt_lang: str = "fr"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    use_custom_linear: bool = False
    output_dir: str = "./output"

    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = "mozhi"
    wandb_entity: str = "qharo"  # Set this to your wandb username or team name

config = Config()
