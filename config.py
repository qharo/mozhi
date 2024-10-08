from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "google-t5/t5-base"

    tkizer_save: bool = False

    src_tkizer_save_path: str = 'data/src_tkizer'
    tgt_tkizer_save_path: str = 'data/tgt_tkizer'

    source_lang: str = "en"
    target_lang: str = "ta" # AI4Bharat Identifier
    
    src_vocab_size = 30000
    tgt_vocab_size = 30000
    
    use_bit_linear: bool = False

    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048

    # model config
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    use_custom_linear: bool = False
    output_dir: str = "./output"


    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = "mozhi"
    wandb_entity: str = "qharo"  # Set this to your wandb username or team name

config = Config()
