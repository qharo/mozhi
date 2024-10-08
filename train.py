from dataset import download_dataset, build_tokenizer
from config import config
import wandb
from model import create_model

def main():
    # get dataset
    # dataset = download_dataset()

    # get tokenizer from source and target vocabularies
    # src_tkizer = build_tokenizer(dataset, 'src', 'data/src_tkizer', False, 3000)
    # tgt_tkizer = build_tokenizer(dataset, 'tgt', 'data/tgt_tkizer', False, 3000)

    # setup wandb
    # if config.use_wandb:
    #     wandb.init(project=config.wandb_project, entity=config.wandb_entity)
    #     wandb.config.update(config)

    # create model
    model = create_model()
    print(model)


if __name__ == '__main__':
    main()
