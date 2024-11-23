import yaml
import torch
from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import TranslationDataset
from torch.utils.data import random_split, DataLoader

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_tgt_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask==0

def get_src_mask(x, pad_id):
    mask = (x!=pad_id).type(torch.int)
    return mask


def load_data():
    ds = load_dataset("Helsinki-NLP/opus-100", name = "en-hi", split="train").select(range(100000))
    return ds


def load_tokenizer(config, lang):
    tokenizer_path = Path(config['experiment_folder'], config['tokenizer_file'].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print(f"{tokenizer_path} not exist !!")
    return tokenizer


def get_config(filepath):
    config = yaml.safe_load(open(filepath))
    return config


def get_dataset(config):

    ## Load dataset
    ds = load_data()
    ## Load tokenizer
    src_tokenizer = load_tokenizer(config, config['src_lang'])
    tgt_tokenizer = load_tokenizer(config, config['tgt_lang'])
    
    ## split the data into train and eval 
    train_size = int(len(ds) * 0.90)
    eval_size = len(ds) - train_size
    train_ds_raw, eval_ds_raw = random_split(ds, [train_size, eval_size])

    ## Prepare data
    train_ds = TranslationDataset(train_ds_raw, src_tokenizer, tgt_tokenizer, 
                                  config['src_lang'], config['tgt_lang'], config['src_seq_len'],config['tgt_seq_len'])
    eval_ds = TranslationDataset(eval_ds_raw, src_tokenizer, tgt_tokenizer, 
                                 config['src_lang'], config['tgt_lang'], config['src_seq_len'], config['tgt_seq_len'])

    ## Data Loaders
    train_dl = DataLoader(train_ds, batch_size= config['batch_size'], shuffle= True)
    eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=True)

    return train_dl, eval_dl, src_tokenizer, tgt_tokenizer

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['experiment_folder']}/{config['model_folder']}"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    model_filename = f"{config['model_file']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['experiment_folder']}/{config['model_folder']}"
    model_filename = f"{config['model_file']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
