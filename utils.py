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


def get_translation(model, src_tokenizer, tgt_tokenizer, src_seq_len, tgt_seq_len):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    def translate(model, sentence: str):

        model.eval()
        with torch.no_grad():
            source = src_tokenizer.encode(sentence)
            source = torch.cat([
                torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64), 
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([src_tokenizer.token_to_id('[PAD]')] * (src_seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(device)
            source_mask = (source != src_tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = model.encode(source, source_mask)

            # Initialize the decoder input with the sos token
            decoder_input = torch.empty(1, 1).fill_(tgt_tokenizer.token_to_id('[SOS]')).type_as(source).to(device)

            # Print the source sentence and target start prompt
            if label != "": print(f"{f'ID: ':>12}{id}") 
            print(f"{f'SOURCE: ':>12}{sentence}")
            if label != "": print(f"{f'TARGET: ':>12}{label}") 
            print(f"{f'PREDICTED: ':>12}", end='')


            # Generate the translation word by word
        while decoder_input.size(1) < tgt_seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            print(f"{tgt_tokenizer.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tgt_tokenizer.token_to_id('[EOS]'):
                break

        # convert ids to tokens
        return tgt_tokenizer.decode(decoder_input[0].tolist())
    
    return translate
