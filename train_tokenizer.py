import utils
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['experiment_folder'], config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print("Loading from file ...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

if __name__=='__main__':
    # Load config
    config = utils.get_config("config.yaml")
    ## Create Exeperiment folder if not exist
    Path(config['experiment_folder']).mkdir(parents=True, exist_ok= True)
    # Load the data
    ds = utils.load_data()
    # Build Tokenizer for english
    build_tokenizer(config, ds, lang=config['src_lang'])
    print("English tokenizer build !!")
    # Build Tokenizer for hindi
    build_tokenizer(config, ds, lang=config['tgt_lang'])
    print("Hindi tokenizer build !!")
    print("END")

