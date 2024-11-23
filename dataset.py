import torch
import utils
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, src_seq_len, tgt_seq_len) -> None:
        super().__init__()
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype= torch.long)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype= torch.long)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype= torch.long)
        
    def __len__(self):
        return len(self.ds)


    def __getitem__(self, index):
        src_pair = self.ds[index]
        src_txt = src_pair['translation'][self.src_lang]
        tgt_txt = src_pair['translation'][self.tgt_lang]

        ## convert text to tokens
        src_token_ids = self.src_tokenizer.encode(src_txt).ids
        tgt_token_ids = self.tgt_tokenizer.encode(tgt_txt).ids

        ## We will truncate the token if length if more than seq_len - 2
        src_token_ids = src_token_ids[:self.src_seq_len -2]
        tgt_token_ids = tgt_token_ids[:self.tgt_seq_len -2]

        ## add the sos and eos token
        num_of_src_pad_tokens = self.src_seq_len - len(src_token_ids) - 2
        ## add the sos token only for decoder and for label add eos
        num_of_tgt_pad_tokens = self.tgt_seq_len - len(tgt_token_ids) - 1

        ## Encoder Input
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_token_ids, dtype= torch.long),
            self.eos_token,
            torch.tensor([self.pad_token] * num_of_src_pad_tokens, dtype = torch.long)
        ],dim = 0)

        ## Decoder Input
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_token_ids, dtype = torch.long),
            torch.tensor([self.pad_token] * num_of_tgt_pad_tokens, dtype=torch.long)
        ], dim = 0)

        ## Labels
        label = torch.cat([
             torch.tensor(tgt_token_ids, dtype = torch.long),
             self.eos_token,
             torch.tensor([self.pad_token] * num_of_tgt_pad_tokens, dtype=torch.long)
        ], dim = 0)
        
        """
        Dimension
            - encoder_input : (seq_len)
            - decoder_input : (seq_len)
            - label : (seq_len)
            - encoder_mask : (1, 1, seq_len)
            - decoder_mask : (1, seq_len, seq_len)
            
        """ 

        return {
            "encoder_input": encoder_input,           # Shape: (seq_len)
            "decoder_input": decoder_input,           # Shape: (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & utils.causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label, 
            "src_text": src_txt,
            "tgt_text": tgt_txt,
        }
    
















    
