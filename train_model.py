import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import utils
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics
from sampling import greedy_decode
from datasets import load_dataset
from model import Transformer
from torch.utils.tensorboard import SummaryWriter

def run_validation(model, eval_dl, src_tokenizer, tgt_tokenizer, tgt_max_len, 
                   device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in eval_dl:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, tgt_max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*80)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def train_model():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    ## Get Data
    train_dl, eval_dl, src_tokenizer, tgt_tokenizer = utils.get_dataset(config)
    config['src_vocab_size'] = src_tokenizer.get_vocab_size()
    config['tgt_vocab_size'] = tgt_tokenizer.get_vocab_size()

    ## Get Model
    model = Transformer(**config).to(device)
    ## Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    ## optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), eps=1e-9)
    ## loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']

    model_filename = utils.latest_weights_file_path(config) if preload == 'latest' else utils.get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dl, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)   # (B, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)
            label = batch['label'].to(device)  # (B, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)   # (B, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # print("proj_output : ",  proj_output.shape)
            # print("label :", label.shape)

            # compute loss
            ## input : (B * seq_len, vocab_size), (B * seq_len)
            loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, eval_dl, src_tokenizer, tgt_tokenizer, config['tgt_seq_len'], 
                       device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = utils.get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    

if __name__ == "__main__":
    ## Load config
    config = utils.get_config("config.yaml")
    ## Train Model 
    train_model()
    print("Training completed !!")




