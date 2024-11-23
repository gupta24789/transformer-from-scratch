import utils
import torch

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    model.eval()

    # Precompute the encoder output and reuse it for every step
    # source : (src_seq_len), source_mask : (src_seq_len)
    encoder_output = model.encode(source, source_mask)   ## (1, src_seq_len, d_model)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) ## (1,1)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        # (1,1, 1)
        decoder_mask = utils.causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate output
        # (1,1, d_model)
        out = model.decode(decoder_input, encoder_output, source_mask,decoder_mask)
        # get next token
        prob = model.project(out[:, -1])   ## (1, tgt_vocab_size)

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, 
             torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
             dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    model.eval()

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = utils.causal_mask(candidate.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(candidate, encoder_output, source_mask, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()


