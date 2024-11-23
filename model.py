import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import utils 

class LayerNorm(nn.Module):

    def __init__(self, features: int, eps = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) ## multiple
        self.bias = nn.Parameter(torch.zeros(features)) ## add

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size : int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        ## (batch , seq_len) --> (batch, seq_len, d_model)
        ## Paper : multiple embedding with sqrt(d_model) to scale embedding
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model : int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        ## create vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        ## create vector of shape (d_model/2)       
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        ## apply sin to even pos
        pe[:, 0::2] = torch.sin(position * div_term)
        ## apply cos to odd pos
        pe[:, 1::2] = torch.cos(position * div_term)
        ## Add batch dim to position encoding (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):

        ## (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)
    
class MultiHeadAttentionBlock(nn.Module):
    """
    This is fast implementation of Attention head
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model : int, hidden_dim : int, dropout : float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class ResidualBlock(nn.Module):

    def __init__(self, features: int, dropout : float) -> None:
        super().__init__()
        self.norm = LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, d_model : int, h : int, hidden_dim : int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.hidden_dim = hidden_dim
        self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, hidden_dim, dropout)
        self.residual_connection = nn.ModuleList([
            ResidualBlock(d_model, dropout) for i in range(2)
        ])

    def forward(self, x, mask):
        # (batch_size, seq_len, d_model)
        ## residual with attention block
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, mask))
        ## residual with feedforward block
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, d_model, h, hidden_dim, dropout, n_layers) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_layers = n_layers

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, h, hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x= layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, d_model, h, hidden_dim, dropout) -> None:
        super().__init__()
        self.attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, hidden_dim, dropout)
        self.residual_connection = nn.ModuleList([
            ResidualBlock(d_model, dropout) for _ in range(3)
        ])

    def forward(self, x, encoder_output,src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.attention_block(x,x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, d_model, h, hidden_dim, dropout, n_layers) -> None:
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, h, hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        ## (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):

    def __init__(self, 
                 src_vocab_size : int, 
                 tgt_vocab_size : int, 
                 src_seq_len : int, 
                 tgt_seq_len : int, 
                 d_model: int = 512, 
                 h : int = 8, 
                 hidden_dim : int = 2048, 
                 n_layers :int = 6, 
                 dropout: float = 0.1, **kwargs) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.d_model = d_model
        self.h = h
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers 
        self.dropout = dropout

        ## layers
        self.src_emb = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_emb = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)
        self.encoder = Encoder(d_model, h, hidden_dim, dropout, n_layers)
        self.decoder = Decoder(d_model, h, hidden_dim, dropout, n_layers)
        self.projection_layer = ProjectLayer(d_model, tgt_vocab_size)

        ## initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        # (batch, seq_len, d_model)
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        x = self.encoder(src, src_mask)

        y = self.tgt_emb(tgt)
        y = self.tgt_pos(y)
        y = self.decoder(y, x, src_mask, tgt_mask)

        out = self.projection_layer(y)

        ## batch, seq_len, vocab_size
        return out







