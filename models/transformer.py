import torch, copy, math
import torch.nn as nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, config.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(config.input_dim, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)
        self.pos_encoding = PositionalEncoding(config)

    def forward(self, x):
        out = self.lut(x) * self.scale
        return self.pos_encoding(out) 


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        assert config.hidden_dim % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        
        self.attn = None
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.linears = clones(nn.Linear(config.hidden_dim, config.hidden_dim), 4)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [lin(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim))
        
        del query, key, value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.src_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)

    def forward(self, x, memory, e_mask, d_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, d_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, e_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(EncoderLayer(config), config.n_layers)

    def forward(self, x, mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(DecoderLayer(config), config.n_layers)

    def forward(self, x, memory, e_mask, d_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.device = config.device
        self.pad_idx = config.pad_idx
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)

    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)

    def forward(self, src, trg):
        e_mask, d_mask = self.pad_mask(src), self.dec_mask(trg)
        memory = self.encoder(src, e_mask)
        dec_out = self.decoder(trg, memory, e_mask, d_mask)
        return self.fc_out(dec_out)        