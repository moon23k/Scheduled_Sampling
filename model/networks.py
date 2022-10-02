import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.input_dim, config.emb_dim)
        scale = torch.sqrt(torch.FloatTensor([config.emb_dim])).to(config.device)
        self.register_buffer('scale', scale)
    
    def forward(self, x):
        return self.embedding(x) * self.scale


class PosEncoding(nn.Module):
    def __init__(self, config, max_len=512):
        super(PosEncoding, self).__init__()
        pe = torch.zeros(max_len, config.emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * (-math.log(10000.0) / config.emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :].detach()


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.tok_emb = TokenEmbedding(config)
        self.pos_enc = PosEncoding(config)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        return self.dropout(self.tok_emb(x) + self.pos_enc(x))


class MultiHeadAttn(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttn, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        assert self.hidden_dim % self.n_heads == 0
        self.head_dim = config.hidden_dim // config.n_heads

        self.fc_q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc_k = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc_v = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc_out = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, query, key, value, mask=None):
        batch_size = key.size(0)
        Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        out = self.attention(Q, K, V, mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.fc_out(out)


    def attention(self, query, key, value, mask):
        d_k = key.size(-1)        
        score = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(d_k)

        if mask is not None:
            attn_score = score.masked_fill(mask==0, -1e9)
        prob = F.softmax(score, dim=-1)
        return torch.matmul(self.dropout(prob), value)
    
    def split(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)


class PositionwiseFFN(nn.Module):
    def __init__(self, config):
        super(PositionwiseFFN, self).__init__()
        self.fc_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.fc_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        out = self.fc_1(x)
        out = self.dropout(F.relu(out))
        return self.fc_2(out)


class ResidualConn(nn.Module):
    def __init__(self, config):
        super(ResidualConn, self).__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        out = x + sublayer(x)
        return self.dropout(self.layer_norm(out))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttn(config)
        self.pff = PositionwiseFFN(config)
        self.residual_conn = get_clones(ResidualConn(config), 2)

    def forward(self, x, src_mask):
        x = self.residual_conn[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residual_conn[1](x, self.pff)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttn(config)
        self.cross_attn = MultiHeadAttn(config)
        self.pff = PositionwiseFFN(config)
        self.residual_conn = get_clones(ResidualConn(config), 3)
    
    def forward(self, x, memory, src_mask, trg_mask):
        x = self.residual_conn[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.residual_conn[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        return self.residual_conn[2](x, self.pff)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = get_clones(EncoderLayer(config), config.n_layers)

    def forward(self, src, src_mask):
        src = self.embeddings(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = get_clones(DecoderLayer(config), config.n_layers)
    
    def forward(self, trg, memory, src_mask, trg_mask):
        trg = self.embeddings(trg)
        for layer in self.layers:
            trg = layer(trg, memory, src_mask, trg_mask)
        return trg



class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.device = config.device
        self.pad_idx = config.pad_idx
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, src, trg):
        src_mask, trg_mask = self.pad_mask(src), self.dec_mask(trg)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return self.fc_out(dec_out).argmax(-1)
    
    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
    
    def dec_mask(self, x):
        pad_mask = self.pad_mask(x)
        sub_mask = torch.tril(torch.ones((x.size(-1), x.size(-1)))).bool().to(self.device)
        return pad_mask & sub_mask



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.device = config.device
        self.act = nn.GELU()
        #아 여기 hidden dim 쪽에 얘들을 좀더 영리하게 나눌수 있을 것 같은데.. 좀 예쁘게 코드를 써보자잉
        self.hidden_dim = config.hidden_dim
        self.half_dim = config.hidden_dim // 4
        self.quarter_dim = config.hidden_dim // 8
        
        self.encoder = Encoder(config)
        self.fc_1 = nn.Linear(self.hidden_dim, self.half_dim)
        self.fc_2 = nn.Linear(self.half_dim, self.quarter_dim)
        self.fc_3 = nn.Linear(self.quarter_dim, 2)

        self.dropout_1 = nn.Dropout(config.dropoput_ratio)
        self.dropout_2 = nn.Dropout(config.dropoput_ratio)
        self.dropout_3 = nn.Dropout(config.dropoput_ratio)


    def forward(self, x):
        out = self.encoder(x)
        out = self.dropout_1(self.act(self.fc_1(out)))
        out = self.dropout_2(self.act(self.fc_2(out)))
        out = self.dropout_3(self.act(self.fc_3(out)))

        return nn.Sigmoid(out)

