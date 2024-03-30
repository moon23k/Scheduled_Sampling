import copy, math, random, torch
import torch.nn as nn
from collections import namedtuple



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        max_len = config.max_len
        pe = torch.zeros(max_len, config.emb_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.pos_dropout = nn.Dropout(config.dropout_ratio)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)
            self.fc_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_dropout(self.pos_emb(out))

        if not self.use_fc_layer:
            return out
        return self.fc_dropout(self.fc(out))



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.layers = clones(layer, config.n_layers)


    def forward(self, x, e_mask):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=e_mask)
        return x



class DecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        x,
        memory=None,
        e_mask=None,
        d_mask=None,
        use_cache=False
    ):

        if not use_cache:
            return super().forward(
                x,
                memory,
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask
            )


        last_token = x[:, -1:, :]

        # self attention part
        _x = self.self_attn(last_token, x, x)[0]

        last_token = last_token + self.dropout1(_x)
        last_token = self.norm1(last_token)


        # encoder-decoder attention
        _x = self.multihead_attn(
            last_token, memory, memory,
            key_padding_mask=e_mask,
        )[0]

        last_token = last_token + self.dropout2(_x)
        last_token = self.norm2(last_token)

        # final feed-forward network
        _x = self.activation(self.linear1(last_token))
        _x = self.linear2(self.dropout(_x))
        last_token = last_token + self.dropout3(_x)
        last_token = self.norm3(last_token)
        
        return last_token



class Decoder(nn.TransformerDecoder):

    def forward(
        self,
        x,
        memory=None,
        cache=None,
        e_mask=None,
        d_mask=None,
        use_cache=True
    ):

        output = x

        #In case of not using Cache
        if not use_cache:
            for layer in self.layers:
                output = layer(output, memory, e_mask, d_mask, False)
            return output, None

        #In case of using Cache
        new_token_cache = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, use_cache=True)
            new_token_cache.append(output)
            
            if cache is not None:  
                output = torch.cat([cache[idx], output], dim=1)

        new_cache = torch.stack(new_token_cache, dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, new_cache], dim=2)

        return output, new_cache




class ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id

        self.device = config.device
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size
        
        self.strategy = config.strategy
        self.aux_ratio = config.aux_ratio
        self.sampling_ratio = config.sampling_ratio

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.CrossEntropyLoss()


    def auxiliary_loss(self, y, memory, e_mask, loss):
        label = y[:, 1]
        y = y[:, 0].unsqueeze(1)

        dec_out, _ = self.decode(y, memory, None, e_mask, None)
        logit = self.generator(dec_out)

        _loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        aux_loss = loss * (1 - self.aux_ratio) + _loss * self.aux_ratio

        return aux_loss


    def sampling_loss(self, logit, y, label, memory, e_mask):
        bs, seq_len = y.shape
        pred = logit.argmax(dim=-1)

        sampled = torch.empty(bs, seq_len, dtype=torch.long)
        sampled = sampled.fill_(self.pad_id).to(self.device)
        sampled[:, 0] = self.bos_id

        for t in range(1, seq_len):
            if random.random() < self.sampling_ratio:
                sampled[:, t] = y[:, t]
            else:
                sampled[:, t] = pred[:, t]
        
        d_mask = self.dec_mask(sampled)
        dec_out, _ = self.decode(sampled, memory, None, e_mask, d_mask)
        logit = self.generator(dec_out)
        
        loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return loss 


    def teacher_forcing_forward(self, x, y):
        y, label = self.shift_y(y)
        
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)

        memory = self.encode(x, e_mask)

        dec_out, _ = self.decode(y, memory, None, e_mask, d_mask, use_cache=False)
        logit = self.generator(dec_out)

        self.out.logit = logit
        loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        ) if self.strategy != 'sampling' else None

        #Getting Loss Process             
        if self.strategy == 'auxiliary':
            self.out.loss = self.auxiliary_loss(y, memory, e_mask, loss)
        elif self.strategy == 'sampling':
            self.out.loss = self.sampling_loss(logit, y, label, memory, e_mask)
        else:
            self.out.loss = loss
            
        return self.out



    def generative_forward(self, x, y):

        _, label = self.shift_y(y)
        batch_size, output_len = label.shape
        
        pred = torch.zeros((batch_size, 1), dtype=torch.long)
        pred = pred.fill_(self.bos_id).to(self.device)
        logit = torch.empty(batch_size, output_len, self.vocab_size).to(self.device)

        cache = None
        e_mask = self.pad_mask(x)
        memory = self.encode(x, e_mask)

        for idx in range(1, output_len+1):
            y = pred[:, :idx]
            d_out, cache = self.decode(y, memory, cache, e_mask, use_cache=True)

            curr_logit = self.generator(d_out[:, -1:, :])
            curr_pred = curr_logit.argmax(dim=-1)

            logit[:, idx-1:idx, :] = curr_logit
            pred = torch.cat([pred, curr_pred], dim=1)
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )        

        return self.out


class Transformer(ModelBase):
    def __init__(self, config):
        super(Transformer, self).__init__(config)

        self.enc_emb = Embeddings(config)
        self.encoder = Encoder(config)

        self.dec_emb = Embeddings(config)
        self.decoder = Decoder(
            DecoderLayer(
                d_model=config.hidden_dim, 
                nhead=config.n_heads, 
                dim_feedforward=config.pff_dim,
                batch_first=True
            ),
            num_layers=config.n_layers,
        )

        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)


    @staticmethod
    def shift_y(y):
        return y[:, :-1], y[:, 1:]


    def pad_mask(self, x):
        return x == self.pad_id


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def encode(self, x, x_mask):
        x = self.enc_emb(x)
        x = self.encoder(x, x_mask)
        return x


    def decode(self, x, memory, cache=None, 
               e_mask=None, d_mask=None, use_cache=False):
        
        x = self.dec_emb(x)
        x, cache = self.decoder(x, memory, cache, e_mask, d_mask, use_cache)
        return x, cache        
        

    def forward(self, x, y, is_generative=False):
        if is_generative:
            return self.generative_forward(x, y)
        else: #This process contains ['standard', 'auxiliary', 'sampling'] Strategies
            return self.teacher_forcing_forward(x, y)
