import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model
from models.transformer import Transformer, Encoder, Decoder



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        #self.act = nn.GELU()
        self.pad_idx = config.pad_idx
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.classifier = nn.Linear(config.hidden_dim, 1)
    

    def forward(self, x):
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        out = self.encoder(x, mask)[:, 0]
        #out = self.act(out)
        #out = self.dropout(out)
        return self.classifier(out)



class GPT_Generator(nn.Module):
    def __init__(self, config):
        super(GPT_Generator, self).__init__()
        self.gpt
        self.encoder
        self.decoder
        self.fc_out

    def forward(self, src, trg):
        return

    def generate(self, src):
        return



class BERT_Discriminator(nn.Module):
    def __init__(self, config):
        super(BERT_Discriminator, self).__init__()
    
    def forward(self, src, trg):
        return