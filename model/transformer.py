import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from collections import namedtuple
from model.components import (
    DecoderLayer, 
    generate_square_subsequent_mask
)



class Encoder(nn.Module):
    def __init__(self, config):
        self.layer


    def forward(self, x):
        return




class Decoder(nn.TransformerDecoder):
    def __init__(self, config):
        self.layer = DecoderLayer(config)


    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            return output

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache




class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

    def forward(self):
        return

    '''
    def forward(self, inputs, teach_forcing_tokens):

        input_embed = self.positional_encoding(
            self.embedding(inputs).permute(1, 0, 2)
        )  # input_len, bsz, hdim

        teach_forcing_embed = self.positional_encoding(
            self.embedding(teach_forcing_tokens).permute(1, 0, 2)
        )  # output_len, bsz, hdim

        memory_mask = inputs == 0  # for source padding masks

        encoded = self.encoder(
            input_embed, src_key_padding_mask=memory_mask
        )  # input_len, bsz, hdim

        if USE_OPTIMIZED_DECODER:
            decoded = self.decoder(
                teach_forcing_embed,
                memory=encoded,
                memory_key_padding_mask=memory_mask,
            )  # output_len, bsz, hdim
        else:
            tgt_mask = generate_square_subsequent_mask(
                teach_forcing_embed.size(0)
            )
            decoded = self.decoder(
                teach_forcing_embed,
                encoded,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_mask,
            )  # output_len, bsz, hdim

        logits = self.classification_layer(decoded)  # output_len, bsz, vocab_size
        return logits.permute(1, 0, 2)  # bsz, output_len, vocab_size

    def predict(self, sent: str):
        """ Used for inference """

        sent_idx = self.vocab.sent_to_idx(sent)
        sent_tensor = torch.LongTensor(sent_idx).to(DEVICE).unsqueeze(0)

        input_embed = self.positional_encoding(
            self.embedding(sent_tensor).permute(1, 0, 2)
        )  # input_len, 1, hdim

        memory_mask = sent_tensor == 0  # for source padding masks

        encoded = self.encoder(
            input_embed, src_key_padding_mask=memory_mask
        )  # input_len, 1, hdim

        decoded_tokens = (
            torch.LongTensor([self.vocab.idx_start_token]).to(DEVICE).unsqueeze(1)
        )  # 1, 1

        output_tokens = []
        cache = None
        # generation loop
        while len(output_tokens) < 256:  # max length of generation

            decoded_embedding = self.positional_encoding(self.embedding(decoded_tokens))

            if USE_OPTIMIZED_DECODER:
                decoded, cache = self.decoder(
                    decoded_embedding,
                    encoded,
                    cache,
                    memory_key_padding_mask=memory_mask,
                )
            else:
                tgt_mask = generate_square_subsequent_mask(
                    decoded_tokens.size(0), DEVICE
                )
                decoded = self.decoder(
                    decoded_embedding,
                    encoded,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_mask,
                )

            logits = self.classification_layer(decoded[-1, :, :])  # 1, vocab_size
            new_token = logits.argmax(1).item()
            
            if new_token == self.vocab.idx_end_token:  # end of generation
                break

            output_tokens.append(new_token)
            decoded_tokens = torch.cat(
                [decoded_tokens,
                 torch.LongTensor([new_token]).unsqueeze(1).to(DEVICE)],
                dim=0,
            )  # current_output_len, 1

        return self.vocab.idx_to_sent(output_tokens)
    '''