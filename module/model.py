import os, torch
import torch.nn as nn
from model import Transformer




def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name:
            nn.init.xavier_uniform_(param)          



def print_model_desc(model):
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")



def load_model(config):
    ckpt = config.ckpt
    model = Transformer(config)
    init_weights(model)
    print('Initialized Transformer Model has Loaded')    

    if config.mode != 'train':
        assert os.path.exists(ckpt)
        model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {ckpt}")

    print_model_desc(model)
    return model.to(config.device)