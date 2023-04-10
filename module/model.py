import torch
from transformers import T5ForConditionalGeneration



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
    model = T5ForConditionalGeneration.from_pretrained(config.mname)
    print('')    

    if config.mode != 'train':
        model_state = torch.torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {config.ckpt}")

    print_model_desc(model)
    return model.to(config.device)