import torch
import torch.nn as nn
from collections import namedtuple
from transformers import T5EncoderModel



class Discriminator(nn.Module):
	def __init__(self, config):
		super(Discriminator, self).__init__()
		self.device = config.device

	    self.encoder = T5EncoderModel.from_pretrained(config.model_name)
	    self.classifier = nn.Linear(generator.config.d_model, 1)
	    self.dropout = nn.Dropout(generator.config.dropout_rate)

	    self.pad_id = self.encoder.pad_token_id
        self.criterion = nn.BCELoss(ignore_index=self.pad_id, 
                                    label_smoothing=0.1).to(self.device)
        self.outputs = namedtuple('Discriminator_Outputs', ('logit', 'loss'))

	    
	def forward(self, input_ids, attention_mask, labels):
		enc_out = self.encoder(input_ids, attention_mask).last_hidden_state
		out = self.classifier(enc_out)
		out = self.dropout(out)
		loss = self.criterion(out, label)
		return self.outputs(out, loss)




def load_models(config):
	if config.mode == 'pretrain':
		if config.model_type == 'generator':
			return load_generator(config), None
		elif config.model_type == 'discriminator':
			return None, load_discriminator(config)

	elif config.mode == 'train':
		return load_generator(config), load_discriminator(config)

	else:
		return load_generator(config), None



def load_generator(config):
    if config.task == 'pretrain':
    	generator = T5ForConditionalGeneration.from_pretrained(config.model_name)
    	print(f"Generator for {config.mode} has loaded")

    elif config.task == 'train':
    	generator = T5ForConditionalGeneration.from_pretrained(config.gen_pre_ckpt)
    	print(f"Generator for {config.mode} has loaded")
        print(f"Model States has loaded from {config.gen_pre_ckpt}")
    
    else:
        generator = T5ForConditionalGeneration.from_pretrained(config.gen_ckpt)
		print(f"Generator for {config.mode} has loaded")
        print(f"Model States has loaded from {config.gen_ckpt}")
    
	print(f"--- Model Params: {count_params(generator):,}")
	print(f"--- Model  Size : {check_size(generator):.3f} MB\n")
    
    return generator.to(config.device)



def load_discriminator(config):
	discriminator = Discriminator(config)
    print(f"Discriminator for {config.mode} has loaded")

    if config.task == 'train':
        model_state = torch.load(config.dis_pre_ckpt, map_location=config.device)['model_state_dict']        
        discriminator.load_state_dict(model_state)
        print(f"Model States has loaded from {config.dis_pre_ckpt}")
    
	print(f"--- Model Params: {count_params(generator):,}")
	print(f"--- Model  Size : {check_size(generator):.3f} MB\n")

    return discriminator.to(config.device)



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