import json, torch
import torch.nn as nn
from .trainer import TrainerBase
from torch.utils.data import DataLoader
from module import Dataset, Collator




class ComplementaryTrainer(TrainerBase):
	def __init__(self, config, model):
		
		super(ComplementaryTrainer, self).__init__(config, model)

		self.train_dataloader = DataLoader(
		    Dataset(tokenizer, split), 
		    batch_size=config.batch_size, 
		    shuffle=False,
		    collate_fn=Collator(config.pad_id),
		    pin_memory=True,
		    num_workers=2
		)

		self.criterion = nn.CrossEntropyLoss()


	def select_data(self):
		self.model.eval()
		epoch_loss = 0.0

		for idx, batch in enumerate(self.train_dataloader):
			x = batch['src'].to(self.device)
			y = batch['trg'][:, 1]

			batch_size = x.size(0)
			d_input = torch.LongTensor(batch_size, 1).to(self.device)
			d_input[:, 0] = self.bos_id
			d_mask = self.model.dec_mask(d_input)

			x_mask = self.model.pad_mask(x)
			memory = self.model.encoder(x, x_mask)

			pred = self.model.decoder(d_input, memory, x_mask, )




		return



	def first_token_generate(self):

		return