import json, torch
import torch.nn as nn
from .trainer import TrainerBase
from torch.utils.data import DataLoader
from module.data import Dataset, Collator




class ComplementaryTrainer(TrainerBase):
	def __init__(self, config, model):
		
		super(ComplementaryTrainer, self).__init__(config, model)

		self.train_dataloader = DataLoader(
		    Dataset(tokenizer, split), 
		    batch_size=config.batch_size, 
		    shuffle=True if is_train else False,
		    collate_fn=Collator(config.pad_id),
		    pin_memory=True,
		    num_workers=2
		)

		self.criterion = nn.CrossEntropyLoss()


	def select_data(self):
		self.model.eval()
		epoch_loss = 0.0

		for idx, batch in enumerate(self.train_dataloader):
			src = batch['src'].to(self.device)
			label = batch['trg'][:, 0]


		return



	def first_token_generate(self):

		return