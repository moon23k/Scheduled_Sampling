import torch
import torch.nn as nn
from .trainer import TrainerBase


class AlteranateTrainer(TrainerBase):
	def __init__(self, config, model, train_dataloader, valid_dataloader):
		super(AlteranateTrainer, self).__init__(config, model, train_dataloader, valid_dataloader)

		self.ce_criterion = nn.CrossEntropyLoss()
		self.mse_criterion = nn.MSE()


	def train_epoch(self):
		self.model.train()
		epoch_loss = 0.0
		return


	def valid_epoch(self):
		self.model.eval()
		epoch_loss = 0.0		
		return		