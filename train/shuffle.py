import torch
import torch.nn as nn
from .trainer import TrainerBase


class ShuffleTrainer(TrainerBase):
	def __init__(
		self, config, model, train_dataloader, valid_dataloader
		):
		
		super(ShuffleTrainer, self).__init__(
			config, model, train_dataloader, valid_dataloader
			)

		self.criterion = nn.CrossEntropyLoss()


	def ce_loss(self, x, y):
		with torch.autocast(device_type=self.device_type, dtype=torch.float16):
			logit = self.model(x, y)
			loss = self.ce_loss(logit, y)
		return loss


	def gen_loss(self, logit, label):
		with torch.autocast(device_type=self.device_type, dtype=torch.float16):
			logit = self.model.predict(x, y)
			loss = self.ce_loss(logit, y)
		
		return loss


	def train_epoch(self):
		epoch_loss = 0.0
		self.model.train()

		for idx, batch in enumerate(self.train_dataloader):
			x = batch['src'].to(self.device)
			y = batch['trg'].to(self.device)

			#Get Loss
			if idx // 2:
				loss = self.ce_loss(x, y)
			else:
				loss = self.gen_loss(x, y)

			loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()     

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()

		return epoch_loss, epoch_ppl


	def valid_epoch(self):
		epoch_loss = 0.0
		self.model.eval()

		with torch.no_grad():
			for idx, batch in enumerate(self.valid_dataloader):
				x = batch['src'].to(self.device)
				y = batch['trg'].to(self.device)			

				

					#Get Loss
					if idx // 2:
						loss = self.ce_loss(logit, y)
					else:
						loss = self.gen_loss(logit, y)


        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        				
		
		return epoch_loss, epoch_ppl