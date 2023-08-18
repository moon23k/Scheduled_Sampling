import torch
import torch.nn as nn
from .trainer import TrainerBase



class StandardTrainer(TrainerBase):
	def __init__(
        self, 
        config, 
        model, 
        train_dataloader, 
        valid_dataloader
    ):
		
        super(StandardTrainer, self).__init__(
            config, 
            model, 
            train_dataloader, 
            valid_dataloader
        )

		self.criterion = nn.CrossEntropyLoss()


    def get_loss(self, logit, label):
        loss = self.criterion(logit, label)
        return loss


    def train_epoch(self):
        
        self.model.train()
        epoch_loss = 0.0
        tot_len = len(self.train_dataloader)
        

        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            src = batch['src'].to(self.device)
            trg = batch['trg'].to(self.device)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                logit = self.model(src, trg)
                loss = self.get_loss(logit, trg)
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
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3) 

        return epoch_loss, epoch_ppl



    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in self.valid_dataloader:
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = self.model(src, trg).loss

                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        
        return epoch_loss, epoch_ppl