import torch, evaluate
import torch.nn as nn
from .trainer import TrainerBase



class GenerativeTrainer(TrainerBase):
	def __init__(self, config, model, tokenizer, train_dataloader, valid_dataloader):
		super(GenerativeTrainer, self).__init__(config, model, train_dataloader, valid_dataloader)
		
        self.tokenizer = tokenizer
        self.criterion = nn.MSE()
		self.bleu_module = evaluate.load('bleu')



    def get_loss(self, pred, ref):
        
        pred = self.tokenizer.batch_decode(pred)
        ref = self.tokenizer.batch_decode(ref)



        bleu_score = self.bleu_module.compute(
            predictions=[p.lower() for p in pred], 
            references=ref
        )['bleu']
        
        if not bleu_score:
            bleu_score = 1e-4

        bleu_score = torch.tensor(bleu_score, requires_grad=True)
        loss = -torch.log(bleu_score)
        
        return self.criterion(loss)


    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            src = batch['src'].to(self.device)
            trg = batch['trg'].to(self.device)        

        return


    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in self.valid_dataloader:
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)

        return      
