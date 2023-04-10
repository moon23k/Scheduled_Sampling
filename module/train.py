import time, math, json, torch, evaluate
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader, tokenizer=None):
        super(Trainer, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = config.strategy

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.max_len = 0
        self.lr = config.lr
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs

        self.iters_to_accumulate = config.iters_to_accumulate
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.early_stop = config.early_stop
        self.patience = config.patience

        self.ckpt = config.ckpt
        self.bleu = evaluate.load('bleu')
        
        self.record_path = f"ckpt/{self.strategy}_record.json"
        self.record_keys = ['epoch', 'train_loss', 'valid_loss', 'learning_rate', 'train_time']


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))

        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              >> Valid Loss: {record_dict['valid_loss']:.3f}""".replace(' ' * 14, ''))


    def get_bleu_loss(self, input_ids, attention_mask, references):

        predictions = self.model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          max_length=self.max_len, 
                                          use_cache=True)

        predictions = self.tokenizer.batch_decode(predictions, 
                                                  skip_special_tokens=True)
        bleu_score = self.bleu.compute(predictions=[p.lower() for p in predictions], 
                                       references=references)['bleu']
        
        if not bleu_score:
            bleu_score = 1e-4

        bleu_score = torch.tensor(bleu_score, requires_grad=True)
        return -torch.log(bleu_score)


    def get_ce_loss(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels).loss

    def get_loss(self, batch, idx):
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        references = batch['references']

        self.max_len = labels.size(1) #update max_len for generation

        if self.strategy == 'fine':
            return self.get_ce_loss(input_ids, attention_mask, labels)
        
        if self.strategy == 'axiliary':
            ce_loss = self.get_ce_loss(input_ids, attention_mask, labels)
            bleu_loss = self.get_bleu_loss(input_ids, attention_mask, references)
            return (ce_loss + bleu_loss) * 0.5

        if self.strategy == 'scheduled':
            if idx // self.schedule_step:
                return self.get_ce_loss(input_ids, attention_mask, labels)
            return self.get_bleu_loss(input_ids, attention_mask, references)

        if self.strategy == 'generative':
            return self.get_bleu_loss(input_ids, attention_mask, references)



    def train(self):
        records = []
        best_loss = float('inf')
        patience = self.patience

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, self.train_epoch(), self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
                #patience intialize
                if self.early_stop:
                    patience = self.patience
            
            else:
                if not self.early_stop:
                    continue
                patience -= 1
                if not patience:
                    print('\n--- Training Ealry Stopped ---')
                    break

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self):
        self.model.train()

        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            loss = self.get_loss(batch, idx)            
            loss.backward()

            if (idx) % self.iters_to_accumulate == 0 or idx == tot_len:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()

        return round(epoch_loss / tot_len, 3)


    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0

        for idx, batch in enumerate(self.valid_dataloader):
            idx += 1
            loss = self.get_loss(batch, idx)
            epoch_loss += loss.item()

        return round(epoch_loss / len(self.valid_dataloader), 3)
        