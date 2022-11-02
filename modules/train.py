import time, math, json, torch
import torch.nn as nn
import torch.optim as optim
from modules.data import load_dataloader



class TrainConfig:
    def __init__(self, config, generator, discriminator):
        super(TrainConfig, self).__init__()    
        self.generator = generator
        self.discriminator = discriminator
        
        self.clip = config.clip
        self.device = config.device
        self.eos_idx = config.eos_idx
        self.n_epochs = config.n_epochs
        self.output_dim = config.output_dim
        self.model_name = config.model_name

        self.train_dataloader = load_dataloader(config, 'train')
        self.valid_dataloader = load_dataloader(config, 'valid')

        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, 
                                                 label_smoothing=0.1).to(self.device)
        self.dis_criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        self.gen_optimizer = optim.Adam(self.generator.parameters(), 
                                        lr=config.learning_rate, 
                                        betas=(0.9, 0.98), 
                                        eps=1e-8)
        
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), 
                                        lr=config.learning_rate, 
                                        betas=(0.9, 0.98), 
                                        eps=1e-8)

        if config.scheduler == 'constant':
            self.scheduler = None
        elif config.scheduler == 'exp':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        elif config.scheduler == 'cycle':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                         base_lr=1e-4, 
                                                         max_lr=1e-3, 
                                                         step_size_up=10, 
                                                         step_size_down=None, 
                                                         mode='exp_range', 
                                                         gamma=0.97,
                                                         cycle_momentum=False)
        
        self.ckpt_path = config.ckpt_path
        self.gen_record_path = f"ckpt/generator.json"
        self.dis_record_path = f"ckpt/discriminator.json"        
        self.record_path = f"ckpt/{config.model_name}_gan.json"
        self.record_keys = ['epoch', 
                            'gen_train_loss', 'dis_train_loss', 
                            'gen_valid_loss', 'dis_valid_loss',
                            'learning_rate', 'train_time']

    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Generator Train Loss: {record_dict['gen_train_loss']:.3f} | \
              Discriminator Train Loss: {record_dict['dis_train_loss']:.3f}""".replace(' ' * 14, ''))

        print(f"""  >> Generator Valid Loss: {record_dict['gen_valid_loss']:.3f} | \
              Discriminator Valid Loss: {record_dict['dis_valid_loss']:.2f}\n""".replace(' ' * 14, ''))                            

    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"
    

    def sampling(self, neg, pos):
        batch_size = pos.size(0)
        ones = torch.ones(batch_size, 1).long().to(self.device)
        zeros = torch.zeros(batch_size, 1).long().to(self.device)
        eos_indice = torch.full((batch_size, 1), self.eos_idx).long().to(self.device)
        
        pos = torch.cat([pos, ones], dim=-1)
        neg = torch.cat([neg, eos_indice, ones], dim=-1)
        
        sample = torch.vstack([pos, neg])
        sample = sample[torch.randperm(sample.size(0))]
        
        return sample[:, :-1], sample[:, -1].float() #returns sample and label each




class Trainer(TrainConfig):
    def __init__(self, config, generator, discriminator):
        super(Trainer, self).__init__(config, generator, discriminator)

    def train(self):
        best_bleu, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.gen_optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)

            if self.scheduler is not None:
                self.scheduler.step()

            #save best model
            if best_bleu > record_dict['gen_valid_loss']:
                best_bleu = record_dict['gen_valid_loss']
                torch.save({'epoch': epoch,
                            'model_state_dict': self.generator.state_dict(),
                            'optimizer_state_dict': self.gen_optimizer.state_dict()},
                            self.ckpt_path)
            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()

        gen_epoch_loss, dis_epoch_loss = 0, 0
        tot_len = len(self.train_dataloader)

        for _, batch in enumerate(self.train_dataloader):
            src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)
            
            gen_logit = self.generator(src, trg[:, :-1])
            gen_pred = gen_logit.argmax(-1)
            
            sample, label = self.sampling(gen_pred, trg)
            dis_logit = self.discriminator(sample)

            dis_loss = self.dis_criterion(dis_logit, label.view(-1, 1))
            gen_loss = self.gen_criterion(gen_logit.contiguous().view(-1, self.output_dim),
                                          trg[:, 1:].contiguous().view(-1))
            gen_loss += (dis_loss.item() * 10)
            dis_loss.backward()
            gen_loss.backward()
            
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.clip)
            
            self.gen_optimizer.step()
            self.dis_optimizer.step()
            self.gen_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

            gen_epoch_loss += gen_loss.item()
            dis_epoch_loss += dis_loss.item()
            
        gen_epoch_loss = round(gen_epoch_loss / tot_len, 3)
        dis_epoch_loss = round(dis_epoch_loss / tot_len, 3)
        return gen_epoch_loss, dis_epoch_loss
    

    def valid_epoch(self):
        self.generator.eval()
        self.discriminator.eval()

        gen_epoch_loss, dis_epoch_loss = 0, 0
        tot_len = len(self.valid_dataloader)
        
        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):
                src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)
                gen_logit = self.generator(src, trg[:, :-1])
                gen_pred = gen_logit.argmax(-1)

                sample, label = self.sampling(gen_pred, trg)
                dis_logit = self.discriminator(sample)

                dis_loss = self.dis_criterion(dis_logit, label.view(-1, 1))
                gen_loss = self.gen_criterion(gen_logit.contiguous().view(-1, self.output_dim),
                                              trg[:, 1:].contiguous().view(-1))
                gen_loss += (dis_loss.item() * 10)
                gen_epoch_loss += gen_loss.item()
                
        gen_epoch_loss = round(gen_epoch_loss / tot_len, 3)
        dis_epoch_loss = round(dis_epoch_loss / tot_len, 3)
        return gen_epoch_loss, dis_epoch_loss





class PreTrainer(TrainConfig):
    def __init__(self, config, generator, discriminator):
        super(PreTrainer, self).__init__(config, generator, discriminator)


    def train(self):
        best_bleu, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)

            if self.scheduler is not None:
                self.scheduler.step()

            #save best model
            if best_bleu > record_dict['valid_loss']:
                best_bleu = record_dict['valid_loss']
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt_path)
            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        for _, batch in enumerate(self.train_dataloader):
            src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)
            logit = self.model(src, trg[:, :-1])

            loss = self.criterion(logit.contiguous().view(-1, self.output_dim),
                                  trg[:, 1:].contiguous().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)    
        return epoch_loss, epoch_ppl
    

    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        tot_len = len(self.valid_dataloader)
        
        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):
                src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)
                logit = self.model(src, trg[:, :-1])

                loss = self.criterion(logit.contiguous().view(-1, self.output_dim),
                                      trg[:, 1:].contiguous().view(-1))
                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        return epoch_loss, epoch_ppl