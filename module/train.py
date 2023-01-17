import time, math, json, torch
from tqdm import tqdm
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim
from collections import namedtuple


class TrainerBase:
    def __init__(self, config):
        super(TrainerBase, self).__init__()
        
        self.src = config.src
        self.trg = config.trg
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        if isinstance(self, PreTrainer):
            print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
                  Valid Loss: {record_dict['valid_loss']:.3f}\n""".replace(' ' * 14, ''))
        elif isinstance(self, Trainer):
            print(f"""  >> Generator Train Loss: {record_dict['gen_train_loss']:.3f} | \
                  Generator Valid Loss: {record_dict['gen_valid_loss']:.3f}""".replace(' ' * 14, ''))            
            print(f"""  >> Discriminator Train Loss: {record_dict['dis_train_loss']:.3f} | \
                  Discriminator Valid Loss: {record_dict['dis_valid_loss']:.3f}\n""".replace(' ' * 14, ''))


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"        


    def save_ckpt(self, epoch, ckpt, model, optimizer):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    ckpt)



class Trainer(TrainerBase):
    def __init__(self, config, generator, discriminator, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__(config)

        self.generator = generator
        self.discriminator = discriminator

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.record_keys = ['epoch', 'gen_train_loss', 'gen_valid_loss',
                            'dis_train_loss', 'dis_valid_loss',  
                            'gen_lr', 'dis_lr', 'train_time']
        
        self.gen_optimizer = optim.AdamW(params=self.generator.parameters(), lr=config.lr)
        self.dis_optimizer = optim.AdamW(params=self.discriminator.parameters(), lr=config.lr)

        self.gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.gen_optimizer, 'min')
        self.dis_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.dis_optimizer, 'min')

        self.gen_ckpt = config.gen_ckpt
        self.dis_ckpt = config.dis_ckpt

        self.gen_record_path = self.gen_ckpt.replace('.pt', '.json')
        self.dis_record_path = self.dis_ckpt.replace('.pt', '.json')

        self.gen_inputs = namedtuple('Generator_Inputs', 
                                     ('input_ids', 'attention_mask', 'labels'))
        self.dis_inputs = namedtuple('Discriminator_Inputs', 
                                     ('input_ids', 'attention_mask', 'labels'))


    def train(self):
        gen_best_loss, dis_best_loss, records = float('inf'), float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.gen_optimizer.param_groups[0]['lr'],
                           self.dis_optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            gen_curr_loss = record_dict['gen_valid_loss']
            dis_curr_loss = record_dict['dis_valid_loss']
            self.gen_scheduler.step(gen_curr_loss)
            self.dis_scheduler.step(dis_curr_loss)

            #save best generator states
            if gen_best_loss >= gen_curr_loss:
                gen_best_loss = gen_curr_loss
                self.save_ckpt(epoch, self.gen_ckpt, self.generator, self.gen_optimizer)

            #save best discriminator states
            if dis_best_loss >= dis_curr_loss:
                dis_best_loss = dis_curr_loss
                self.save_ckpt(epoch, self.dis_ckpt, self.discriminator, self.dis_optimizer)

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def update_inputs(self, batch):
        
        #Update Generator Inputs
        gen_ids = batch[f'{self.src}_ids'].to(self.device)
        gen_mask =  batch[f'{self.src}_mask'].to(self.device)
        gen_labels = batch[f'{self.trg}_ids'].to(self.device)
        #self.gen_inputs(gen_ids, gen_mask, gen_labels)
        self.gen_inputs.input_ids = gen_ids
        self.gen_inputs.attention_mask = gen_mask
        self.gen_inputs.labels = gen_labels

        #Update Discriminator Inputs
        labels = gen_labels
        batch_size = gen_ids.size(0)
        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            samples = self.generator.generate(input_ids=gen_ids,
                                              attention_mask=gen_mask, 
                                              max_new_tokens=gen_ids.size(1), 
                                              use_cache=True)[:, 1:]

        pad_len = samples.size(-1) - labels.size(-1)
        if pad_len > 0:
            pad_tensor = torch.zeros(batch_size, pad_len, dtype=torch.long)
            labels = torch.cat((labels, pad_tensor.to(self.device)), dim=-1)
        elif pad_len < 0:
            pad_tensor = torch.zeros(batch_size, -pad_len, dtype=torch.long)
            samples = torch.cat((samples, pad_tensor.to(self.device)), dim=-1)

        dis_ids = torch.cat((samples, labels), dim=0)
        dis_labels = torch.cat((torch.zeros(batch_size), 
                                torch.ones(batch_size)), dim=0)
        dis_mask = (dis_ids != 0).long()
        dis_indice = torch.randperm(dis_ids.size(0))

        dis_ids = dis_ids[dis_indice].to(self.device)
        dis_mask = dis_mask[dis_indice].to(self.device)
        dis_labels = dis_labels[dis_indice].to(self.device)

        #self.dis_inputs(dis_ids, dis_mask, dis_labels)
        self.dis_inputs.input_ids = dis_ids
        self.dis_inputs.attention_mask = dis_mask
        self.dis_inputs.labels = dis_labels



    def get_losses(self):
        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            gen_loss = self.generator(input_ids=self.gen_inputs.input_ids, 
                                      attention_mask=self.gen_inputs.attention_mask,
                                      labels=self.gen_inputs.labels).loss

            dis_loss = self.discriminator(input_ids=self.dis_inputs.input_ids, 
                                          attention_mask=self.dis_inputs.attention_mask,
                                          labels=self.dis_inputs.labels).loss

        return (gen_loss + dis_loss.item()) * 0.5, dis_loss


    def train_epoch(self):
        gen_epoch_loss, dis_epoch_loss = 0, 0
        tot_len = len(self.train_dataloader)

        self.generator.train()
        self.discriminator.train()

        for idx, batch in tqdm(enumerate(self.train_dataloader)):
            self.update_inputs(batch)
            gen_loss, dis_loss = self.get_losses()

            gen_loss = gen_loss / self.iters_to_accumulate
            dis_loss = dis_loss / self.iters_to_accumulate

            self.scaler.scale(gen_loss).backward()
            self.scaler.scale(dis_loss).backward()
            
            if (idx + 1) % self.iters_to_accumulate == 0:
                #Gradient Clipping
                self.scaler.unscale_(self.gen_optimizer)
                self.scaler.unscale_(self.dis_optimizer)
                nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.clip)
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.gen_optimizer)
                self.scaler.step(self.dis_optimizer)
                
                self.scaler.update()
                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()

            gen_epoch_loss += gen_loss.item()
            dis_epoch_loss += dis_loss.item()
        
        gen_epoch_loss = round(gen_epoch_loss / tot_len, 3)
        dis_epoch_loss = round(dis_epoch_loss / tot_len, 3)
        return gen_epoch_loss, dis_epoch_loss
    


    def valid_epoch(self):
        gen_epoch_loss, dis_epoch_loss = 0, 0
        tot_len = len(self.valid_dataloader)

        self.generator.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.valid_dataloader)):   
                self.update_inputs(batch)       
                gen_loss, dis_loss = self.get_losses()

                gen_epoch_loss += gen_loss.item()
                dis_epoch_loss += dis_loss.item()
    
        gen_epoch_loss = round(gen_epoch_loss / tot_len, 3)
        dis_epoch_loss = round(dis_epoch_loss / tot_len, 3)
        return gen_epoch_loss, dis_epoch_loss




class PreTrainer(TrainerBase):
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(PreTrainer, self).__init__(config)

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.ckpt = config.gen_pre_ckpt \
        if config.model_type == 'generator' else config.dis_pre_ckpt
        self.record_path = self.ckpt.replace('.pt', '.json')
        self.record_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'train_time']


    def split_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask =  batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)        
        return input_ids, attention_mask, labels        


    def train(self):
        best_loss, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, self.train_epoch(), self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            curr_loss = record_dict['valid_loss']
            self.scheduler.step(curr_loss)

            #save best generator states
            if best_loss >= curr_loss:
                best_loss = curr_loss
                self.save_ckpt(epoch, self.ckpt, self.model, self.optimizer)

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)


    def train_epoch(self):
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        self.model.train()
        for idx, batch in enumerate(self.train_dataloader):
            input_ids, attention_mask, labels = self.split_batch(batch)
    
            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(input_ids=input_ids, 
                                  attention_mask=attention_mask, 
                                  labels=labels).loss    
                loss = loss / self.iters_to_accumulate
            self.scaler.scale(loss).backward()
            
            if (idx + 1) % self.iters_to_accumulate == 0:
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        return epoch_loss
    

    def valid_epoch(self):
        epoch_loss = 0
        tot_len = len(self.valid_dataloader)
        
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):   
                input_ids, attention_mask, labels = self.split_batch(batch)       
      
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):    
                    loss = self.model(input_ids=input_ids, 
                                      attention_mask=attention_mask, 
                                      labels=labels).loss

                epoch_loss += loss.item()
    
        epoch_loss = round(epoch_loss / tot_len, 3)
        return epoch_loss        