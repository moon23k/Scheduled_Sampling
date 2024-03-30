import json, torch
import torch.nn as nn
import torch.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau




class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__()

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.strategy = config.strategy
        self.iters_to_generate = config.iters_to_generate

        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs
        self.vocab_size = config.vocab_size
        self.early_stop = config.early_stop
        self.patience = config.patience
        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate        

        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=2)

        self.ckpt = config.ckpt
        self.record_path = self.ckpt.replace('model.pt', 'report.json')



    def print_epoch(self, epoch_report):
        train_loss = f"{epoch_report['train_loss']:.3f}"
        valid_loss = f"{epoch_report['valid_loss']:.3f}"
        gpu_memory = epoch_report['gpu_memory']
        max_memory = epoch_report['gpu_max_memory']

        max_len = max(len(train_loss), len(valid_loss), len(gpu_memory), len(max_memory))

        elapsed_time = epoch_report['epoch_time']
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))

        txt = f"""Epoch {epoch_report['epoch']}/{self.n_epochs} | Time: {elapsed_min}m {elapsed_sec}s
            >> Train Loss: {train_loss:>{max_len}}  |  Valid Loss: {valid_loss:>{max_len}}
            >> GPU Memory: {gpu_memory:>{max_len}}  |  Max Memory: {max_memory:>{max_len}}\n"""
        print(txt.replace(' '* 11, ''))
        


    def train(self):
        #Train Prerequisites
        records = []
        patience = self.patience
        prev_loss, best_loss = float('inf'), float('inf')

        torch.cuda.empty_cache()
        for epoch in range(1, self.n_epochs + 1):

            start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_time.record()

            train_epoch_loss = self.train_epoch()
            valid_epoch_loss = self.valid_epoch()

            epoch_gpu_memory = f"{torch.cuda.memory_allocated(device=None) / 1024**3:.2f}GB"
            epoch_gpu_max_memory = f"{torch.cuda.max_memory_allocated(device=None) / 1024**3:.2f}GB"

            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) // 1000

            epoch_record = {
                'epoch': epoch, 
                'train_loss': train_epoch_loss, 
                'valid_loss': valid_epoch_loss, 
                'learning_rate': self.optimizer.param_groups[0]['lr'], 
                'epoch_time': elapsed_time,
                'gpu_memory': epoch_gpu_memory, 
                'gpu_max_memory': epoch_gpu_max_memory                
            }

            records.append(epoch_record)
            self.print_epoch(epoch_record)
            self.lr_scheduler.step(valid_epoch_loss)

            #save best model
            if best_loss > valid_epoch_loss:
                best_loss = valid_epoch_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)

            #Early Stopping Process
            if self.early_stop:
                if prev_loss > valid_epoch_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = valid_epoch_loss

            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)



    def train_epoch(self):
        self.model.train()
        tot_len = len(self.train_dataloader)
        epoch_loss = 0


        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            is_generative = False
            if self.strategy == 'generative' and not (idx % self.iters_to_generate):
                is_generative = True
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['is_generative'] = is_generative

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(**batch).loss      
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

        return round(epoch_loss / tot_len, 3)
    


    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in self.valid_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = self.model(**batch).loss
                    epoch_loss += loss.item()
        
        return round(epoch_loss / len(self.valid_dataloader), 3)