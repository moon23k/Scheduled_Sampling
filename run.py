import os, argparse, torch
from torch import Transformer

from module.test import Tester
from module.train import Trainer, PreTrainer
from module.data import load_dataloader

from transformers import (set_seed,
                          T5Config, 
                          T5TokenizerFast, 
                          T5ForConditionalGeneration)



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.task = args.task
        self.model_type = args.model
        self.src, self.trg = self.task[:2], self.task[2:]
        self.ckpt = f"ckpt/{self.task}_{self.model_type}.pt" \
                      if self.task != 'pretrain' \
                      else f"ckpt/pre_{self.task}_{self.model_type}.pt"

        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 16
        self.learning_rate = 5e-5
        self.iters_to_accumulate = 4
        
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def load_model(config):
    if config.mode == 'train':
        model = BartForConditionalGeneration.from_pretrained(config.model_name)
        print(f"Pretrained {config.task.upper()} BART Model for has loaded")
    
    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_config = BartConfig.from_pretrained(config.model_name)
        model = BartForConditionalGeneration(model_config)
        print(f"Initialized {config.task.upper()} BART Model has loaded")

        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model states has loaded from {config.ckpt}")




    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params
        
    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    return model.to(config.device)



def inference(model, tokenizer):
    model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = tokenizer(input_seq)['input_ids']
        output_ids = model.generate(input_ids, beam_size=4, max_new_tokens=300, use_cache=True)
        output_seq = tokenizer.decode(output_ids, skip_special_tokens=True)

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")       



def train(config, model):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')
    trainer = Trainer(config, model, train_dataloader, valid_dataloader)
    trainer.train()


def test(config, model, tokenizer):
    test_dataloader = load_dataloader(config, 'test')
    tester = Tester(config, model, tokenizer, test_dataloader)
    tester.test()    




def generate(config, model):
    generated = []
    train_dataloader = load_dataloader(config, 'train')
    
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(train_dataloader):   
            
            input_ids = batch[f'{config.src}_ids'].to(config.device)
            labels = batch[f'{config.trg}_ids'].to(config.device)
                            
            with torch.autocast(device_type=config.device_type, dtype=torch.float16):
                preds = model.generate(input_ids, max_new_tokens=300, use_cache=True)    
    
            



def main(args):
    set_seed(42)
    config = Config(args.task, args.task)
    model = load_model(config)

    setattr(config, 'pad_id', model.config.pad_token_id)

    if config.task != 'train':
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model_name, model_max_length=300)


    if config.mode == 'pretrain':
        pretrain(config, model)

    elif config.mode == 'generate':
        generate(config, model, tokenizer)

    elif config.mode == 'train':
        train(config, model)

    elif config.mode == 'test':
        test(config, model, tokenizer)
    
    elif config.mode == 'inference':
        inference(model, tokenizer)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', default='generator', required=False)
    
    args = parser.parse_args()


    assert args.task in ['ende', 'deen']
    assert args.mode in ['train', 'test', 'inference', 'pretrain', 'generate']

    if args.mode == 'pretrain':
        assert args.model in ['generator', 'discriminator']
    else:
        if args.mode == 'train'
            assert os.path.exists(f'ckpt/pre_{args.task}_{args.model}.pt')
            if args.model == 'discriminator':
                assert os.path.exists('data/samples.json')
        else:
            assert os.path.exists(f'ckpt/{args.task}_{args.model}.pt')

    main(args)