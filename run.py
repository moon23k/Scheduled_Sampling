import os, argparse, torch
from torch import Transformer

from module.data import load_dataloader
from module.model import load_models
from module.train import Trainer, PreTrainer
from module.test import Tester
from transformers import (set_seed,
                          T5TokenizerFast, 
                          T5ForConditionalGeneration)



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.task = args.task
        self.model_type = args.model
        self.model_name = 't5-small'
        self.src, self.trg = self.task[:2], self.task[2:]

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

        self.gen_ckpt = f'ckpt/{self.task}_generator.pt'
        self.dis_ckpt = f'ckpt/{self.task}_discriminator.pt'
        self.gen_pre_ckpt= f'ckpt/pre_{self.task}_generator.pt'
        self.dis_pre_ckpt = f'ckpt/pre_{self.task}_discriminator.pt'

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



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


def pretrain(config, generator, discriminator):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')
    
    model = generator if generator is not None else discriminator
    pretrainer = PreTrainer(config, model, train_dataloader, valid_dataloader)
    pretrainer.train()


def train(config, generator, discriminator):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')

    trainer = Trainer(config, generator, discriminator, train_dataloader, valid_dataloader)
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
                preds = model.generate(input_ids, max_new_tokens=labels.size(1), use_cache=True)    



def main(args):
    set_seed(42)
    config = Config(args.task, args.task)    
    generator, discriminator = load_models(config)
    
    if generator is not None:
        setattr(config, 'pad_id', generator.config.pad_token_id)
    else:
        setattr(config, 'pad_id', discriminator.pad_id)

    if config.task not in  ['pretrain','train']:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.model_name, model_max_length=300)

    #Actual Processing Codes
    if config.mode == 'pretrain':
        pretrain(config, generator, discriminator)
    elif config.mode == 'generate':
        generate(config, generator, tokenizer)
    elif config.mode == 'train':
        train(config, generator, discriminator)
    elif config.mode == 'test':
        test(config, generator, tokenizer)
    elif config.mode == 'inference':
        inference(generator, tokenizer)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', default='generator', required=False)
    
    args = parser.parse_args()
    assert args.task in ['ende', 'deen']
    assert args.mode in ['pretrain', 'generate', 'train', 'test', 'inference']


    if args.mode == 'pretrain':
        assert args.model in ['generator', 'discriminator']
        if args.model == 'discriminator':
            assert os.path.exists('data/samples.json')
    else:
        if args.mode == 'train'
            assert os.path.exists(f'ckpt/pre_{args.task}_generator.pt')
            assert os.path.exists(f'ckpt/pre_{args.task}_discriminator.pt')
        else:
            assert os.path.exists(f'ckpt/{args.task}_generator.pt')

    main(args)