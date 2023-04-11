import os, argparse, torch
from module.model import load_model
from module.data import load_dataloader
from module.train import Trainer
from module.test import Tester
from transformers import set_seed, T5TokenizerFast



class Config(object):
    def __init__(self, args):    

        self.strategy = args.strategy
        self.mode = args.mode
        self.mname = 't5-small'
        self.ckpt = f'ckpt/{self.strategy}_model.pt'

        self.clip = 1
        self.lr = 5e-5
        self.n_epochs = 10
        self.batch_size = 16
        self.iters_to_accumulate = 4
        
        self.early_stop = True
        self.patience = 3

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        
        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device_type)


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


def train(config, model, tokenizer):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')    
    trainer = Trainer(config, model, train_dataloader, valid_dataloader, 
                      tokenizer=None if config.strategy == 'fine' else tokenizer)
    trainer.train()


def main(args):
    #prerequisites
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = T5TokenizerFast.from_pretrained(config.mname, model_max_length=512)
    config.pad_id = tokenizer.pad_token_id
    
    #Train
    if config.mode == 'train':
        if config.strategy != 'consecutive':
            train(config, model)

        elif config.strategy == 'consecutive':
            if not os.path.exists('ckpt/fine_model.pt'):
                config.strategy = 'fine'
                train(config, model, tokenizer)
                config.strategy = 'consecutive'                                
            else:
                model_state = torch.load('ckpt/fine_model.pt', 
                                         map_location=config.device)['model_state_dict']
                model.load_state_dict(model_state)                

            train(config, model, tokenizer)


    #Test
    elif config.mode == 'test':
        assert os.path.exists(config.ckpt)
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
    

    #Inference    
    elif config.mode == 'inference':
        assert os.path.exists(config.ckpt)
        inference(model, tokenizer)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', required=True)
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.strategy in ['fine', 'auxiliary', 'generative', 'consecutive']
    assert args.mode in ['train', 'test', 'inference']

    main(args)