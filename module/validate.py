import json, torch, evaluate
import torch.nn as nn
import torch.optim as optim



class Validator:
    def __init__(self, config, model, tokenizer, valid_dataloader):
        super(Validator, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = valid_dataloader        
        self.metric_module = evaluate.load('bleu')
        


    def validate(self):
        self.model.eval()
        avg_score = 0
        valid_scores = {}
       
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):   
                
                input_tensor = batch['src'].to(self.device)
                label = batch['trg'].tolist()
                                
                pred = self.model.generate(input_tensor)
                pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)

                curr_score = self.metric_module.compute(
                    predictions=pred, 
                    references=label
                )['bleu'] * 100

                avg_score += curr_score
                valid_scores[idx: round(curr_score, 2)]

        avg_score = round(avg_score / len(self.dataloader), 2)
        selected

        print('Validation Result')
        print(f"  >> Average BLEU Score: {avg_score:.2f}")
        self.select_data(valid_scores, avg_score)


    def select_data(self, valid_scores, avg_score):
        with open('data/valid.json', 'r') as f:
            valid_data = json.load(f)

        for score, sample in zip(valid_rst, valid_data):

        with open('data/act_train.json', 'w') as f:
            json.dump()


        print(f"  >> Complementary Data Volumn: {}")