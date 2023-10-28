import torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.bos_id = config.bos_id
        self.device = config.device
        self.max_len = config.max_len
        self.model_type = config.model_type
        
        self.metric_name = 'BLEU' if self.task == 'translation' else 'ROUGE'
        self.metric_module = evaluate.load(self.metric_name.lower())
        


    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['x'].to(self.device)
                y = self.tokenize(batch['y'])

                pred = self.predict(x)
                pred = self.tokenize(pred)
                
                score += self.evaluate(pred, y)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, self.max_len))
        pred = pred.type(torch.LongTensor).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.pad_mask(x)
        memory = self.model.encoder(x, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_out = self.model.decoder(y, memory, e_mask, None)

            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

            #Early Stop Condition
            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred



    def evaluate(self, pred, label):
        #For NMT Evaluation
        if self.task == 'translation':
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['bleu']
        #For Dialg & Sum Evaluation
        else:
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['rouge2']

        return score * 100