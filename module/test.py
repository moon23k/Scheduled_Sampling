import torch, evaluate


class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.bleu_module = evaluate.load('bleu')


    def test(self):
        self.model.eval()
       
        with torch.no_grad():
            for batch in self.dataloader:   
                
                input_tensors = batch['src'].to(self.device)
                labels = batch['trg'].tolist()
                                
                predictions = self.model.generate(input_tensor)
                predictions = self.tokenizer.batch_decode(predictions)

                bleu_score = self.bleu_module.compute(
                    predictions=predictions, 
                    references=references
                )['bleu'] * 100


        print('Test Results')
        print(f"  >> BLEU Score: {bleu_score:.2f}")
    