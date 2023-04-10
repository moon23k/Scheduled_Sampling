from tqdm import tqdm
import torch, time, evaluate


class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader
        self.strategy = config.strategy
        self.device = config.device
        self.bleu = evaluate.load('bleu')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        start_time = time.time()

        with torch.no_grad():
            for batch in tqdm(self.dataloader):   
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                references = batch['references']
                                
                predictions = self.model.generate(input_ids=input_ids, 
                                                  attention_mask=attention_mask,
                                                  max_new_tokens=512, 
                                                  use_cache=True)
                
                predictions = self.tokenizer.batch_decode(predictions, 
                                                          skip_special_tokens=True)

                self.bleu.add_batch(predictions=predictions, 
                                    references=references)    

        bleu_score = self.bleu.compute()['bleu'] * 100

        print('Test Results')
        print(f"  >> BLEU Score: {bleu_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")
    