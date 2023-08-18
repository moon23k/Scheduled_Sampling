import torch, operator
from itertools import groupby
from queue import PriorityQueue
from collections import namedtuple




class Generator:
    def __init__(self, config, model, tokenizer):
        super(Generator, self).__init__()
        
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer

        self.beam_size = 4
        self.max_len = config.max_len

        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id
        
        self.Node = namedtuple(
            'Node', 
            ['prev_node', 'pred', 'log_prob', 'length']
        )



    def inference(self):
        print(f'--- Inference Process Started! ---')
        print('[ Type "quit" on user input to stop the Process ]')
        
        while True:
            input_seq = input('\nUser Input Sequence >> ').lower()

            #End Condition
            if input_seq == 'quit':
                print('\n--- Inference Process has terminated! ---')
                break        

            output_seq = self.generate(input_seq, search=self.search)
            print(f"Model Out Sequence >> {output_seq}")       
            


    def generate(self, input_tensor, search='greedy'):
        if isinstance(input_tensor, str):
            input_tensor = torch.LongTensor([[input_tensor]]).to(self.device)

        with torch.no_grad():
            if search == 'greedy':
                generated_ids = self.greedy_search(input_tensor)
            elif search == 'beam':
                generated_ids = self.beam_search(input_tensor)
        
        return self.tokenizer.decode(generated_ids)



    def greedy_search(self, input_tensor):
        output = torch.LongTensor([[self.bos_id]]).to(self.device)

        e_mask = self.model.pad_mask(input_tensor)
        memory = self.model.encoder(input_tensor, e_mask)        

        
        for i in range(1, self.max_len):
            #Masking
            d_mask = self.model.dec_mask(output)

            dec_out = self.model.decoder(output, memory, e_mask, d_mask)            
            logit = self.model.generator(dec_out)
            
            next_token = logit[:, -1].argmax(-1).unsqueeze(0)
            output = torch.cat([output, next_token], dim=1)

            if next_token == self.eos_id:
                break

        return output.squeeze(0).tolist()



    ### Below Methods are for Beam Seach
    def init_nodes(self):
        #returns [ Node, nodes, end_nodes ]
        
        Node = self.Node
        nodes = PriorityQueue()
        start_tensor = [self.bos_id]

        start_node = Node(
            prev_node = None,
            pred = start_tensor,
            log_prob = 0.0,
            length = 0
        )

        for _ in range(self.beam_size):
            nodes.put((0, start_node))
                    
        return Node, nodes, []



    def get_score(self, node, max_repeat=5, min_length=5, alpha=1.2): 
        if not node.log_prob:
            return node.log_prob

        #find max number of consecutively repeated tokens
        repeat = max(
            [sum(1 for token in group if token != self.pad_id) for _, group in groupby(node.pred)]
        )

        repeat_penalty = 0.5 if repeat > max_repeat else 1
        len_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        
        score = node.log_prob / len_penalty
        score = score * repeat_penalty

        return float(score)



    def beam_search(self, input_tensor):
        Node, nodes, end_nodes = self.init_nodes()

        e_mask = self.model.pad_mask(input_tensor)
        memory = self.model.encoder(input_tensor, e_mask)

        for t in range(self.max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]

            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[-1] == self.eos_id and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue

                d_input = torch.LongTensor([curr_node.pred]).to(self.device)
                d_mask = self.model.dec_mask(d_input)

                d_out = self.model.decoder(d_input, memory, e_mask, d_mask)                                           
                out = self.model.generator(d_out)[:, -1]
                
                logits, preds = torch.topk(out, self.beam_size)
                log_probs = torch.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[:, k].item()
                    log_prob = log_probs[:, k].item()
                    
                    next_node = Node(
                        prev_node = curr_node,
                        pred = curr_node.pred + [pred],
                        log_prob = curr_node.log_prob + log_prob,
                        length = curr_node.length + 1
                    )
                    
                    next_score = self.get_score(next_node)                    
                    nodes.put((next_score, next_node))
                        
                if (not t) or (len(end_nodes) == self.beam_size):
                    break

        if len(end_nodes) == 0:
            _, beam_pred = nodes.get()
        else:
            _, beam_pred = sorted(
                end_nodes, 
                key=operator.itemgetter(0), 
                reverse=True
            )[0]
        
        return beam_pred.pred