import os, re, json, yaml
from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def process_data(orig_data, volumn=101100):
    min_len = 10 
    max_len = 300
    max_diff = 50

    volumn_cnt = 0
    corpus, processed = [], []
    
    for elem in orig_data:
        temp_dict = dict()
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict['src'] = src
            temp_dict['trg'] = trg
            processed.append(temp_dict)
            corpus.append(src)
            corpus.append(trg)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    return processed




def train_tokenizer():
    corpus_path = 'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(WordPiece(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_config['vocab_size'], 
                               special_tokens=[vocab_config['pad_token'], 
                                               vocab_config['unk_token'],
                                               vocab_config['bos_token'],
                                               vocab_config['eos_token']])

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def main():
    orig_data = load_dataset('wmt14', 'de-en', split='train')['translation']
    processed = process_data(orig_data)

    #Train Tokenizer
    train_tokenizer(task)

    #Save Data
    save_data(task, processed)



if __name__ == '__main__':
        main()    