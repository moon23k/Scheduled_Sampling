import os, re, json, yaml, argparse
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




#NMT
def process_translation_data(data_volumn):
    #load original dataset
    nmt_data = load_dataset('wmt14', 'de-en', split='train')['translation']
    
    min_len = 10 
    max_len = 300
    max_diff = 50
    volumn_cnt = 0

    corpus, processed = [], []
    
    for elem in nmt_data:
        temp_dict = dict()
        x, y = elem['en'].strip().lower(), elem['de'].strip().lower()
        x_len, y_len = len(x), len(y)

        #Filtering Conditions
        min_condition = (x_len >= min_len) & (y_len >= min_len)
        max_condition = (x_len <= max_len) & (y_len <= max_len)
        dif_condition = abs(x_len - y_len) < max_diff

        if max_condition & min_condition & dif_condition:
            corpus.append(x)
            corpus.append(y)
            processed.append({'x': x, 'y':y})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    #Save Corpus
    with open('data/translation/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    return processed 



#Dialog
def process_dialogue_data(data_volumn):
    volumn_cnt = 0
    corpus, processed = [], []

    #Load original Datasets
    daily_data = load_dataset('daily_dialog')


    #Daily-Dialogue Dataset Processing
    x_data, y_data = [], []
    for split in ['train', 'validation', 'test']:
        for dial in daily_data[split]['dialog']:
            dial_list = []
            dial_turns = len(dial)

            if max([len(d) for d in dial]) > 300:
                continue
            
            for uttr in dial:
                _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
                _uttr = re.sub(r'([’])\s+', r'\1', _uttr).strip().lower()
                if len(_uttr) > 300:
                    break
                dial_list.append(_uttr)
            
            if dial_turns < 2:
                continue

            elif dial_turns == 2:
                x_data.append(dial_list[0])
                y_data.append(dial_list[1])
                continue  #To avoid duplicate on below condition

            #Incase of dial_turns is even
            elif dial_turns % 2 == 0:
                x_data.extend(dial_list[0::2])
                y_data.extend(dial_list[1::2])

                x_data.extend(dial_list[1:-1:2])
                y_data.extend(dial_list[2::2])
            
            #Incase of dial_turns is odds
            elif dial_turns % 2 == 1:
                x_data.extend(dial_list[0:-1:2])
                y_data.extend(dial_list[1::2])
                
                x_data.extend(dial_list[1::2])
                y_data.extend(dial_list[2::2])   


    assert len(x_data) == len(y_data)
    
    for x, y in zip(x_data, y_data):        
        corpus.append(x)
        corpus.append(y)
        processed.append({'x': x, 'y': y})

        volumn_cnt += 1
        if volumn_cnt == data_volumn:
            break        

    
    #Save Corpus
    with open('data/dialogue/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))    

    return processed



#Summarization
def process_summarization_data(data_volumn):    
    volumn_cnt = 0
    corpus, processed = [], []
    min_len, max_len = 500, 2300

    #Load Original Dataset
    cnn_data = load_dataset('cnn_dailymail', '3.0.0')

    for split in ['train', 'validation', 'test']:
        for elem in cnn_data[split]:

            x, y = elem['article'], elem['highlights']

            if min_len < len(x) < max_len:
                if len(y) < min_len:
                    
                    #Lowercase
                    x, y = x.lower(), y.lower()

                    #Remove unnecessary characters in trg sequence
                    y = re.sub(r'\n', ' ', y)                 #remove \n
                    y = re.sub(r"\s([.](?:\s|$))", r'\1', y)  #remove whitespace in front of dot

                    processed.append({'x': x, 'y': y})
                    corpus.append(x)
                    corpus.append(y)

                    #End Condition
                    volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    with open('data/summarization/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))
    
    return processed           



def train_tokenizer(task):
    corpus_path = f'data/{task}/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"data/{task}/tokenizer.json")



def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5100], data_obj[-5100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{task}/{key}.json')




def main(task):
    #Prerequisite
    os.makedirs(f'data/{task}', exist_ok=True)

    #PreProcess Data
    data_volumn = 55100
    if task == 'translation':
        processed = process_translation_data(data_volumn)
    elif task == 'dialogue':
        processed = process_dialogue_data(data_volumn)
    elif task == 'summarization':
        processed = process_summarization_data(data_volumn)        

    #Train Tokenizer
    train_tokenizer(task)

    #Save Data
    save_data(task, processed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    
    args = parser.parse_args()
    assert args.task in ['all', 'translation', 'dialogue', 'summarization']
    
    if args.task == 'all':
        for task in ['translation', 'dialogue', 'summarization']:
            main(task)
    else: 
        main(args.task)    