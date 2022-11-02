import os, json
import sentencepiece as spm
from datasets import load_dataset



def concat_data(train, valid, test):
    src, trg = [], []
    for split in (train, valid, test):
        for elem in split:
            src.append(elem['src'])
            trg.append(elem['trg'])

    with open('data/src.txt', 'w') as f:
        f.write('\n'.join(src))
    with open('data/trg.txt', 'w') as f:
        f.write('\n'.join(trg))



def build_vocab():
    for file in ["src", 'trg']:        
        assert os.path.exists(f'data/{file}.txt')
        opt = f"--input=data/{file}.txt\
                --model_prefix=data/{file}_spm\
                --vocab_size=10000\
                --character_coverage=1\
                --model_type=bpe\
                --pad_id=0 --pad_piece='[PAD]'\
                --unk_id=1 --unk_piece='[UNK]'\
                --bos_id=2 --bos_piece='[BOS]'\
                --eos_id=3 --eos_piece='[EOS]'"
        spm.SentencePieceTrainer.Train(opt)
        os.remove(f'data/{file}.txt')



def tokenize_datasets(train, valid, test, src_tokenizer, trg_tokenizer):
    tokenized_data = []

    for split in (train, valid, test):
        split_tokenized = []
        
        for elem in split:
            temp_dict = dict()
            
            temp_dict['src'] = src_tokenizer.EncodeAsIds(elem['src'])
            temp_dict['trg'] = trg_tokenizer.EncodeAsIds(elem['trg'])
            
            split_tokenized.append(temp_dict)
        
        tokenized_data.append(split_tokenized)
    
    return tokenized_data



def load_tokenizers():
    tokenizers = []
    for lang in ['src', 'trg']:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(f'data/{lang}_spm.model')
        tokenizer.SetEncodeExtraOptions('bos:eos')    
        tokenizers.append(tokenizer)
    return tokenizers




def save_datasets(train, valid, test):
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}
    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)



def filter_dataset(data, min_len=10, max_len=300):
    filtered = []
    for elem in data:
        temp_dict = dict()
        src_len, trg_len = len(elem['en']), len(elem['de'])
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        min_condition = (src_len >= min_len) & (trg_len >= min_len)

        if max_condition & min_condition:
            temp_dict['src'] = elem['en']
            temp_dict['trg'] = elem['de']
            filtered.append(temp_dict)

    return filtered



def main(downsize=True, sort=True):
    #Download datasets
    train = load_dataset('wmt14', 'de-en', split='train')['translation']
    valid = load_dataset('wmt14', 'de-en', split='validation')['translation']
    test = load_dataset('wmt14', 'de-en', split='test')['translation']

    train = filter_dataset(train)
    valid = filter_dataset(valid)
    test = filter_dataset(test)

    if downsize:
        train = train[::100]

    if sort:
        train = sorted(train, key=lambda x: len(x['src']))
        valid = sorted(valid, key=lambda x: len(x['src']))
        test = sorted(test, key=lambda x: len(x['src']))

    #create concat
    concat_data(train, valid, test)
    build_vocab()
    src_tokenizer, trg_tokenizer = load_tokenizers()
    
    train, valid, test = tokenize_datasets(train, valid, test, src_tokenizer, trg_tokenizer)
    save_datasets(train, valid, test)



if __name__ == '__main__':
    main()
    assert os.path.exists(f'data/train.json')
    assert os.path.exists(f'data/valid.json')
    assert os.path.exists(f'data/test.json')