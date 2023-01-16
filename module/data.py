import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        en_ids = self.data[idx]['en_ids']
        en_mask = self.data[idx]['en_mask']

        de_ids = self.data[idx]['de_ids']
        de_mask = self.data[idx]['de_mask']
        
        return en_ids, en_mask, de_ids, de_mask


def pad_batch(batch_list, pad_id):
    return pad_sequence(batch_list,
                        batch_first=True,
                        padding_value=pad_id)


def load_dataloader(config, split):
    global pad_id
    pad_id = config.pad_id    

    def collate_fn(batch):
        en_ids_batch, en_mask_batch = [], []
        de_ids_batch, de_mask_batch = [], []

        for en_ids, en_mask, de_ids, de_mask in batch:

            en_ids_batch.append(torch.LongTensor(en_ids))
            en_mask_batch.append(torch.LongTensor(en_mask))

            de_ids_batch.append(torch.LongTensor(de_ids))
            de_mask_batch.append(torch.LongTensor(de_mask))

        en_ids_batch = pad_batch(en_ids_batch, pad_id)
        en_mask_batch = pad_batch(en_mask_batch, pad_id)
        
        de_ids_batch = pad_batch(de_ids_batch, pad_id)
        de_mask_batch = pad_batch(de_mask_batch, pad_id)

        return {'en_ids': en_ids_batch, 
                'en_mask': en_mask_batch,
                'de_ids': de_ids_batch, 
                'de_mask': de_mask_batch}

    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)