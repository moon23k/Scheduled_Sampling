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
        input_ids = self.data[idx]['input_ids']
        attention_mask = self.data[idx]['attention_mask']
        labels = self.data[idx]['labels']
        reference = self.data[idx]['reference']
        
        return input_ids, attention_mask, labels, reference




class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        ids_batch, masks_batch, labels_batch, ref_batch = [], [], [], []
        
        for ids, masks, labels, ref in batch:
            ids_batch.append(torch.LongTensor(ids)) 
            masks_batch.append(torch.LongTensor(masks))
            labels_batch.append(torch.LongTensor(labels))
            ref_batch.append([ref])

        ids_batch = self.pad_batch(ids_batch)
        masks_batch = self.pad_batch(masks_batch)
        labels_batch = self.pad_batch(labels_batch)

        return {'input_ids': ids_batch, 
                'attention_mask': masks_batch,
                'labels': labels_batch,
                'references': ref_batch}

    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, split):
    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False,
                      collate_fn=Collator(config.pad_id),
                      pin_memory=True,
                      num_workers=2)