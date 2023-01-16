import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        
        if (config.mode == 'pretrain') & (config.model_type == 'discriminator'):
            self.dis_data = True
        else:
            self.dis_data = False

        self.data = self.load_data(split)


    def load_data(self, split):
        f_name = f'data/dis_{split}.json' if self.dis_data else f'data/gen_{split}.json'
        
        with open(f_name, 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        if self.dis_data:
            input_ids = self.data[idx]['input_ids']
            attention_mask = [1 for _ in range(len(input_ids))]
            labels = self.data[idx]['labels']
            return input_ids, attention_mask, labels

        else:
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


    def gen_collate_fn(batch):
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


    def dis_collate_fn(batch):
        ids_batch, mask_batch, labels_batch = [], [], []

        for input_ids, attention_mask, labels in batch:
            ids_batch.append(torch.LongTensor(input_ids)) 
            mask_batch.append(torch.LongTensor(attention_mask))
            labels_batch.append(labels) 
        
        ids_batch = pad_batch(ids_batch, pad_id)
        mask_batch = pad_batch(mask_batch, pad_id)
        labels_batch = torch.Tensor(labels_batch)

        return {'input_ids': ids_batch,
                'attention_mask': mask_batch,
                'labels': labels_batch}


    dataset = Dataset(config, split)
    collate_fn = dis_collate_fn if dataset.dis_data else gen_collate_fn
    
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)