import torch
import config
import pandas as pd
from esm_utils import ESMWrapper
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, csv_file):
        super(ProteinDataset, self).__init__()
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        exp = self.data.iloc[idx]['expression']

        attn_mask = torch.ones(len(sequence), dtype=torch.float) # attention mask
        expressions = torch.tensor(exp, dtype=torch.float) # expression value

        return sequence, attn_mask, expressions

def collate_fn(batch):
    sequences, attn_masks, expressions = zip(*batch)

    padded_attn_mask = pad_sequence([torch.tensor(m) for m in attn_masks], batch_first=True, padding_value=0)
    padded_attn_mask = (padded_attn_mask != 0).long()

    #padded_exps = pad_sequence([torch.tensor(l) for l in expressions], batch_first=True, padding_value=0)
    exps = torch.stack(expressions, dim=0)
    
    return sequences, padded_attn_mask, exps

def get_dataloaders(config):
    train_dataset = ProteinDataset(config.data.train_data_path)
    val_dataset = ProteinDataset(config.data.val_data_path)
    test_dataset = ProteinDataset(config.data.test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader