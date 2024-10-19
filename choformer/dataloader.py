import torch
import pandas as pd
import tokenizer
import random as random
from esm_utils import ESMWrapper
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, csv_file, config):
        super(ProteinDataset, self).__init__()
        self.config = config
        self.data = pd.read_csv(csv_file)[:5]
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.dna_tokenizer = tokenizer
        #self.dna_tokenizer = AutoTokenizer.from_pretrained(config.decoder_model.dna_model_path)

        self.dna_max_length = config.decoder_model.dna_seq_len

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        l = random.randint(15, 20)
        protein_sequence = self.data.iloc[idx]['protein_sequence'][:20]
        dna_sequence = self.data.iloc[idx]['dna_sequence'][:60]
        exp = self.data.iloc[idx]['expression']

        # protein sequence embeddings with ESM-3
        protein_tokens = self.esm_tokenizer(
            protein_sequence,
            return_tensors='pt',
            max_length=self.config.data.protein_max_length,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            protein_embeddings = self.esm_model(**protein_tokens).last_hidden_state.squeeze(0)

        # labeled protein expression value
        expressions = torch.tensor(exp, dtype=torch.float) # expression value

        # tokenized dna sequence
        dna_tokens = tokenizer.encode([dna_sequence], max_length=self.dna_max_length)

        return protein_embeddings, dna_tokens, expressions

def collate_fn(batch):
    protein_embeddings, dna_tokens, expressions = zip(*batch)

    # prepare only expression values and embeddings as AutoTokenizer already paded dna input ids
    exps = torch.stack(expressions, dim=0)
    padded_embeddings = pad_sequence([torch.tensor(l) for l in protein_embeddings], batch_first=True, padding_value=0)
    
    return padded_embeddings, dna_tokens, exps

def get_dataloaders(config):
    train_dataset = ProteinDataset(config.data.train_data_path, config)
    val_dataset = ProteinDataset(config.data.val_data_path, config)
    test_dataset = ProteinDataset(config.data.test_data_path, config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader