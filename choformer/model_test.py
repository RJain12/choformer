import torch
import sys
import torch.nn as nn
from omegaconf import OmegaConf
from dataloader import get_dataloaders
from tqdm import tqdm
from esm_utils import ESMWrapper
from model import DNADecoder
from torch.nn.utils.rnn import pad_sequence


def main(config_path):
    config = OmegaConf.load(config_path)

    # instantiate ESM and CHOFormer models
    #esm_model = ESMWrapper()
    choformer_model = DNADecoder(config)
    
    # protein_tokens = self.esm_tokenizer(protein_sequence, return_tensors="pt")
    # with torch.no_grad():
    #     protein_embeddings = self.esm_model(**protein_tokens).last_hidden_state.squeeze(0)

    # # labeled protein expression value
    # expressions = torch.tensor(exp, dtype=torch.float) # expression value

    # # tokenized dna sequence
    # dna_tokens = self.dna_tokenizer(
    #     dna_sequence,
    #     return_tensors='pt',
    #     max_length=self.config.decoder_model.dna_seq_len,
    #     padding=True,
    #     truncation=True
    # )
    
    
