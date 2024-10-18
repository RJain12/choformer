import torch
import math
import config
import torch.nn.functional as F
import torch.nn as nn
from esm_utils import ESMWrapper
from lstm_train import LSTMClassifier
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from transformers import AutoModel, AutoTokenizer


# class ExpressionLoss(nn.Module):
#     def __init__(self, lstm_ckpt_path):
#         super().__init__()
#         self.model = LSTMClassifier(**config.lstm_model) # init model class
#         self.model.load_state_dict(torch.load(lstm_ckpt_path)) # load model weights
#         self.model.eval()
#         for param in self.model.parameters:
#             param.requires_grad=False 

#         self.alpha = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
#     def forward(self, og_seq, optim_seq):
#         with torch.no_grad():
#             og_exp = self.model(og_seq)
#             optim_exp = self.model(optim_seq)

#         exp_loss = self.alpha * (1 - og_exp) * (optim_exp - og_exp)
#         return torch.abs(exp_loss).mean()

# class CrossEntropyLoss:
#     def __init__(self, dna_model_path):
#         super().__init__()
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path)
#         self.dna_bert = AutoModel.from_pretrained(dna_model_path)
#         for param in self.dna_bert.parameters:
#             param.requires_grad=False

#         self.beta = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
#     def forward(self, og_seq,  optim_seq):
#         og_ids = self.dna_tokenizer(og_seq, return_tensors='pt', padding=True, truncation=True)['input_ids']
#         optim_ids = self.dna_tokenizer(optim_seq, return_tensors='pt', padding=True, truncation=True)['input_ids']

#         with torch.no_grad():
#             og_logits = self.dna_bert(og_ids).last_hidden_state.squeeze(0)
#             optim_logits = self.dna_bert(optim_ids).last_hidden_state.squeeze(0)

#         return self.beta * self.ce_loss(og_logits.view(-1, og_logits.size(-1)), optim_logits.view(-1))

# class CosineLoss:
#     def __init__(self, dna_model_path):
#         super().__init__()
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path)
#         self.dna_bert = AutoModel.from_pretrained(dna_model_path)
#         for param in self.dna_bert.parameters:
#             param.requires_grad=False

#         self.cosine_loss = nn.CosineEmbeddingLoss()
    
#     def forward(self, og_seq,  optim_seq):
#         og_ids = self.dna_tokenizer(og_seq, return_tensors='pt', padding=True, truncation=True)['input_ids']
#         optim_ids = self.dna_tokenizer(optim_seq, return_tensors='pt', padding=True, truncation=True)['input_ids']

#         with torch.no_grad():
#             og_logits = self.dna_bert(og_ids).last_hidden_state.squeeze(0)
#             optim_logits = self.dna_bert(optim_ids).last_hidden_state.squeeze(0)
        
#         target = torch.ones(og_logits.size(0), device=og_logits.device)  # All targets are 1

#         # Calculate cosine similarity loss between the embeddings
#         return self.cosine_loss(og_logits, optim_logits, target)


# class ProteinEncoder(nn.Module):
#     def __init__(self, esm_model_path, use_local_device: bool):
#         super().__init__()
#         self.esm_api = ESMWrapper(model_name=esm_model_path, use_local=use_local_device)
        
#     def forward(self, protein_sequence: str):
#         protein_embedding = self.esm_api(protein_sequence)
#         return protein_embedding

class DNADecoder(nn.Module):
    def __init__(self, protein_embedding_size, decoder_size, dna_seq_len,
                 dna_vocab_size, dna_model_path,
                 num_layers, nhead, dropout):
        super().__init__()
        # init dna bert tokenizer
        self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path)

        # length of dna sequence to be optimized
        self.dna_seq_len = dna_seq_len
        # number of unique dna bert tokens
        self.dna_vocab_size = dna_vocab_size

        # simple MLP for mapping protein embeddings to decoder size (in between encoder/decoder stack)
        self.protein_to_deocder = nn.Sequential([
            nn.Linear(protein_embedding_size, decoder_size*2),
            nn.ReLU(),
            nn.Linear(decoder_size*2, decoder_size)
        ])
        
        # dna embeddings and positional information
        self.dna_embeddings = nn.Embedding(dna_vocab_size, decoder_size)
        self.pos_embedding = PositionalEmbedding(d_model=decoder_size, max_len=dna_seq_len, dropout=dropout)

        # decoder stack for the choformer model
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # Mapping decoder size to the dna vocab size – create probability distribution over each token
        self.decoder_to_dna_vocab = nn.Linear(decoder_size, dna_vocab_size)
        self.register_buffer("mask", self._generate_square_mask(size=dna_seq_len))

        self.loss_fn = nn.CrossEntropyLoss()
    
    def _generate_square_mask(self, size):
        """Causual masking to pay attention to future tokens"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _generate_padding_mask(self, input_token):
        pad_token = self.dna_tokenizer.pad_token_id
        return (input_token != pad_token).unsqueeze(1)


    def forward(self, protein_embeddings, input_ids, labels=None):
        seq_len = input_ids.size(1)

        # map protein embeddings to decoder dimension size
        protein_embeddings = self.protein_to_deocder(protein_embeddings)

        dna_embeddings = self.dna_embeddings(input_ids)
        dna_embeddings = self.pos_embedding(dna_embeddings)

        padding_mask = self._generate_padding_mask(input_ids)
        causual_mask = self._generate_square_mask(seq_len).to(protein_embeddings.device)

        decoder_out = self.decoder(dna_embeddings,
                                   protein_embeddings,
                                   tgt_mask=causual_mask,
                                   tgt_key_padding_mask=~padding_mask.squeeze(1))
        
        logits = self.decoder_to_dna_vocab(decoder_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.dna_vocab_size), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        } 
    
    def generate(self, protein_embeddings, max_length=None):
        bsz = protein_embeddings.size(0)
        input_ids = torch.full((bsz, 1), self.dna_tokenizer.cls_token_id, dtype=torch.long).to(protein_embeddings.device)

        for _ in range(max_length - 1):
            outputs = self.forward(protein_embeddings=protein_embeddings, input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        generated_sequences = self.dna_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return {
            "logits": outputs.logits,
            "generated_sequences": generated_sequences,
            "loss": outputs.loss
        }

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
