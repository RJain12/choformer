import torch
import math
import config
import torch.nn as nn
from esm_utils import ESMWrapper
from lstm_train import LSTMClassifier
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from transformers import AutoModel, AutoTokenizer


class ExpressionLoss:
    def __init__(self, lstm_ckpt_path):
        self.model = LSTMClassifier(**config.lstm_model) # init model class
        self.model.load_state_dict(torch.load(lstm_ckpt_path)) # load model weights
        self.model.eval()
        for param in self.model.parameters:
            param.requires_grad=False 

        self.alpha = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
    def compute_expression_loss(self, original_sequence, optimized_sequence):
        with torch.no_grad():
            og_exp = self.model(original_sequence)
            optim_exp = self.model(optimized_sequence)

        exp_loss = self.alpha * (1 - self.model(original_sequence)) * (self.model(optim_exp) - self.model(og_exp))
        return math.abs(exp_loss)

class CrossEntropyLoss:
    def __init__(self, dna_model_path):
        self.ce_loss = nn.CrossEntropyLoss()
        self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path)
        self.dna_bert = AutoModel.from_pretrained(dna_model_path)
        for param in self.dna_bert.parameters:
            param.requires_grad=False

        self.beta = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
    def compute_ce_loss(self, og_seq,  optim_seq):
        og_ids = self.dna_tokenizer(og_seq)['input_ids']
        optim_ids = self.dna_tokenizer(optim_seq)['input_ids']

        with torch.no_grad():
            og_embed = self.dna_bert(og_ids)
            optim_embed = self.dna_bert(optim_ids)

        return self.beta * self.ce_loss(og_embed, optim_embed)


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


class ProteinEncoder(nn.Module):
    def __init__(self, esm_model_path, use_local_device: bool):
        super().__init__()
        self.esm_api = ESMWrapper(model_name=esm_model_path, use_local=use_local_device)
        
    def forward(self, protein_sequence: str):
        protein_embedding = self.esm_api(protein_sequence)
        return protein_embedding

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

        # Actual decoder stack for the choformer model
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # Mapping decoder size to the dna vocab size – create probability distribution over each token
        self.decoder_to_dna_vocab = nn.Linear(decoder_size, dna_vocab_size)
        self.register_buffer("mask", self.generate_square_mask(size=dna_seq_len))
    
    def generate_square_mask(self, size):
        """Causual masking to pay attention to future tokens"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_padding_mask(self, input_token):
        pad_token = self.dna_tokenizer.pad_token_id
        return (input_token != pad_token).unsqueeze(1)


    def forward(self, protein_embeddings, input_token):
        bsz = protein_embeddings.size(0)

        # map protein embeddings to decoder dimension size
        protein_embeddings = self.protein_to_deocder(protein_embeddings)
        protein_embeddings = protein_embeddings.unsqueeze(1).repeat(1, self.max_dna_seq_len, 1)
        
        outputs = torch.zeros(bsz, self.max_dna_seq_len, self.dna_vocab_size).to(protein_embeddings.device)

        for t in range(self.max_dna_seq_len):
            dna_embeddings = self.dna_embeddings(input_token).unsqueeze(1)
            dna_embeddings = self.pos_embedding(dna_embeddings)

            padding_mask = self.generate_padding_mask(input_token)
            target_mask = self.mask[:t+1, :t+1].to(protein_embeddings.device)
            final_mask = target_mask + padding_mask

            decoder_out = self.decoder(dna_embeddings,
                                       protein_embeddings,
                                       tgt_mask=final_mask)

            # greedy decoding
            vocab_out = self.decoder_to_dna_vocab(decoder_out[-1])
            vocab_probabilities = torch.softmax(vocab_out, dim=-1)
            outputs[:, t, :] = vocab_probabilities

            input_token = torch.argmax(vocab_probabilities, dim=-1) # preapre tokens for next timestep
        
        return outputs

    def generate(self, protein_embeddings, max_len):
        input_token = torch.zeros(protein_embeddings.size(0), dtype=torch.long).to(protein_embeddings.device)
        
        outputs = []
        for _ in range(max_len):
            token_probabilities = self.forward(protein_embeddings, input_token)[:, -1, :] # get updated token probabilities
            generated_token = torch.argmax(token_probabilities, dim=-1) # select most likely token
            outputs.append(generated_token.unsqueeze(1))
            input_token = generated_token
        
        # collect all the outputs in a tensor for quick batch decoding
        outputs = torch.cat(outputs, dim=-1)
        generated_sequences = self.dna_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_sequences