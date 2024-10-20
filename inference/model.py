import torch
from . import tokenizer
import torch.nn.functional as F
import torch.nn as nn
# from esm_utils import ESMWrapper
#from lstm_train import LSTMClassifier
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

class DNADecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get parameters from config
        self.config = config
        protein_embedding_size = config.decoder_model.protein_embedding_size
        decoder_size = config.decoder_model.decoder_size
        dna_seq_len = config.decoder_model.dna_seq_len
        dna_model_path = config.decoder_model.dna_model_path
        num_layers = config.decoder_model.layers
        nhead = config.decoder_model.heads
        dropout = config.decoder_model.dropout

        # Custom 3-mer tokenizer
        self.dna_tokenizer = tokenizer
        self.dna_vocab_size = config.data.tokenizer_length
        self.dna_seq_len = dna_seq_len

        # Simple MLP for mapping protein embeddings to decoder size (in between encoder/decoder stack)
        self.protein_to_deocder = nn.Sequential(
            nn.Linear(protein_embedding_size, decoder_size*2),
            nn.ReLU(),
            nn.Linear(decoder_size*2, decoder_size)
        )
        
        # Lookup table for DNA tokens to dense embeddings
        self.dna_embeddings = nn.Embedding(
            num_embeddings=self.dna_vocab_size,
            embedding_dim=decoder_size,
            padding_idx=self.dna_tokenizer.vocab['[PAD]']
        )

        # Injecting positional information into DNA embeddings
        self.pos_embedding = PositionalEmbedding(
            d_model=decoder_size,
            max_len=dna_seq_len,
            dropout=dropout
        )
        self.pos_embedding = PositionalEmbedding(d_model=decoder_size, max_len=dna_seq_len, dropout=dropout)

        # Actual Transformer decoder stack for generation
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # Mapping decoder size to the dna vocab size, creating a probability distribution over each token
        self.decoder_to_dna_vocab = nn.Linear(decoder_size, self.dna_vocab_size)
        self.register_buffer("mask", self._generate_causal_mask(size=dna_seq_len))

        # Simple MLP to predict expression
        self.predict_expression = nn.Sequential(
            nn.Linear(decoder_size, decoder_size // 2),
            nn.ReLU(),
            nn.Linear(decoder_size // 2, 1),
            nn.Sigmoid()
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.dna_tokenizer.vocab['[PAD]'])
    
    def _generate_causal_mask(self, size):
        """
        Creates a causal mask to prevent the model from "peeking" forward
        Args:
            - size (int): size of mask to be created
        
        Returns:
            - mask (torch.Tensor): causal mask, an upper triangular matrix of size [seq_len x seq_len]
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _generate_padding_mask(self, input_token):
        pad_token = self.dna_tokenizer.vocab['[PAD]']
        return (input_token == pad_token)

    def forward(self, protein_embeddings, labels=None):
        bsz, seq_len = protein_embeddings.size(0), self.dna_seq_len

        # map protein embeddings to decoder dimension size
        protein_embeddings = self.protein_to_deocder(protein_embeddings)
        protein_embeddings = self.pos_embedding(protein_embeddings)

        if labels is not None:
            seq_len = labels.size(1)
            padding_mask = self._generate_padding_mask(labels).to(protein_embeddings.device)  # [bsz x seq_len]
            causual_mask = self._generate_causal_mask(seq_len).to(protein_embeddings.device)  # [seq_len x seq_len]
            
            decoder_out = self.decoder(tgt=protein_embeddings.float(),
                                    memory=protein_embeddings.float(),
                                    tgt_mask=causual_mask,
                                    # tgt_is_causal=True,
                                    tgt_key_padding_mask=padding_mask)
            
            logits = self.decoder_to_dna_vocab(decoder_out)
            loss = self.loss_fn(logits.view(-1, self.dna_vocab_size), labels.view(-1))

            return {
                "loss": loss,
                "logits": logits
            }
        
        else:
            # If labels are not provided, skip the loss computation
            causual_mask = self._generate_causal_mask(seq_len).to(protein_embeddings.device)
            decoder_out = self.decoder(tgt=protein_embeddings.float(),
                                    memory=protein_embeddings.float(),
                                    tgt_mask=causual_mask,
                                    # tgt_is_causal=True
                                    )
            logits = self.decoder_to_dna_vocab(decoder_out)

            return {
                "logits": logits
            }
        
    def generate(self, protein_embeddings, labels=None, max_length=None):
        """"Generation of all tokens simultaneously or autoregressively"""
        bsz = protein_embeddings.size(0)
        
        # Handle case when labels are provided for evaluation
        if labels is not None:
            sequence_lengths = (labels != self.dna_tokenizer.vocab['[PAD]']).sum(dim=1)
            outputs = self.forward(protein_embeddings, labels)  # Call forward to get logits and optionally loss
            logits = outputs['logits']
            all_gen_tokens = []

            for i in range(bsz):
                seq_len = sequence_lengths[i].item()
                
                # Select the most probable token for each position
                sequence_logits = logits[i, :seq_len, :]
                generated_tokens = torch.argmax(sequence_logits, dim=-1)
                all_gen_tokens.append(generated_tokens)

            # Decode token IDs to codons
            generated_sequences = [self.dna_tokenizer.decode(gen_tokens.unsqueeze(0))[0] for gen_tokens in all_gen_tokens]

            hamming = (labels[0][:all_gen_tokens[0].size(0)] != all_gen_tokens[0]).sum() / all_gen_tokens[0].size(0)

            return {
                "logits": logits,
                "generated_sequences": generated_sequences,
                "loss": outputs.get('loss'),  # The loss might not be computed if labels=None
                "hamming": hamming
            }

        # Handle case during inference when only protein embeddings are provided
        else:
            outputs = self.forward(protein_embeddings)  # Call forward without labels
            logits = outputs['logits']

            # Optionally generate tokens autoregressively or use logits directly
            all_gen_tokens = []
            for i in range(bsz):
                generated_tokens = torch.argmax(logits[i, :, :], dim=-1)
                all_gen_tokens.append(generated_tokens)

            # Decode token IDs to codons
            generated_sequences = [self.dna_tokenizer.decode(gen_tokens.unsqueeze(0))[0] for gen_tokens in all_gen_tokens]

            return {
                "logits": logits,
                "generated_sequences": generated_sequences
            }

    
    # def generate(self, protein_embeddings, labels, max_length=None):
    #     """Standard autoregressive generation (next token prediction)"""
    #     bsz = protein_embeddings.size(0)
    #     sequence_lengths = (labels != self.dna_tokenizer.vocab['[PAD]']).sum(dim=1)

    #     all_gen_tokens = []
    #     for i in range(bsz):
    #         generated_tokens = torch.full((1, 1), self.dna_tokenizer.vocab['[CLS]'], dtype=torch.long).to(protein_embeddings.device)
    #         seq_len = sequence_lengths[i].item()-1

    #         # Generate tokens equivalent to the sequence length
    #         for _ in range(seq_len):
    #             outputs = self.forward(protein_embeddings=protein_embeddings[i:i+1], labels=generated_tokens)
    #             next_token_logits = outputs['logits'][:, -1, :]
    #             padding_mask = self._generate_padding_mask(generated_tokens).to(protein_embeddings.device)
    #             next_token_logits.masked_fill_(padding_mask[:, -1].unsqueeze(1), float('-inf'))

    #             # Select the next most probable token
    #             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    #             generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

    #         all_gen_tokens.append(generated_tokens[:, 1:])
        
    #     # Decode token IDs to codons
    #     generated_sequences = [self.dna_tokenizer.decode(gen_tokens)[0] for gen_tokens in all_gen_tokens]

    #     return {
    #         "logits": outputs['logits'],
    #         "generated_sequences": generated_sequences,
    #         "loss": outputs['loss']
    #         #"predicted_expression": predicted_expression
    #     }