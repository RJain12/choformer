import io
import csv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.param_functions import Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F


import sys
sys.path.append('/root/hackathon/choformer')

from inference.transformer import Transformer

# config
num_layers = 8
dim = 384
dim_head = 128
heads = 4

import torch



import torch.nn.functional as F
import torch.nn as nn

# from choformer import tokenizer
# from choformer.model import DNADecoder
from inference.model import DNADecoder

from omegaconf import OmegaConf
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from transformers import AutoModel, AutoTokenizer

config = OmegaConf.load("/root/hackathon/choformer/choformer/config.yaml")

print(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

choformer_ = DNADecoder(config).to(device).eval()
choformer_.load_state_dict(torch.load("/root/hackathon/choformer/choformer/ckpts/best_model.pth", map_location=device))

esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


pretrained_path = "/root/hackathon/expmodel.pt" # replace this with the file from google drive

model = Transformer(num_layers=num_layers, dim=dim, n_classes=1, heads=heads, dim_head=dim_head)

model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
model.to(device).eval()


vocab = {"[CLS]": 0, "[EOS]": 1, "[PAD]": 2}
AGCT = {"A": 0, "G": 1, "C": 2, "T": 3, "N": 4}

def process_codon(seq: str):
    try:
        idx_1 = AGCT[seq[0]]
        idx_2 = AGCT[seq[1]]
        idx_3 = AGCT[seq[2]]
        return 25 * idx_1 + 5 * idx_2 + idx_3 + 3
    except:
        return 1  # return a default index for invalid codons

def embed(seq: str):
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    tokens = [0, *[process_codon(codon) for codon in codons]]
    # Ensure tokens do not exceed max_length
    
    return torch.tensor(tokens).unsqueeze(0)

def run_choexp_inference(dna_sequences: list[str]):
    res = []
    for seq in dna_sequences:
        inputs = embed(seq)
        att = torch.ones_like(inputs)
        
        output = model(inputs.to(device), att.to(device))
        normalized_output = F.sigmoid(output)
        
        # print(f"Expression: {normalized_output.item()}")
        res.append(normalized_output.item())
    return res


def run_choformer_inference(protein_sequences: list[str]):
    print('p', protein_sequences)
    longest_protein_length = max([len(sequence) for sequence in protein_sequences])

    protein_tokens = esm_tokenizer(
        protein_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=longest_protein_length
    ).to(device)


    with torch.no_grad():
        protein_embeddings = esm_model(**protein_tokens).last_hidden_state.squeeze(0)

    print('pp', protein_embeddings)

    outputs = choformer_.generate(protein_embeddings)
    
    return outputs['generated_sequences']


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

class ProteinSequences(BaseModel):
    sequences: list[str]

class GeneratedSequences(BaseModel):
    sequences: list[str]

class DNASequences(BaseModel):
    sequences: list[str]

class ExpressionLevels(BaseModel):
    levels: list[float]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/choexp_inference/", response_model=ExpressionLevels)
async def choexp_inference(dna_sequences: DNASequences):
    try:
        expression_levels = run_choexp_inference(dna_sequences.sequences)
        return ExpressionLevels(levels=expression_levels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/choformer_inference/", response_model=GeneratedSequences)
async def choformer_inference(protein_sequences: ProteinSequences):
    try:
        generated_sequences = run_choformer_inference(protein_sequences.sequences)
        return GeneratedSequences(sequences=generated_sequences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await file.read()
    csv_reader = csv.reader(io.StringIO(content.decode('utf-8')))
    results = []
    for row in csv_reader:
        # result = process_text(row[0])
        result = ''
        results.append(result) 
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7999)